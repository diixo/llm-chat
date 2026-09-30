"""Background training: python -m app_main.ml.worker RUN_DIRECTORY."""
import logging
import math
from pathlib import Path
import random
import sys
import time
import traceback
from functools import partial

from .jobs import read_json, write_json

logger = logging.getLogger(__name__)


def publish_status(path, status):
    # Record the final outcome separately: a locked progress file must not hide
    # a successfully saved model or replace the original training error.
    destinations = [path / "status.json"]
    if status["state"] in {"completed", "stopped", "failed"}:
        destinations.insert(0, path / "final-status.json")
    for destination in destinations:
        try:
            write_json(destination, status)
        except OSError:
            logger.warning("Could not update %s; training state: %s", destination, status["state"], exc_info=True)


class Stopped(Exception):
    pass


def train(path):
    # Wait until the launcher has persisted the process identity.
    for _ in range(100):
        if (path / "launched").exists():
            break
        time.sleep(0.1)
    status = read_json(path / "status.json")
    config = read_json(path / "config.json")

    def update(state, message, **values):
        status.update(state=state, message=message, **values)
        print(message, flush=True)
        publish_status(path, status)

    def check_stop():
        if (path / "stop").exists():
            raise Stopped()

    try:
        if not (path / "launched").exists():
            raise RuntimeError("Launcher did not register the training process.")
        check_stop()
        update("loading", "Loading GPT-2 124M and tokenizer (first run downloads weights)…")
        import torch
        from torch.utils.data import DataLoader
        from transformers import GPT2DoubleHeadsModel, GPT2Tokenizer
        from .data import SPECIAL_TOKENS, PersonaDataset, collate

        random.seed(42)
        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_num_threads(min(4, torch.get_num_threads()))
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2", cache_dir=config["cache_dir"])
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
        check_stop()
        model = GPT2DoubleHeadsModel.from_pretrained(
            "openai-community/gpt2", cache_dir=config["cache_dir"], use_safetensors=True
        )
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.to(device)
        check_stop()
        prefix = Path(config["data_dir"]) / "personachat_truecased_full"
        train_paths = [f"{prefix}_train.part{part}.json" for part in range(1, 4)]
        train_data = PersonaDataset(train_paths, tokenizer, config["max_length"])
        valid_data = PersonaDataset(str(prefix) + "_valid.json", tokenizer, config["max_length"])
        batcher = partial(collate, pad_id=tokenizer.pad_token_id)
        loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, collate_fn=batcher)
        validation = DataLoader(valid_data, batch_size=1, collate_fn=batcher)
        # Keep the configured learning rate constant across every step and epoch.
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
        total = len(loader) * config["epochs"]
        step = 0
        for epoch in range(config["epochs"]):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulation = config["gradient_accumulation"]
            for index, batch in enumerate(loader):
                check_stop()
                batch = {key: value.to(device) for key, value in batch.items()}
                # Correct scaling also for the final incomplete accumulation group.
                group_size = min(accumulation, len(loader) - (index // accumulation) * accumulation)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                    output = model(**batch)
                    loss = output.loss + output.mc_loss
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss. Try a lower learning rate.")
                scaler.scale(loss / group_size).backward()
                del output
                if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                step += 1
                if step == 1 or step % 10 == 0 or index + 1 == len(loader):
                    update("training", f"Epoch {epoch + 1}/{config['epochs']} · batch {index + 1}/{len(loader)}",
                           epoch=epoch + 1, step=step, total=total, progress=round(100 * step / total, 1),
                           loss=round(loss.item(), 4), device=device)
            model.eval()
            nll, tokens, correct, examples = 0.0, 0, 0, 0
            update("validating", f"Validating epoch {epoch + 1}…")
            with torch.inference_mode():
                for index, batch in enumerate(validation):
                    check_stop()
                    batch = {key: value.to(device) for key, value in batch.items()}
                    with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                        output = model(**batch)
                    count = (batch["labels"][..., 1:] != -100).sum().item()
                    if not torch.isfinite(output.loss):
                        raise ValueError("Non-finite validation loss. Try a lower learning rate.")
                    nll += output.loss.item() * count
                    tokens += count
                    correct += (output.mc_logits.argmax(-1) == batch["mc_labels"]).sum().item()
                    examples += len(batch["mc_labels"])
                    del output
                    if index % 100 == 0:
                        update("validating", f"Validation {index + 1}/{len(validation)}")
            metrics = {"validation_loss": round(nll / tokens, 4),
                       "perplexity": round(math.exp(min(nll / tokens, 50)), 3),
                       "candidate_accuracy": round(correct / examples, 4)}
            write_json(path / f"metrics-epoch-{epoch + 1}.json", metrics)
            update("validating", f"Epoch {epoch + 1}: {metrics}", **metrics)
        check_stop()
        update("saving", "Saving trained model and tokenizer…")
        model.save_pretrained(path / "model", safe_serialization=True)
        tokenizer.save_pretrained(path / "model")
        check_stop()
        update("completed", "Model is ready. Open Dialogue to chat.", progress=100)
    except Stopped:
        update("stopped", "Training stopped. No final model was published.")
    except Exception as exc:
        traceback.print_exc()
        message = str(exc)
        if isinstance(exc, TypeError) and "NoneType" in message and status["state"] == "loading":
            message = "Could not load GPT-2 files. Check Hugging Face access and the local model cache."
        update("failed", message)


if __name__ == "__main__":
    train(Path(sys.argv[1]))
