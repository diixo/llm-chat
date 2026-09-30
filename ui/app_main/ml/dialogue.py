"""CPU inference keeps the GPU available for training."""
import threading

from .data import encode_context

_lock = threading.Lock()
_cached = None


def reply(path, persona, history):
    global _cached
    import torch
    from transformers import GPT2DoubleHeadsModel, GPT2Tokenizer

    with _lock:
        torch.set_num_threads(min(4, torch.get_num_threads()))
        if _cached is None or _cached[0] != str(path):
            tokenizer = GPT2Tokenizer.from_pretrained(path, local_files_only=True)
            model = GPT2DoubleHeadsModel.from_pretrained(path, local_files_only=True)
            model.eval()
            _cached = (str(path), tokenizer, model)
        _, tokenizer, model = _cached
        ids, types = encode_context(tokenizer, persona, history, min(512, model.config.n_positions - 64))
        generated = []
        with torch.inference_mode():
            input_ids = torch.tensor([ids])
            token_types = torch.tensor([types])
            past_key_values = None
            # Explicit token types retain the same speaker encoding as training.
            for _ in range(64):
                output = model(input_ids=input_ids, token_type_ids=token_types,
                               past_key_values=past_key_values, use_cache=True)
                past_key_values = output.past_key_values
                logits = output.logits[0, -1].float() / 0.7
                for token in tokenizer.all_special_ids:
                    if token != tokenizer.eos_token_id:
                        logits[token] = float("-inf")
                if not generated:
                    logits[tokenizer.eos_token_id] = float("-inf")
                values, indices = torch.topk(logits, min(40, logits.size(-1)))
                probabilities = torch.softmax(values, dim=-1)
                mask = probabilities.cumsum(-1) - probabilities > 0.9
                probabilities[mask] = 0
                token = indices[torch.multinomial(probabilities, 1)].item()
                if token == tokenizer.eos_token_id:
                    break
                generated.append(token)
                input_ids = torch.tensor([[token]])
                token_types = token_types[:, -1:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()
