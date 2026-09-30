"""PersonaChat inputs shared by training and inference."""
import json
from os import PathLike

SPECIAL_TOKENS = {
    "bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>"],
}


def encode_context(tokenizer, persona, history, budget):
    speaker1, speaker2 = tokenizer.convert_tokens_to_ids(["<speaker1>", "<speaker2>"])
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    prefix = [tokenizer.bos_token_id] + encode(" ".join(persona))[:budget // 3]
    # The final history utterance is always from the user; the reply is speaker2.
    # For odd-length histories, zero-based indices 1, 3, 5, ... are the assistant.
    turns = []
    for index, text in enumerate(history):
        speaker = speaker1 if (len(history) - index) % 2 else speaker2
        turns.append(([speaker] + encode(text), speaker))
    remaining = budget - len(prefix) - 1
    selected = []
    for tokens, speaker in reversed(turns):
        if remaining <= 1:
            break
        if len(tokens) > remaining:
            tokens = [speaker] + tokens[-(remaining - 1):]
        selected.append((tokens, speaker))
        remaining -= len(tokens)
    ids = prefix[:]
    types = [speaker2] * len(prefix)
    for tokens, speaker in reversed(selected):
        ids.extend(tokens)
        types.extend([speaker] * len(tokens))
    return ids + [speaker2], types + [speaker2]


class PersonaDataset:
    """Load one JSON file or an ordered collection of required JSON parts."""

    def __init__(self, path, tokenizer, max_length=256):
        paths = [path] if isinstance(path, (str, PathLike)) else path
        self.examples = []
        for source_path in paths:
            with open(source_path, encoding="utf-8") as source:
                records = json.load(source)
            self.examples.extend(
                (record["personality"], turn)
                for record in records for turn in record["utterances"]
            )
        if not self.examples:
            raise ValueError(f"Dataset is empty: {path}")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        persona, turn = self.examples[index]
        if len(turn["candidates"]) < 2:
            raise ValueError("Each turn needs a distractor and a gold response.")
        # Use one distractor plus the gold answer, as in the original default.
        answers = [self.tokenizer.encode(text, add_special_tokens=False)[:self.max_length // 2 - 1]
                   + [self.tokenizer.eos_token_id] for text in turn["candidates"][-2:]]
        context, types = encode_context(
            self.tokenizer, persona, turn["history"], self.max_length - max(map(len, answers))
        )
        choices = []
        for i, answer in enumerate(answers):
            ids = context + answer
            labels = [-100] * len(context) + (answer if i == 1 else [-100] * len(answer))
            choices.append({"input_ids": ids, "token_type_ids": types + [types[-1]] * len(answer),
                            "labels": labels, "mc_token_ids": len(ids) - 1})
        return choices


def collate(examples, pad_id):
    import torch
    length = max(len(choice["input_ids"]) for example in examples for choice in example)
    result = {key: [] for key in ("input_ids", "token_type_ids", "labels", "attention_mask", "mc_token_ids")}
    for choices in examples:
        for key in result:
            values = []
            for choice in choices:
                size = len(choice["input_ids"])
                if key == "mc_token_ids":
                    value = choice[key]
                elif key == "attention_mask":
                    value = [1] * size + [0] * (length - size)
                else:
                    fill = -100 if key == "labels" else pad_id
                    value = choice[key] + [fill] * (length - size)
                values.append(value)
            result[key].append(values)
    result = {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}
    result["mc_labels"] = torch.ones(len(examples), dtype=torch.long)
    return result
