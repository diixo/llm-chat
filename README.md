# llm-chat

## GPT-2 124M: training and dialogue

Install the dependencies into the same Python environment used for Django:

```bash
python -m pip install -r requirements.txt
python ui/manage.py migrate
python ui/manage.py runserver
```

For GPU training, install a CUDA-enabled PyTorch build appropriate for your
machine using https://pytorch.org/get-started/locally/. An existing compatible
CUDA installation of PyTorch can be used as-is.

Open **Training** in the sidebar. The default sample run uses the two small
files in `data/personachat_truecased`; select **Full PersonaChat** for
the three `personachat_truecased_full_train.part1.json`, `.part2.json`, and
`.part3.json` training files and `personachat_truecased_full_valid.json`.
Each training part is a standalone JSON array under 100 MB. The loader combines
all three in order, preserving the full training dataset; all parts are required.
Validation data is never used for optimizer updates.

The first run downloads `openai-community/gpt2` (the 124M base model).
Each run starts from those pretrained weights. The worker runs separately
from Django, uses CUDA when available, mixed precision on CUDA, gradient
checkpointing, and gradient accumulation. Start with batch size 1, length 256,
and accumulation 8. Larger settings may exceed a 6 GB GPU's memory.
The sample run checks the pipeline; it is not enough to train a useful chatbot.

This implementation follows the persona/history/speaker-token approach from
[Hugging Face's TransferTransfo example](https://github.com/huggingface/transfer-learning-conv-ai),
using the current project's Transformers 4.56 API. It trains both language
modeling on the gold reply and a multiple-choice head on two candidates (the
last distractor and the final, gold candidate). Context and padding have no
language-model loss. Each epoch reports token-weighted validation loss,
perplexity, and accuracy over those two candidates; this accuracy is not the
original benchmark's full-candidate Hits@1. Long inputs retain a bounded persona
and the most recent dialogue context.
The learning rate stays constant throughout a run, unlike the original
`train.py`'s linear decay. Persona permutations and distributed training are
not implemented. This is an adaptation, not an exact benchmark reproduction.

Status, logs, configuration, epoch metrics, and the final model/tokenizer are
stored under `models/runs/<run-id>/`. Downloaded files are cached under
`models/hf-cache/`. Both are excluded by the existing `models/` Git ignore rule.
Only one training process can run at a time. **Stop training** is cooperative:
it waits for the current download/batch to finish. Failed/stopped runs do not
appear as chat models. This version saves a final model after all epochs;
it does not support resuming interrupted training.

Open **Dialogue**, choose a completed run, enter persona sentences, and send
a message. Persona, model selection, and conversation history are saved as plain
JSON in `ui/data/dialogues/<conversation-id>.json` (excluded from Git). A signed
browser cookie identifies the file; the dialogue page does not use Django
sessions or write to `db.sqlite3`. The last few exchanges are used as context.
Changing persona/model resets that context. Dialogue inference uses CPU with a
token cache so training can use the GPU. Requests sharing a conversation are
serialized; a reset from another tab waits for the current reply, then clears
the JSON history. These pages are intended for the project's local development
server.

Run the checks:

```bash
python ui/manage.py test app_main
```


## Run django-server:
```bash
py ui/manage.py runserver
```


## Create repository structure

```bash
django-admin startproject ui
```

```bash
cd ui
```

```bash
py manage.py startapp app_main
```

- DB migrations with migrates:
```bash
py ui/manage.py makemigrations
py ui/manage.py migrate
```
