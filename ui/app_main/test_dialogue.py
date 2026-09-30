import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from .ml import dialogue
from .ml.data import SPECIAL_TOKENS


class DialogueCacheTests(TestCase):
    def test_cached_generation_matches_full_context_logits(self):
        import torch
        from transformers import GPT2Config, GPT2DoubleHeadsModel, GPT2Tokenizer
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

        torch.set_num_threads(2)
        torch.manual_seed(42)
        with TemporaryDirectory() as directory:
            path = Path(directory)
            vocabulary = {token: i for i, token in enumerate(bytes_to_unicode().values())}
            vocabulary["<|endoftext|>"] = len(vocabulary)
            (path / "vocab.json").write_text(json.dumps(vocabulary), encoding="utf-8")
            (path / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
            tokenizer = GPT2Tokenizer(vocab_file=str(path / "vocab.json"),
                                      merges_file=str(path / "merges.txt"))
            tokenizer.add_special_tokens(SPECIAL_TOKENS)
            model = GPT2DoubleHeadsModel(GPT2Config(
                vocab_size=len(tokenizer), n_positions=128, n_embd=16, n_layer=1,
                n_head=2, use_cache=False,
            )).eval()
            forward = model.forward
            seen_ids = None
            seen_types = None
            lengths = []

            def checked_forward(**kwargs):
                nonlocal seen_ids, seen_types
                ids, types = kwargs["input_ids"], kwargs["token_type_ids"]
                lengths.append(ids.size(-1))
                if seen_ids is None:
                    seen_ids, seen_types = ids, types
                else:
                    self.assertEqual(ids.size(-1), 1)
                    self.assertTrue((types == tokenizer.convert_tokens_to_ids("<speaker2>")).all())
                    seen_ids = torch.cat((seen_ids, ids), dim=-1)
                    seen_types = torch.cat((seen_types, types), dim=-1)
                expected = forward(input_ids=seen_ids, token_type_ids=seen_types, use_cache=False)
                actual = forward(**kwargs)
                self.assertIsNotNone(actual.past_key_values)
                torch.testing.assert_close(actual.logits[:, -1], expected.logits[:, -1],
                                           rtol=1e-4, atol=1e-5)
                return actual

            previous_cache = dialogue._cached
            dialogue._cached = (str(path), tokenizer, model)
            try:
                with patch.object(model, "forward", side_effect=checked_forward):
                    result = dialogue.reply(path, ["I like books."], ["Hi!", "Hello!", "How are you?"])
                self.assertIsInstance(result, str)
                self.assertGreater(len(lengths), 1)
                self.assertGreater(lengths[0], 1)
            finally:
                dialogue._cached = previous_cache
