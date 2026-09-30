import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from django.test import SimpleTestCase, override_settings

from .forms import TrainingForm
from .ml import jobs


class TrainingPagesTests(SimpleTestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings_override = override_settings(
            TRAINING_ROOT=self.root, SESSION_ENGINE="django.contrib.sessions.backends.signed_cookies"
        )
        self.settings_override.enable()
        self.addCleanup(self.settings_override.disable)

    def test_empty_pages_and_navigation(self):
        for url in ("/", "/training", "/dialogue"):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertContains(response, 'href="/training"')
            self.assertContains(response, 'href="/dialogue"')
        self.assertEqual(self.client.get("/training/status").json(), {"runs": [], "log": ""})

    def test_invalid_training_settings_do_not_launch(self):
        with patch.object(jobs, "start") as start:
            response = self.client.post("/training", {"dataset": "../../other", "epochs": "-1"})
            self.assertEqual(response.status_code, 200)
            start.assert_not_called()
        form = TrainingForm({"dataset": "sample", "epochs": 1, "batch_size": 1,
                             "gradient_accumulation": 8, "max_length": 256, "learning_rate": "nan"})
        self.assertFalse(form.is_valid())

    def test_valid_launch_and_duplicate_error(self):
        data = {"dataset": "sample", "epochs": 1, "batch_size": 1,
                "gradient_accumulation": 8, "max_length": 256, "learning_rate": 0.0000625}
        with patch.object(jobs, "start", return_value="run") as start:
            self.assertEqual(self.client.post("/training", data).status_code, 302)
            self.assertEqual(start.call_args.args[0]["max_length"], 256)
        with patch.object(jobs, "start", side_effect=ValueError("A training run is already active.")):
            self.assertContains(self.client.post("/training", data), "already active")

    def test_csrf_required_for_training(self):
        from django.test import Client
        self.assertEqual(Client(enforce_csrf_checks=True).post("/training", {}).status_code, 403)

    def test_stale_worker_and_path_validation(self):
        run = self.root / "abc123"
        run.mkdir()
        jobs.write_json(run / "status.json", {"id": "abc123", "state": "training", "pid": -1,
                                               "created": 0, "config": {}})
        self.assertEqual(jobs.runs()[0]["state"], "failed")
        self.assertFalse(jobs.runs()[0]["ready"])
        for value in ("..", "../abc123", "C:\\", ""):
            with self.assertRaises(ValueError):
                jobs.run_path(value)
        self.assertEqual(self.client.post("/training", {"action": "stop", "run_id": "abc123"}).status_code, 302)
        self.assertTrue((run / "stop").exists())

    def test_dialogue_session_and_persona_change(self):
        run = self.root / "abc123"
        (run / "model").mkdir(parents=True)
        for name in jobs.MODEL_FILES:
            (run / "model" / name).touch()
        jobs.write_json(run / "status.json", {"id": "abc123", "state": "completed", "config": {}})
        data = {"run_id": "abc123", "persona": "I like books.", "message": "Hello"}
        with patch("app_main.ml.dialogue.reply", return_value="Hi!") as reply:
            self.assertEqual(self.client.post("/dialogue", data).status_code, 302)
            self.assertEqual(self.client.session["dialogue"]["history"], ["Hello", "Hi!"])
            data["message"] = "How are you?"
            self.client.post("/dialogue", data)
            self.assertEqual(reply.call_args.args[2], ["Hello", "Hi!", "How are you?"])
            data["persona"] = "I like dogs."
            self.client.post("/dialogue", data)
            self.assertEqual(reply.call_args.args[2], ["How are you?"])
        self.client.post("/dialogue", {"action": "reset"})
        self.assertNotIn("dialogue", self.client.session)

    def test_launch_failure_terminates_unregistered_worker(self):
        process = Mock(pid=123456)
        process.poll.return_value = None
        with patch.object(jobs.subprocess, "Popen", return_value=process), \
             patch.object(jobs.psutil, "Process", side_effect=jobs.psutil.AccessDenied(123456)):
            with self.assertRaisesRegex(RuntimeError, "Could not start training"):
                jobs.start({"dataset": "sample"})
        process.terminate.assert_called_once()
        process.wait.assert_called_once()
        self.assertEqual(jobs.runs()[0]["state"], "failed")

    def test_incomplete_checkpoint_is_not_available(self):
        run = self.root / "abc123"
        (run / "model").mkdir(parents=True)
        jobs.write_json(run / "model" / "config.json", {})
        jobs.write_json(run / "status.json", {"id": "abc123", "state": "completed", "config": {}})
        self.assertFalse(jobs.runs()[0]["ready"])
        self.assertContains(self.client.get("/dialogue"), "No trained models yet")

    def test_unregistered_worker_does_not_load_model(self):
        from .ml.worker import train
        run = self.root / "abc123"
        run.mkdir()
        jobs.write_json(run / "status.json", {"id": "abc123", "state": "starting"})
        jobs.write_json(run / "config.json", {})
        with patch("app_main.ml.worker.time.sleep"), patch("transformers.GPT2Tokenizer.from_pretrained") as load:
            train(run)
        load.assert_not_called()
        self.assertEqual(json.loads((run / "status.json").read_text())["state"], "failed")

    def test_runs_are_ordered_by_start_time_not_random_id(self):
        for run_id, started_at in (("fff", 1), ("aaa", 2)):
            run = self.root / run_id
            run.mkdir()
            jobs.write_json(run / "status.json", {"id": run_id, "state": "failed", "config": {},
                                                   "started_at": started_at})
        self.assertEqual([run["id"] for run in jobs.runs()], ["aaa", "fff"])

    def test_active_worker_blocks_duplicate_start_even_if_access_denied(self):
        run = self.root / "abc123"
        run.mkdir()
        jobs.write_json(run / "status.json", {"id": "abc123", "state": "training", "pid": 1234,
                                               "created": 0, "config": {}})
        with patch.object(jobs.psutil, "Process", side_effect=jobs.psutil.AccessDenied(1234)), \
             patch.object(jobs.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(ValueError, "already active"):
                jobs.start({"dataset": "sample"})
        spawn.assert_not_called()


class PersonaDatasetFilesTests(SimpleTestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.paths = [root / f"personachat_truecased_full_train.part{part}.json"
                      for part in range(1, 4)]
        self.records = []
        for part, path in enumerate(self.paths, 1):
            records = [{"personality": [f"Persona {part}"], "utterances": [
                {"history": [f"Question {part}.{turn}"],
                 "candidates": ["Distractor", f"Answer {part}.{turn}"]}
                for turn in range(2)
            ]}]
            jobs.write_json(path, records)
            self.records.extend(records)

    def test_training_parts_preserve_all_examples_and_order(self):
        from .ml.data import PersonaDataset
        dataset = PersonaDataset(self.paths, tokenizer=None)
        self.assertEqual(len(dataset), 6)
        self.assertEqual(dataset.examples, [
            (record["personality"], turn)
            for record in self.records for turn in record["utterances"]
        ])

    def test_any_missing_training_part_prevents_loading(self):
        from .ml.data import PersonaDataset
        for missing in self.paths:
            with self.subTest(part=missing.name):
                content = missing.read_bytes()
                missing.unlink()
                try:
                    with self.assertRaises(FileNotFoundError) as error:
                        PersonaDataset(self.paths, tokenizer=None)
                    self.assertEqual(error.exception.filename, str(missing))
                finally:
                    missing.write_bytes(content)


class ModelPipelineTests(SimpleTestCase):
    def setUp(self):
        import torch
        from transformers import GPT2Tokenizer
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
        from .ml.data import SPECIAL_TOKENS
        torch.set_num_threads(2)
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        vocabulary = {token: i for i, token in enumerate(bytes_to_unicode().values())}
        vocabulary["<|endoftext|>"] = len(vocabulary)
        jobs.write_json(self.path / "vocab.json", vocabulary)
        (self.path / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
        self.tokenizer = GPT2Tokenizer(vocab_file=str(self.path / "vocab.json"),
                                      merges_file=str(self.path / "merges.txt"))
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
        self.record = [{"personality": ["I like books."], "utterances": [
            {"history": ["Hi!"], "candidates": ["No.", "Hello!"]},
            {"history": ["Hi!", "Hello!", "How are you?"], "candidates": ["Dog.", "Good."]},
        ]}]
        for split in ("train", "valid"):
            jobs.write_json(self.path / f"personachat_truecased_sample_{split}.json", self.record)

    def test_labels_padding_and_context_budget(self):
        from .ml.data import PersonaDataset, collate, encode_context
        dataset = PersonaDataset(self.path / "personachat_truecased_sample_train.json", self.tokenizer, 128)
        choices = dataset[0]
        self.assertTrue(all(label == -100 for label in choices[0]["labels"]))
        gold = [label for label in choices[1]["labels"] if label != -100]
        self.assertEqual(self.tokenizer.decode(gold, skip_special_tokens=True), "Hello!")
        self.assertEqual(gold[-1], self.tokenizer.eos_token_id)
        batch = collate([dataset[0], dataset[1]], self.tokenizer.pad_token_id)
        self.assertTrue((batch["labels"][batch["attention_mask"] == 0] == -100).all())
        ids, types = encode_context(self.tokenizer, ["persona " * 100], ["old " * 100, "new " * 100], 64)
        self.assertLessEqual(len(ids), 64)
        self.assertEqual(len(ids), len(types))
        self.assertEqual(ids[-1], self.tokenizer.convert_tokens_to_ids("<speaker2>"))

    def test_real_tiny_model_training_save_and_reply(self):
        from transformers import GPT2Config, GPT2DoubleHeadsModel
        from .ml.worker import train
        from .ml import dialogue
        model = GPT2DoubleHeadsModel(GPT2Config(vocab_size=len(self.tokenizer), n_positions=128,
                                               n_embd=16, n_layer=1, n_head=2))
        original_weights = model.transformer.wte.weight.detach().clone()
        run = self.path / "run"
        run.mkdir()
        (run / "launched").touch()
        jobs.write_json(run / "status.json", {"id": "test", "state": "starting"})
        for part in range(1, 4):
            jobs.write_json(self.path / f"personachat_truecased_full_train.part{part}.json", self.record)
        jobs.write_json(self.path / "personachat_truecased_full_valid.json", self.record)
        jobs.write_json(run / "config.json", {"data_dir": str(self.path), "cache_dir": str(self.path),
                        "dataset": "full", "max_length": 128, "batch_size": 1, "epochs": 1,
                        "gradient_accumulation": 8, "learning_rate": 0.0000625})
        with patch("transformers.GPT2Tokenizer.from_pretrained", return_value=self.tokenizer), \
             patch("transformers.GPT2DoubleHeadsModel.from_pretrained", return_value=model), \
             patch("torch.cuda.is_available", return_value=False):
            train(run)
        status = json.loads((run / "status.json").read_text())
        self.assertEqual(status["state"], "completed", status)
        self.assertEqual(status["total"], 6)
        self.assertGreater(status["perplexity"], 0)
        self.assertFalse(original_weights.equal(model.transformer.wte.weight.detach()))
        self.assertTrue((run / "model" / "model.safetensors").exists())
        try:
            result = dialogue.reply(run / "model", ["I like books."], ["Hi!"])
            self.assertIsInstance(result, str)
        finally:
            dialogue._cached = None
        (run / "stop").touch()
        with patch("transformers.GPT2Tokenizer.from_pretrained", return_value=self.tokenizer):
            train(run)
        self.assertEqual(json.loads((run / "status.json").read_text())["state"], "stopped")
