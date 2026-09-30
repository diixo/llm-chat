"""Training log tails preserve complete lines for every run."""
from pathlib import Path
from tempfile import TemporaryDirectory

from django.test import SimpleTestCase, override_settings

from .ml import jobs


class TrainingLogTests(SimpleTestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        settings_override = override_settings(TRAINING_ROOT=self.root)
        settings_override.enable()
        self.addCleanup(settings_override.disable)
        self.run_id = "abc123"
        self.run = self.root / self.run_id
        self.run.mkdir()
        jobs.write_json(self.run / "status.json", {
            "id": self.run_id, "state": "failed", "started_at": 2, "config": {},
        })
        self.log = self.run / "train.log"

    def test_missing_and_empty_logs(self):
        self.assertEqual(jobs.log_tail(self.run_id), "")
        self.log.write_bytes(b"")
        self.assertEqual(jobs.log_tail(self.run_id), "")

    def test_short_and_exactly_100_line_logs_are_preserved(self):
        for count in (1, 12, 100):
            for trailing_newline in (False, True):
                with self.subTest(count=count, trailing_newline=trailing_newline):
                    content = "\n".join(f"Epoch 1/1 - batch {index}" for index in range(count))
                    if trailing_newline:
                        content += "\n"
                    self.log.write_bytes(content.encode("utf-8"))
                    self.assertEqual(jobs.log_tail(self.run_id), content)

    def test_only_last_100_lines_are_returned_for_different_run_logs(self):
        for index, count in enumerate((101, 199, 1500)):
            for trailing_newline in (False, True):
                with self.subTest(count=count, trailing_newline=trailing_newline):
                    run_id = f"def{index}"
                    run = self.root / run_id
                    run.mkdir(exist_ok=True)
                    jobs.write_json(run / "status.json", {"id": run_id, "state": "failed"})
                    lines = [f"Run {index} - batch {number}\n" for number in range(count)]
                    if not trailing_newline:
                        lines[-1] = lines[-1].rstrip("\n")
                    (run / "train.log").write_bytes("".join(lines).encode("utf-8"))
                    self.assertEqual(jobs.log_tail(run_id), "".join(lines[-100:]))

    def test_first_retained_line_can_exceed_previous_byte_limit(self):
        retained = ["Long diagnostic: " + "x" * 20000 + "\n"]
        retained.extend(f"Batch {index}\n" for index in range(99))
        content = "Older line\n" + "".join(retained)
        self.log.write_bytes(content.encode("utf-8"))
        self.assertEqual(jobs.log_tail(self.run_id), "".join(retained))
        self.assertEqual(self.log.read_bytes(), content.encode("utf-8"))

    def test_one_long_unterminated_line_is_preserved(self):
        content = "Diagnostic: " + "x" * 25000
        self.log.write_bytes(content.encode("utf-8"))
        self.assertEqual(jobs.log_tail(self.run_id), content)

    def test_utf8_and_crlf_are_preserved_across_read_boundaries(self):
        # The last 8192 bytes start inside a UTF-8 character or between CR and LF.
        cases = {
            "utf8": ["First caf\u00e9" + "x" * 7595 + "\r\n"] + ["tail\r\n"] * 99,
            "crlf": ["First line\r\n", "x" * 7895 + "\r\n"] + ["x\r\n"] * 98,
        }
        for name, retained in cases.items():
            with self.subTest(boundary=name):
                self.log.write_bytes(("Discarded\r\n" + "".join(retained)).encode("utf-8"))
                self.assertEqual(jobs.log_tail(self.run_id), "".join(retained))

    def test_training_status_returns_latest_run_log_with_whole_lines(self):
        older = self.root / "fff"
        older.mkdir()
        jobs.write_json(older / "status.json", {
            "id": "fff", "state": "failed", "started_at": 1, "config": {},
        })
        (older / "train.log").write_bytes(b"Older run\n")
        lines = [f"Epoch 1/1 - batch {index}\n" for index in range(240)]
        self.log.write_bytes("".join(lines).encode("utf-8"))

        response = self.client.get("/training/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["runs"][0]["id"], self.run_id)
        self.assertEqual(payload["log"], "".join(lines[-100:]))
