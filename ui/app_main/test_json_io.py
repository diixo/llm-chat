"""Training status must survive Windows file sharing conflicts."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
import errno
from io import StringIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest import skipUnless
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from .ml import jobs, worker


class TrainingJsonTests(SimpleTestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        settings_override = override_settings(TRAINING_ROOT=self.root)
        settings_override.enable()
        self.addCleanup(settings_override.disable)
        self.run = self.root / "abc123"
        self.run.mkdir()
        self.path = self.run / "status.json"
        self.initial = {"id": "abc123", "state": "training", "step": 10, "config": {}}
        jobs.write_json(self.path, self.initial)

    def test_transient_permission_error_retries_and_publishes_complete_json(self):
        replacement = {**self.initial, "step": 20}
        original_replace = os.replace
        attempts = []

        def replace(source, destination):
            attempts.append((source, destination))
            if len(attempts) < 3:
                raise PermissionError("Destination is temporarily open")
            original_replace(source, destination)

        with patch.object(jobs.os, "replace", side_effect=replace), patch.object(jobs.time, "sleep"):
            jobs.write_json(self.path, replacement)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(jobs.read_json(self.path), replacement)
        self.assertEqual(list(self.run.glob("*.tmp")), [])

    def test_permanent_permission_error_preserves_previous_json_and_cleans_temporary(self):
        previous = self.path.read_bytes()
        with patch.object(jobs.os, "replace", side_effect=PermissionError("Always denied")) as replace, \
                patch.object(jobs.time, "sleep"):
            with self.assertRaisesRegex(PermissionError, "Always denied"):
                jobs.write_json(self.path, {**self.initial, "step": 20})
        self.assertGreater(replace.call_count, 1)
        self.assertLessEqual(replace.call_count, 10)
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertEqual(list(self.run.glob("*.tmp")), [])

    def test_unrelated_io_error_is_not_retried(self):
        previous = self.path.read_bytes()
        with patch.object(jobs.os, "replace", side_effect=OSError(errno.ENOSPC, "Disk full")) as replace, \
                patch.object(jobs.time, "sleep") as sleep:
            with self.assertRaises(OSError) as error:
                jobs.write_json(self.path, {**self.initial, "step": 20})
        self.assertEqual(error.exception.errno, errno.ENOSPC)
        replace.assert_called_once()
        sleep.assert_not_called()
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertEqual(list(self.run.glob("*.tmp")), [])

    @skipUnless(os.name == "nt", "Windows prevents replacing a file held by a normal reader")
    def test_windows_reader_can_close_during_replacement_retry(self):
        blocked = Event()
        original_replace = os.replace

        def replace(source, destination):
            try:
                return original_replace(source, destination)
            except PermissionError:
                blocked.set()
                raise

        replacement = {**self.initial, "step": 20}
        with patch.object(jobs.os, "replace", side_effect=replace), ThreadPoolExecutor(max_workers=1) as executor:
            with self.path.open("r", encoding="utf-8"):
                completed = executor.submit(jobs.write_json, self.path, replacement)
                self.assertTrue(blocked.wait(timeout=5), "A real Windows sharing conflict did not occur")
            completed.result(timeout=5)
        self.assertEqual(jobs.read_json(self.path), replacement)
        self.assertEqual(list(self.run.glob("*.tmp")), [])

    def test_progress_reporting_failure_does_not_raise(self):
        status = {**self.initial, "step": 20, "message": "Epoch 1/1"}
        with patch.object(worker, "write_json", side_effect=PermissionError("Status is locked")), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            worker.publish_status(self.run, status)
        self.assertEqual(jobs.read_json(self.path), self.initial)

    def test_terminal_status_survives_primary_status_write_failure(self):
        original_write = worker.write_json

        def write(path, value):
            if path.name == "status.json":
                raise PermissionError("Primary status is locked")
            return original_write(path, value)

        for state in ("completed", "stopped", "failed"):
            with self.subTest(state=state):
                status = {**self.initial, "state": state, "message": "Final result", "step": 20}
                with patch.object(worker, "write_json", side_effect=write), \
                        redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    worker.publish_status(self.run, status)
                self.assertEqual(jobs.read_json(self.path), self.initial)
                self.assertEqual(jobs.read_json(self.run / "final-status.json"), status)
                self.assertEqual(jobs.runs()[0]["state"], state)
                self.assertEqual(jobs.runs()[0]["message"], "Final result")
                self.assertEqual(jobs.runs()[0]["step"], 20)

    def test_failed_final_status_write_still_attempts_primary_status(self):
        original_write = worker.write_json

        def write(path, value):
            if path.name == "final-status.json":
                raise OSError(errno.ENOSPC, "Cannot write final status")
            return original_write(path, value)

        status = {**self.initial, "state": "failed", "message": "Original training error"}
        with patch.object(worker, "write_json", side_effect=write), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            worker.publish_status(self.run, status)
        self.assertEqual(jobs.read_json(self.path), status)
        self.assertEqual(jobs.runs()[0]["message"], "Original training error")

    def test_both_status_writes_failing_do_not_raise(self):
        status = {**self.initial, "state": "failed", "message": "Original training error"}
        with patch.object(worker, "write_json", side_effect=OSError(errno.ENOSPC, "Disk full")) as write, \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            worker.publish_status(self.run, status)
        self.assertEqual({call.args[0].name for call in write.call_args_list},
                         {"status.json", "final-status.json"})
        self.assertEqual(jobs.read_json(self.path), self.initial)
