import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.core.signing import get_cookie_signer
from django.test import Client, SimpleTestCase, override_settings
from filelock import FileLock

from . import conversations, views
from .ml import jobs


class DialogueConcurrencyTests(SimpleTestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.override = override_settings(
            TRAINING_ROOT=self.root,
            DIALOGUE_ROOT=self.root / "dialogues",
            SESSION_ENGINE="django.contrib.sessions.backends.db",
        )
        self.override.enable()
        self.addCleanup(self.override.disable)
        for target, value in (
            ("runs", [{"id": "abc123", "ready": True}]),
            ("run_path", self.root),
        ):
            mock = patch.object(jobs, target, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)
        self.client.get("/dialogue")
        self.conversation_id = get_cookie_signer(
            salt=conversations.COOKIE_NAME + "app_main.dialogue",
        ).unsign(self.client.cookies[conversations.COOKIE_NAME].value)
        self.other = Client()
        self.other.cookies = self.client.cookies.copy()
        self.data = {"run_id": "abc123", "persona": "I like books.", "message": "First message"}

    def run_overlapping_posts(self, second_data):
        generating = threading.Event()
        resume = threading.Event()
        acquiring = threading.Event()
        second_finished = threading.Event()
        results = {}
        histories = []
        original_acquire = FileLock.acquire

        def generate(_path, _persona, history):
            histories.append(list(history))
            if len(histories) == 1:
                generating.set()
                if not resume.wait(5):
                    raise RuntimeError("Test did not release generation")
            return "Reply " + str(len(histories))

        def acquire(lock, *args, **kwargs):
            if threading.current_thread().name == "second-dialogue-request":
                acquiring.set()
            return original_acquire(lock, *args, **kwargs)

        def post(name, client, data):
            try:
                results[name] = client.post("/dialogue", data).status_code
            except BaseException as exc:
                results[name] = exc
            finally:
                if name == "second":
                    second_finished.set()

        with patch("app_main.ml.dialogue.reply", side_effect=generate), patch.object(FileLock, "acquire", acquire):
            first = threading.Thread(target=post, args=("first", self.client, self.data))
            second = threading.Thread(name="second-dialogue-request", target=post,
                                      args=("second", self.other, second_data))
            first.start()
            try:
                self.assertTrue(generating.wait(5))
                # Reading a conversation remains available during generation.
                self.assertEqual(self.other.get("/dialogue").status_code, 200)
                second.start()
                self.assertTrue(acquiring.wait(5))
                self.assertFalse(second_finished.is_set())
            finally:
                resume.set()
                first.join(5)
                if second.ident is not None:
                    second.join(5)
            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
        self.assertEqual(results, {"first": 302, "second": 302})
        return histories

    def test_reset_waits_and_does_not_restore_in_flight_reply(self):
        histories = self.run_overlapping_posts({"action": "reset"})
        self.assertEqual(histories, [["First message"]])
        self.assertEqual(conversations.load(self.conversation_id), {})

    def test_parallel_messages_reload_history_after_waiting(self):
        second_data = {**self.data, "message": "Second message"}
        histories = self.run_overlapping_posts(second_data)
        self.assertEqual(histories, [["First message"], ["First message", "Reply 1", "Second message"]])
        self.assertEqual(conversations.load(self.conversation_id)["history"],
                         ["First message", "Reply 1", "Second message", "Reply 2"])

    def test_view_exception_releases_json_lock(self):
        with patch.object(views, "_dialogue_response", side_effect=RuntimeError("test failure")):
            with self.assertRaisesMessage(RuntimeError, "test failure"):
                self.client.post("/dialogue", self.data)
        with FileLock(conversations.path(self.conversation_id).with_suffix(".lock"), timeout=0):
            pass
        with patch("app_main.ml.dialogue.reply", return_value="Recovered"):
            self.assertEqual(self.client.post("/dialogue", self.data).status_code, 302)
