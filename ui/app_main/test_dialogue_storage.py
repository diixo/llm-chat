import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.conf import settings
from django.core.signing import get_cookie_signer
from django.test import Client, SimpleTestCase, override_settings

from . import conversations
from .ml import jobs


class DialogueJsonTests(SimpleTestCase):
    """Real request lifecycles must not access Django's session database."""

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        settings_override = override_settings(
            TRAINING_ROOT=self.root / "training",
            DIALOGUE_ROOT=self.root / "dialogues",
            SESSION_ENGINE="django.contrib.sessions.backends.db",
        )
        settings_override.enable()
        self.addCleanup(settings_override.disable)
        for target, value in (
            ("runs", [{"id": "abc123", "ready": True}]),
            ("run_path", self.root / "training" / "abc123"),
        ):
            mock = patch.object(jobs, target, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)
        self.data = {"run_id": "abc123", "persona": "I like books.", "message": "Hello!"}

    def conversation_id(self, client):
        return get_cookie_signer(
            salt=conversations.COOKIE_NAME + "app_main.dialogue",
        ).unsign(client.cookies[conversations.COOKIE_NAME].value)

    def test_open_send_reload_and_reset_ignore_existing_database_session(self):
        self.client.cookies[settings.SESSION_COOKIE_NAME] = "a" * 32
        opened = self.client.get("/dialogue")
        self.assertEqual(opened.status_code, 200)
        cookie = opened.cookies[conversations.COOKIE_NAME]
        self.assertTrue(cookie["httponly"])
        self.assertEqual(cookie["samesite"], "Lax")
        conversation_id = self.conversation_id(self.client)
        self.assertRegex(conversation_id, r"^[0-9a-f]{64}$")
        with patch("app_main.ml.dialogue.reply", return_value="Hello, friend!"):
            sent = self.client.post("/dialogue", self.data)
        self.assertEqual(sent.status_code, 302)
        reloaded = self.client.get("/dialogue")
        self.assertContains(reloaded, "Hello, friend!")
        self.assertEqual(self.conversation_id(self.client), conversation_id)
        self.assertEqual(conversations.load(conversation_id), {
            "run_id": "abc123", "persona": "I like books.",
            "history": ["Hello!", "Hello, friend!"],
        })
        reset = self.client.post("/dialogue", {"action": "reset"})
        self.assertEqual(reset.status_code, 302)
        self.assertEqual(conversations.load(conversation_id), {})
        self.assertNotContains(self.client.get("/dialogue"), "Hello, friend!")
        for response in (opened, sent, reloaded, reset):
            self.assertNotIn(settings.SESSION_COOKIE_NAME, response.cookies)

    def test_browsers_keep_separate_conversations_and_can_reload_from_disk(self):
        other = Client()
        self.client.get("/dialogue")
        other.get("/dialogue")
        first_id = self.conversation_id(self.client)
        second_id = self.conversation_id(other)
        self.assertNotEqual(first_id, second_id)
        with patch("app_main.ml.dialogue.reply", side_effect=["First private reply", "Second private reply"]):
            self.assertEqual(self.client.post("/dialogue", self.data).status_code, 302)
            self.assertEqual(other.post("/dialogue", {**self.data, "message": "Second browser"}).status_code, 302)
        reopened = Client()
        reopened.cookies = self.client.cookies.copy()
        first_page = reopened.get("/dialogue")
        second_page = other.get("/dialogue")
        self.assertContains(first_page, "First private reply")
        self.assertNotContains(first_page, "Second private reply")
        self.assertContains(second_page, "Second private reply")
        self.assertNotContains(second_page, "First private reply")
        other.post("/dialogue", {"action": "reset"})
        self.assertEqual(conversations.load(second_id), {})
        stored = json.loads(conversations.path(first_id).read_text(encoding="utf-8"))
        self.assertEqual(stored["history"], ["Hello!", "First private reply"])

    def test_exported_session_is_found_without_accessing_database(self):
        old_key = "b" * 32
        exported_id = conversations.legacy_id(old_key)
        conversations.save(exported_id, {
            "run_id": "abc123", "persona": "I like books.",
            "history": ["Existing question", "Existing reply"],
        })
        self.client.cookies[settings.SESSION_COOKIE_NAME] = old_key
        response = self.client.get("/dialogue")
        self.assertContains(response, "Existing reply")
        self.assertEqual(self.conversation_id(self.client), exported_id)
        with patch("app_main.ml.dialogue.reply", return_value="Continued reply") as reply:
            self.assertEqual(self.client.post("/dialogue", self.data).status_code, 302)
        self.assertEqual(reply.call_args.args[2], ["Existing question", "Existing reply", "Hello!"])

    def test_tampered_or_invalid_signed_identifiers_are_replaced(self):
        signer = get_cookie_signer(salt=conversations.COOKIE_NAME + "app_main.dialogue")
        for cookie_value in ("tampered", signer.sign("../outside"), signer.sign("a" * 63)):
            with self.subTest(cookie=cookie_value):
                client = Client()
                client.cookies[conversations.COOKIE_NAME] = cookie_value
                response = client.get("/dialogue")
                self.assertEqual(response.status_code, 200)
                self.assertIn(conversations.COOKIE_NAME, response.cookies)
                self.assertRegex(self.conversation_id(client), r"^[0-9a-f]{64}$")

    def test_failed_reply_keeps_last_saved_conversation(self):
        self.client.get("/dialogue")
        conversation_id = self.conversation_id(self.client)
        with patch("app_main.ml.dialogue.reply", return_value="Saved reply"):
            self.client.post("/dialogue", self.data)
        original = conversations.path(conversation_id).read_bytes()
        with patch("app_main.ml.dialogue.reply", side_effect=RuntimeError("Model unavailable")):
            response = self.client.post("/dialogue", {**self.data, "message": "Failed message"})
        self.assertContains(response, "Model unavailable")
        self.assertEqual(conversations.path(conversation_id).read_bytes(), original)
