"""Browser conversations stored as plain JSON, independently of Django sessions."""
import json
from pathlib import Path
import re
import secrets

from django.conf import settings
from django.utils.crypto import salted_hmac

from .ml.jobs import write_json

COOKIE_NAME = "llm_chat_dialogue"
COOKIE_SALT = "app_main.dialogue"


def root():
    return Path(getattr(settings, "DIALOGUE_ROOT", settings.BASE_DIR / "data" / "dialogues"))


def legacy_id(session_key):
    """Stable ID for conversations exported from the previous session storage."""
    return salted_hmac("app_main.dialogue.legacy", session_key, algorithm="sha256").hexdigest()


def identify(request):
    value = request.get_signed_cookie(COOKIE_NAME, default=None, salt=COOKIE_SALT)
    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
        return value, False
    # This only looks up previously exported JSON. It never accesses the database.
    old_key = request.COOKIES.get(settings.SESSION_COOKIE_NAME, "")
    if re.fullmatch(r"[a-z0-9]{32}", old_key):
        value = legacy_id(old_key)
        if path(value).is_file():
            return value, True
    return secrets.token_hex(32), True


def path(conversation_id):
    if not isinstance(conversation_id, str) or not re.fullmatch(r"[0-9a-f]{64}", conversation_id):
        raise ValueError("Invalid conversation ID.")
    return root() / f"{conversation_id}.json"


def load(conversation_id):
    try:
        with path(conversation_id).open(encoding="utf-8") as source:
            conversation = json.load(source)
    except FileNotFoundError:
        return {}
    if not isinstance(conversation, dict):
        raise ValueError("Conversation JSON must contain an object.")
    return conversation


def save(conversation_id, conversation):
    destination = path(conversation_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, conversation)


def set_cookie(response, conversation_id):
    response.set_signed_cookie(
        COOKIE_NAME, conversation_id, salt=COOKIE_SALT,
        max_age=settings.SESSION_COOKIE_AGE, httponly=True,
        secure=settings.SESSION_COOKIE_SECURE, samesite="Lax",
    )
