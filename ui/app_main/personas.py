"""Read the assistant personas available in the local PersonaChat datasets."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from threading import Lock

from django.conf import settings


_catalog_lock = Lock()


def _source_label(path):
    if "_train" in path.stem:
        return "Train"
    if "_valid" in path.stem:
        return "Validation"
    return path.name


def _personalities(path):
    """Release each complete dataset before reading the next file."""
    try:
        with path.open(encoding="utf-8") as source:
            records = json.load(source)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Invalid PersonaChat JSON in {path.name}: {error}") from error
    if not isinstance(records, list):
        raise ValueError(f"Invalid PersonaChat dataset in {path.name}: expected a list of records.")
    for index, record in enumerate(records, 1):
        facts = record.get("personality") if isinstance(record, dict) else None
        if (not isinstance(facts, list) or not facts
                or any(not isinstance(fact, str) or not fact.strip() for fact in facts)):
            raise ValueError(
                f"Invalid personality in {path.name}, record {index}: "
                "expected a nonempty list of text facts."
            )
        yield tuple(dict.fromkeys(" ".join(fact.split()) for fact in facts))


@lru_cache(maxsize=2)
def _read_catalog(signature):
    catalog = {}
    for filename, _modified, _size in signature:
        path = Path(filename)
        source = _source_label(path)
        for facts in _personalities(path):
            key = tuple(sorted(facts))
            if key not in catalog:
                persona_id = hashlib.sha256(
                    json.dumps(key, ensure_ascii=False).encode("utf-8")
                ).hexdigest()
                catalog[key] = {"id": persona_id, "facts": facts, "sources": set()}
            catalog[key]["sources"].add(source)
    # Cache immutable data so filtering or modifying a response cannot corrupt it.
    return tuple(
        (persona["id"], persona["facts"], tuple(sorted(persona["sources"])))
        for persona in catalog.values()
    )


def get_personas():
    """Return unique persona profiles without database access or file writes.

    Profiles are deduplicated by their complete set of facts, ignoring fact order
    and redundant whitespace. Cache entries expire when a dataset file changes,
    appears or disappears. A lock prevents concurrent requests parsing the same
    large datasets simultaneously.
    """
    root = Path(getattr(
        settings, "PERSONACHAT_ROOT", Path(settings.BASE_DIR).parent / "data" / "personachat_truecased",
    ))
    with _catalog_lock:
        try:
            root.stat()
        except FileNotFoundError:
            return []
        if not root.is_dir():
            raise NotADirectoryError(f"PersonaChat dataset directory is not a directory: {root}")
        signature = []
        for path in sorted(root.glob("personachat_truecased_*.json")):
            if path.is_file():
                stat = path.stat()
                signature.append((str(path.resolve()), stat.st_mtime_ns, stat.st_size))
        personas = _read_catalog(tuple(signature))
    return [
        {"id": persona_id, "facts": list(facts), "sources": list(sources)}
        for persona_id, facts, sources in personas
    ]
