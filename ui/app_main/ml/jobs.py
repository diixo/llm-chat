"""Persistent local jobs; model work runs outside the Django process."""
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

import psutil
from django.conf import settings
from filelock import FileLock

ACTIVE = {"starting", "loading", "training", "validating", "saving"}
MODEL_FILES = ("config.json", "model.safetensors", "vocab.json", "merges.txt", "tokenizer_config.json")
logger = logging.getLogger(__name__)


def root():
    return Path(getattr(settings, "TRAINING_ROOT", settings.BASE_DIR.parent / "models" / "runs"))


def read_json(path):
    # On Windows an open reader can prevent the atomic replacement of this file.
    with FileLock(str(path) + ".lock", timeout=5):
        with path.open(encoding="utf-8") as source:
            return json.load(source)


def write_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
        with FileLock(str(path) + ".lock", timeout=5):
            # External readers (editors, scanners) do not honor our sidecar lock.
            delays = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8)
            for attempt in range(len(delays) + 1):
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError:
                    if attempt == len(delays):
                        raise
                    time.sleep(delays[attempt])
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove temporary JSON file %s", temporary, exc_info=True)


def runs():
    result = []
    for path in sorted(root().glob("*/status.json"), reverse=True):
        status = read_json(path)
        final_path = path.parent / "final-status.json"
        if final_path.is_file():
            status.update(read_json(final_path))
        status.setdefault("started_at", status.get("created", path.stat().st_mtime))
        if status["state"] in ACTIVE:
            try:
                process = psutil.Process(status["pid"])
                alive = (abs(process.create_time() - status["created"]) < 1
                         and process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
            except psutil.AccessDenied:
                # An inaccessible process may still own the GPU; do not allow a second run.
                alive = True
            except (psutil.Error, KeyError, ValueError):
                alive = False
            if not alive:
                status.update(state="failed", message="Training process exited. See the run log.")
        status["stopping"] = (path.parent / "stop").exists() and status["state"] in ACTIVE
        status["ready"] = status["state"] == "completed" and all(
            (path.parent / "model" / name).is_file() for name in MODEL_FILES
        )
        result.append(status)
    return sorted(result, key=lambda status: status["started_at"], reverse=True)


def run_path(run_id):
    if not run_id or Path(run_id).name != run_id or any(c not in "0123456789abcdef-" for c in run_id):
        raise ValueError("Invalid training run.")
    path = root() / run_id
    if not (path / "status.json").is_file():
        raise ValueError("Training run not found.")
    return path


def start(config):
    config = dict(config)
    config["dataset"] = "full"
    root().mkdir(parents=True, exist_ok=True)
    with FileLock(str(root() / "launch.lock"), timeout=5):
        if any(run["state"] in ACTIVE for run in runs()):
            raise ValueError("A training run is already active.")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        path = root() / run_id
        path.mkdir()
        config["data_dir"] = str(settings.BASE_DIR.parent / "data" / "personachat_truecased")
        config["cache_dir"] = str(settings.BASE_DIR.parent / "models" / "hf-cache")
        write_json(path / "config.json", config)
        initial = {"id": run_id, "state": "starting", "message": "Starting worker…", "config": config,
                   "started_at": datetime.now(timezone.utc).timestamp()}
        write_json(path / "status.json", initial)
        process = None
        with (path / "train.log").open("w", encoding="utf-8") as log:
            try:
                process = subprocess.Popen(
                    [sys.executable, "-m", "app_main.ml.worker", str(path.resolve())],
                    cwd=settings.BASE_DIR, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                initial.update(pid=process.pid, created=psutil.Process(process.pid).create_time())
                write_json(path / "status.json", initial)
                (path / "launched").touch()
                threading.Thread(target=process.wait, daemon=True).start()
            except Exception as exc:
                if process is not None:
                    # Never leave an unregistered worker training in the background.
                    if process.poll() is None:
                        process.terminate()
                    process.wait()
                initial.update(state="failed", message=str(exc))
                write_json(path / "status.json", initial)
                raise RuntimeError(f"Could not start training: {exc}") from exc
        return run_id


def stop(run_id):
    (run_path(run_id) / "stop").touch()


def log_tail(run_id):
    path = run_path(run_id) / "train.log"
    if not path.exists():
        return ""
    with path.open("rb") as source:
        source.seek(max(0, path.stat().st_size - 12000))
        return source.read().decode("utf-8", errors="replace")
