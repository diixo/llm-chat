import hashlib
import json

from django.conf import settings
from django.shortcuts import render, redirect
from django.utils.text import slugify
from django.http import HttpResponse, JsonResponse
from django.views.decorators.http import require_GET, require_http_methods
from filelock import FileLock, Timeout

from .forms import TrainingForm, DialogueForm
from .ml import jobs


def main(request):
    return render(request, "app_main/index.html", context={
        "title": "Diixo",
        "description": "Diixo description"})


def dashboard(request):
    return render(request, "app_main/dashboard.html", context={
        "title": "Diixo - Dashboard",
        "description": "Diixo dashboard description"})


def report(request):
    return render(request, "app_main/report.html", context={
        "title": "Diixo - Reports",
        "description": "Reports output",
    })


@require_http_methods(["GET", "POST"])
def training(request):
    form = TrainingForm(request.POST if request.method == "POST" else None)
    error = ""
    if request.method == "POST":
        try:
            if request.POST.get("action") == "stop":
                jobs.stop(request.POST.get("run_id", ""))
                return redirect("app_main:training")
            if form.is_valid():
                jobs.start(dict(form.cleaned_data))
                return redirect("app_main:training")
        except (ValueError, OSError, RuntimeError) as exc:
            error = str(exc)
    return render(request, "app_main/training.html", {"form": form, "error": error})


@require_GET
def training_status(request):
    runs = jobs.runs()
    return JsonResponse({"runs": runs, "log": jobs.log_tail(runs[0]["id"]) if runs else ""})


@require_http_methods(["GET", "POST"])
def dialogue(request):
    if request.method == "GET":
        # Establish the cookie before the first reply, so other tabs and reset
        # requests refer to the same server-side session while it is generated.
        request.session.get("dialogue")
        if not request.session.session_key:
            request.session["_dialogue_initialized"] = True
        return _dialogue_response(request)

    lock_directory = jobs.root() / ".dialogue-locks"
    lock_directory.mkdir(parents=True, exist_ok=True)
    session_key = request.session.session_key
    lock_name = hashlib.sha256((session_key or "new-session").encode()).hexdigest()
    lock = FileLock(lock_directory / f"{lock_name}.lock", timeout=120, thread_local=False)
    try:
        lock.acquire()
    except Timeout:
        return HttpResponse("Another dialogue request is still running. Please retry shortly.", status=409)
    try:
        # A preceding request may have changed the session while we waited.
        request.session = request.session.__class__(session_key=session_key)
        request.session.get("dialogue")
        if not request.session.session_key:
            request.session["_dialogue_initialized"] = True
        response = _dialogue_response(request)
    except BaseException:
        lock.release()
        raise
    # SessionMiddleware saves after the view returns. Keep the lock through
    # that save; Django closes the response after middleware/server processing.
    response._resource_closers.append(lock.release)
    return response


def _dialogue_response(request):
    available = [run for run in jobs.runs() if run["ready"]]
    conversation = request.session.get("dialogue", {})
    if request.method == "POST" and request.POST.get("action") == "reset":
        request.session.pop("dialogue", None)
        return redirect("app_main:dialogue")
    form = DialogueForm(request.POST if request.method == "POST" else None, available=available,
                        initial={"run_id": conversation.get("run_id"),
                                 **({"persona": conversation["persona"]} if conversation else {})})
    error = ""
    if request.method == "POST" and form.is_valid():
        values = form.cleaned_data
        history = conversation.get("history", [])
        if any(conversation.get(key) != values[key] for key in ("run_id", "persona")):
            history = []
        try:
            from .ml.dialogue import reply
            history = history[-10:] + [values["message"]]
            answer = reply(jobs.run_path(values["run_id"]) / "model",
                           values["persona"].splitlines(), history)
            if not answer:
                raise ValueError("The model returned an empty reply. Please try again.")
            request.session["dialogue"] = {"run_id": values["run_id"], "persona": values["persona"],
                                           "history": history + [answer]}
            return redirect("app_main:dialogue")
        except Exception as exc:
            error = f"Could not generate a reply: {exc}"
    return render(request, "app_main/dialogue.html", {
        "form": form, "available": available, "history": conversation.get("history", []), "error": error,
    })


def datasets(request):
    data_path = settings.BASE_DIR / "data" / "datasets.json"
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    if request.method == "POST":
        action = request.POST.get("action", "add")
        if action == "delete":
            dataset_id = request.POST.get("dataset_id", "")
            if dataset_id:
                data["datasets"] = [
                    item for item in data.get("datasets", [])
                    if item.get("id") != dataset_id
                ]
                with open(data_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            return redirect("app_main:datasets")

        name = request.POST.get("name", "").strip()
        task = request.POST.get("task", "").strip()
        tags = request.POST.get("tags", "").strip()
        description = request.POST.get("description", "").strip()
        website = request.POST.get("website", "").strip()

        if name:
            items = data.setdefault("datasets", [])
            task_list = [t.strip() for t in task.split(",") if t.strip()] if task else []
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
            links = {"website": website} if website else {}

            if action == "edit":
                dataset_id = request.POST.get("dataset_id", "")
                for item in items:
                    if item.get("id") == dataset_id:
                        item["name"] = name
                        item["task"] = task_list
                        item["tags"] = tag_list
                        item["description"] = description
                        item["links"] = links
                        break
            else:
                existing_ids = {d.get("id") for d in items}
                new_id = slugify(name) or f"dataset-{len(items) + 1}"
                while new_id in existing_ids:
                    new_id = f"{new_id}-1"
                items.append({
                    "id": new_id,
                    "name": name,
                    "task": task_list,
                    "tags": tag_list,
                    "description": description,
                    "links": links,
                })

            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        return redirect("app_main:datasets")

    return render(request, "app_main/datasets.html", context={
        "title": "Diixo - Datasets",
        "description": "Diixo datasets description",
        "goal": data.get("goal", ""),
        "source": data.get("source", ""),
        "datasets": data.get("datasets", []),
    })
