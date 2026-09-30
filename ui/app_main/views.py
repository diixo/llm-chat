import json

from django.conf import settings
from django.shortcuts import render, redirect
from django.utils.text import slugify


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
