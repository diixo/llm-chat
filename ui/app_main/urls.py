
from django.urls import path
from . import views

app_name = "app_main"

urlpatterns = [
    path("", views.main, name="main"),
    path("datasets", views.datasets, name="datasets"),
    path("training", views.training, name="training"),
    path("training/status", views.training_status, name="training_status"),
    path("personas", views.personas, name="personas"),
    path("dialogue", views.dialogue, name="dialogue"),
    path("dashboard", views.dashboard, name="dashboard"),
    path("report", views.report, name="report"),
]
