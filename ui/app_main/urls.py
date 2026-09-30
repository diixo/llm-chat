
from django.urls import path
from . import views

app_name = "app_main"

urlpatterns = [
    path("", views.main, name="main"),
    path("datasets", views.datasets, name="datasets"),
    path("dashboard", views.dashboard, name="dashboard"),
    path("report", views.report, name="report"),
]
