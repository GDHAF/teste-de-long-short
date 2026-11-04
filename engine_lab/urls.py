"""Rotas da aplicação engine_lab (Dashboard de demonstração)."""

from django.urls import path

from .views import dashboard_view

app_name = "engine_lab"

urlpatterns = [
    path("", dashboard_view, name="dashboard"),
    path("dashboard/", dashboard_view, name="dashboard_alias"),
    path("lab/", dashboard_view, name="lab_alias"),
]
