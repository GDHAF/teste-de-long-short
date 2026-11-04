"""Roteamento principal de URLs do projeto Long & Short Analyzer."""

from django.urls import include, path

urlpatterns = [
    path("", include("engine_lab.urls")),
]
