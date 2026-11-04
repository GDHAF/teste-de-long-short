"""Views Django para a tela de demonstração e Dashboard inicial do sistema Long & Short."""

from __future__ import annotations

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from .services import build_dummy_dashboard_context


@require_GET
def dashboard_view(request: HttpRequest) -> HttpResponse:
    """Renderiza o protótipo inicial de Dashboard com resultados dummy da statistical_engine."""
    context = build_dummy_dashboard_context()
    return render(request, "engine_lab/dashboard.html", context)
