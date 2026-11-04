"""Configuração da aplicação Django engine_lab."""

from django.apps import AppConfig


class EngineLabConfig(AppConfig):
    """Configuração do app de Dashboard e demonstração da Engine."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "engine_lab"
    verbose_name = "Long & Short Dashboard"
