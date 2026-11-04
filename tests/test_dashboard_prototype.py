"""Testes automatizados para a tela simples de teste e protótipo de Dashboard (somente leitura)."""

from __future__ import annotations

import json
import os

import django
import numpy as np
import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from django.test import Client  # noqa: E402

from engine_lab.services import (  # noqa: E402
    DUMMY_CONFIDENCE_LEVEL,
    DUMMY_PRIMARY_PERIODS,
    DUMMY_ROLLING_WINDOW,
    DUMMY_SEED,
    DUMMY_TICKER_A,
    DUMMY_TICKER_B,
    DUMMY_TOTAL_OBSERVATIONS,
    MANUAL_PERIODS,
    VOLATILITY_WINDOWS,
    build_dummy_dashboard_context,
    generate_dummy_pair_data,
)
from statistical_engine import pair_stats  # noqa: E402


def test_generate_dummy_pair_data_is_deterministic_and_aligned() -> None:
    """Garante que os valores dummy sejam determinísticos, positivos e com índices alinhados."""
    d1 = generate_dummy_pair_data()
    d2 = generate_dummy_pair_data()

    series_a1 = d1["prices_a"]
    series_b1 = d1["prices_b"]
    series_a2 = d2["prices_a"]
    series_b2 = d2["prices_b"]

    assert len(series_a1) == DUMMY_TOTAL_OBSERVATIONS
    assert len(series_b1) == DUMMY_TOTAL_OBSERVATIONS
    assert series_a1.index.equals(series_b1.index)
    assert np.all(series_a1.to_numpy() > 0.0)
    assert np.all(series_b1.to_numpy() > 0.0)
    np.testing.assert_allclose(series_a1.to_numpy(), series_a2.to_numpy(), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(series_b1.to_numpy(), series_b2.to_numpy(), rtol=0.0, atol=0.0)
    assert d1["params"]["seed"] == DUMMY_SEED


def test_build_dummy_dashboard_context_matches_pair_stats_directly() -> None:
    """Verifica que todos os números do contexto do Dashboard vêm diretamente de pair_stats."""
    dummy = generate_dummy_pair_data()
    ctx = build_dummy_dashboard_context()

    expected_primary = pair_stats(
        prices_a=dummy["prices_a"],
        prices_b=dummy["prices_b"],
        n_periods=DUMMY_PRIMARY_PERIODS,
        with_time=False,
        conf=DUMMY_CONFIDENCE_LEVEL,
        rolling_window=DUMMY_ROLLING_WINDOW,
        volatility_windows=VOLATILITY_WINDOWS,
    )

    primary = ctx["primary_stats"]
    assert primary["is_valid"] is True
    assert primary["warnings"] == []
    assert pytest.approx(primary["alpha"], rel=1e-12) == expected_primary["alpha"]
    assert pytest.approx(primary["beta"], rel=1e-12) == expected_primary["beta"]
    assert pytest.approx(primary["dickey_fuller"], rel=1e-12) == expected_primary["dickey_fuller"]
    assert primary["cointegration_level"] == 99
    assert pytest.approx(primary["z_score_current"], rel=1e-12) == expected_primary["z_score_current"]
    assert pytest.approx(primary["half_life"], rel=1e-12) == expected_primary["half_life"]
    assert pytest.approx(primary["cv_beta"], rel=1e-12) == expected_primary["cv_beta"]

    # Verifica que o cenário único é claro: par cointegrado com gatilho operacional (|Z| >= 2.0)
    assert ctx["signal_summary"]["is_entry_signal"] is True
    assert abs(primary["z_score_current"]) >= 2.0

    # Verifica as 10 janelas do painel operacional do Manual
    assert len(ctx["multi_period_rows"]) == len(MANUAL_PERIODS)
    assert [r["n_periods"] for r in ctx["multi_period_rows"]] == list(MANUAL_PERIODS)
    assert ctx["panel_summary"]["cointegrated_count"] == 10

    # Verifica o JSON dos gráficos
    chart_data = json.loads(ctx["chart_payload_json"])
    assert len(chart_data["z_scores"]) == DUMMY_PRIMARY_PERIODS
    assert len(chart_data["prices_a"]) == DUMMY_PRIMARY_PERIODS
    assert len(chart_data["fitted_values"]) == DUMMY_PRIMARY_PERIODS


def test_dashboard_view_renders_readonly_without_inputs_or_forms() -> None:
    """Verifica que a tela renderiza com sucesso e não possui campos de inserção/alteração."""
    client = Client()
    for url in ("/", "/dashboard/", "/lab/"):
        response = client.get(url)
        assert response.status_code == 200
        html = response.content.decode("utf-8")

        # Verifica presença dos identificadores e resultados principais do par dummy
        assert DUMMY_TICKER_A in html
        assert DUMMY_TICKER_B in html
        assert "Dashboard de Cointegração" in html
        assert "Painel Operacional Multi-Período" in html
        assert "Somente Leitura" in html

        # Restrição estrita: nenhum campo de formulário, input, select ou textarea
        html_lower = html.lower()
        assert "<form" not in html_lower
        assert "<input" not in html_lower
        assert "<select" not in html_lower
        assert "<textarea" not in html_lower


def test_dashboard_view_rejects_post_requests() -> None:
    """Garante que a rota de demonstração seja estritamente somente leitura (GET)."""
    client = Client()
    response = client.post("/", data={"any": "value"})
    assert response.status_code == 405
