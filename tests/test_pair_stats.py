"""Testes de integração e contrato para pair_stats(prices_a, prices_b, n_periods)."""

import numpy as np
import pytest

from statistical_engine import (
    DegenerateSeriesError,
    InsufficientDataError,
    InvalidSeriesError,
    pair_stats,
)

REQUIRED_CONTRACT_KEYS = {
    "beta",
    "alpha",
    "t_coef",
    "residuals",
    "dickey_fuller",
    "cointegration_level",
    "z_score_current",
    "half_life",
    "fisher_min",
    "fisher_max",
    "ic_beta_min",
    "ic_beta_max",
    "t_beta",
    "t_alpha",
    "t_time",
    "std_residual",
    "beta_rotation",
    "cv_beta",
}


def test_pair_stats_cointegrated_synthetic_pair():
    """Valida pair_stats em um par sintético cointegrado controlado (passeio aleatório comum + resíduo O.-U.)."""
    rng = np.random.default_rng(2026)
    n_total = 250
    n_periods = 120

    # Ativo B (independente X): passeio aleatório estritamente positivo
    steps_b = rng.normal(0.05, 0.4, n_total)
    prices_b = 30.0 + np.cumsum(steps_b)

    # Resíduo estacionário Ornstein-Uhlenbeck com reversão rápida (theta = 0.35 -> HL ~ 1.98 dias)
    theta = 0.35
    e = np.zeros(n_total)
    for t in range(1, n_total):
        e[t] = (1.0 - theta) * e[t - 1] + rng.normal(0.0, 0.25)

    # Força uma distorção de entrada no último período (> 2 desvios padrão)
    e[-1] = 3.0 * np.std(e[-n_periods:-1], ddof=1)

    true_alpha = 5.0
    true_beta = 0.80
    prices_a = true_alpha + true_beta * prices_b + e

    stats_dict = pair_stats(prices_a, prices_b, n_periods=n_periods, with_time=False)

    # Verifica presença de todas as chaves do contrato
    assert REQUIRED_CONTRACT_KEYS.issubset(stats_dict.keys())

    # Coeficientes próximos aos valores reais geradores
    assert np.isclose(stats_dict["beta"], true_beta, atol=0.05)
    assert np.isclose(stats_dict["alpha"], true_alpha, atol=2.0)
    assert stats_dict["t_coef"] is None
    assert stats_dict["t_time"] is None

    # Estacionariedade e cointegração Engle-Granger
    assert stats_dict["dickey_fuller"] < -4.32
    assert stats_dict["cointegration_level"] == 99

    # Intervalos de confiança positivos (filtros do manual)
    assert stats_dict["ic_beta_min"] > 0.0
    assert stats_dict["ic_beta_max"] > stats_dict["ic_beta_min"]
    assert stats_dict["fisher_min"] > 0.0
    assert stats_dict["fisher_max"] > stats_dict["fisher_min"]

    # Z-score atual reflete o choque positivo (> +2 desvios)
    assert stats_dict["z_score_current"] > 2.0
    assert stats_dict["residuals"].shape == (n_periods,)

    # Meia-vida finita e curta (< 10 dias)
    assert 0.5 < stats_dict["half_life"] < 10.0

    # Beta rotation estável (CV < 15%)
    assert stats_dict["beta_rotation"].shape == (n_periods - 60 + 1,)
    assert stats_dict["cv_beta"] < 0.15


def test_pair_stats_non_cointegrated_synthetic_pair():
    """Valida pair_stats em séries não cointegradas (resíduo com tendência divergente)."""
    rng = np.random.default_rng(77)
    n = 120
    prices_b = 50.0 + np.cumsum(rng.normal(0.0, 0.5, n))
    # Resíduo não-estacionário com raiz explosiva / drift quadrático
    non_stationary_spread = np.linspace(0.0, 25.0, n) ** 1.5 + np.cumsum(
        rng.normal(0.2, 0.1, n)
    )
    prices_a = 10.0 + 0.5 * prices_b + non_stationary_spread

    stats_dict = pair_stats(prices_a, prices_b, n_periods=n)
    assert stats_dict["dickey_fuller"] > -3.28
    assert stats_dict["cointegration_level"] == 0


def test_pair_stats_multiple_regression_with_time():
    """Valida pair_stats com with_time=True quando há tendência temporal determinística."""
    rng = np.random.default_rng(555)
    n = 100
    t = np.arange(1, n + 1, dtype=np.float64)
    prices_b = 20.0 + np.cumsum(rng.normal(0.0, 0.5, n))
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.5 * e[i - 1] + rng.normal(0.0, 0.1)

    prices_a = -4.0 + 1.2 * prices_b + 0.08 * t + e

    stats_dict = pair_stats(prices_a, prices_b, n_periods=n, with_time=True)
    assert stats_dict["t_coef"] is not None
    assert np.isclose(stats_dict["t_coef"], 0.08, atol=0.02)
    assert stats_dict["t_time"] is not None
    assert stats_dict["t_time"] > 2.39
    assert stats_dict["t_time_level"] == 99


def test_pair_stats_slices_last_n_periods_consistently():
    """Garante que pair_stats usa exatamente os últimos n_periods das séries."""
    rng = np.random.default_rng(909)
    prices_b = 40.0 + np.cumsum(rng.normal(0.0, 0.5, 200))
    prices_a = 2.0 + 0.9 * prices_b + rng.normal(0.0, 0.3, 200)

    res_from_full = pair_stats(prices_a, prices_b, n_periods=100)
    res_from_slice = pair_stats(prices_a[-100:], prices_b[-100:], n_periods=100)

    assert np.isclose(res_from_full["beta"], res_from_slice["beta"], rtol=1e-12)
    assert np.isclose(res_from_full["alpha"], res_from_slice["alpha"], rtol=1e-12)
    assert np.isclose(
        res_from_full["dickey_fuller"], res_from_slice["dickey_fuller"], rtol=1e-12
    )
    assert np.allclose(
        res_from_full["residuals"], res_from_slice["residuals"], rtol=1e-12
    )


def test_pair_stats_error_handling_on_invalid_inputs():
    rng = np.random.default_rng(1)
    valid_x = 20.0 + np.cumsum(rng.normal(0.0, 0.5, 80))
    valid_y = 5.0 + 0.7 * valid_x + rng.normal(0.0, 0.2, 80)

    # Série constante em B
    with pytest.raises(DegenerateSeriesError):
        pair_stats(valid_y, np.full(80, 10.0), n_periods=70)

    # Série constante em A
    with pytest.raises(DegenerateSeriesError):
        pair_stats(np.full(80, 10.0), valid_x, n_periods=70)

    # Dados insuficientes para n_periods
    with pytest.raises(InsufficientDataError):
        pair_stats(valid_y[:50], valid_x[:50], n_periods=70)

    # Valores ausentes (NaN)
    y_with_nan = valid_y.copy()
    y_with_nan[10] = np.nan
    with pytest.raises(InvalidSeriesError, match="NaN"):
        pair_stats(y_with_nan, valid_x, n_periods=70)
