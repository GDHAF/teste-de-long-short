"""Testes unitários para o módulo de Estacionariedade e Reversão à Média (Dickey-Fuller, Z-Score, Meia-Vida)."""

import numpy as np
import pytest

from statistical_engine.exceptions import (
    DegenerateSeriesError,
    InsufficientDataError,
    InvalidSeriesError,
)
from statistical_engine.stationarity import (
    ENGLE_GRANGER_CRITICAL_VALUES,
    classify_cointegration_level,
    dickey_fuller,
    half_life,
    z_score,
)


def test_manual_example_dickey_fuller_classification_page_42():
    """Valida o exemplo numérico da página 42 do Manual:
    gamma = -0,190; EP(gamma) = 0,048 => Tau = -0,190 / 0,048 = -3,958333...
    Como -4,32 < -3,958 < -3,67 => estacionário com 95% de confiança.
    Se DF = -4,50 => 99% de confiança.
    Se DF = -3,50 => 90% de confiança.
    Se DF > -3,28 => 0% (não estacionário).
    """
    tau_example = -0.190 / 0.048
    assert np.isclose(tau_example, -3.958333, atol=1e-5)
    assert classify_cointegration_level(tau_example) == 95
    assert classify_cointegration_level(-4.50) == 99
    assert classify_cointegration_level(-4.32) == 99
    assert classify_cointegration_level(-3.67) == 95
    assert classify_cointegration_level(-3.50) == 90
    assert classify_cointegration_level(-3.28) == 90
    assert classify_cointegration_level(-3.20) == 0
    assert classify_cointegration_level(0.50) == 0


def test_dickey_fuller_stationary_mean_reverting_series():
    """Série sintética AR(1) estacionária com forte reversão à média deve ser cointegrada a 99%."""
    rng = np.random.default_rng(101)
    n = 200
    # e_t = 0.65 * e_{t-1} + u_t => gamma = -0.35
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.65 * e[t - 1] + rng.normal(0.0, 0.5)

    res = dickey_fuller(e)
    assert res["tau"] < ENGLE_GRANGER_CRITICAL_VALUES[99]
    assert res["cointegration_level"] == 99
    assert res["is_stationary"] is True
    assert res["gamma"] < 0.0
    assert np.isclose(res["tau"], res["gamma"] / res["se_gamma"], rtol=1e-12)


def test_dickey_fuller_non_stationary_random_walk():
    """Série em passeio aleatório com drift positivo não deve rejeitar a raiz unitária."""
    rng = np.random.default_rng(2024)
    n = 150
    # Série explosiva / tendência: e_t = 1.02 * e_{t-1} + u_t => gamma > 0 => tau > 0
    e = np.zeros(n)
    e[0] = 10.0
    for t in range(1, n):
        e[t] = 1.02 * e[t - 1] + rng.normal(0.0, 0.1)

    res = dickey_fuller(e)
    assert res["tau"] > ENGLE_GRANGER_CRITICAL_VALUES[90]
    assert res["cointegration_level"] == 0
    assert res["is_stationary"] is False


def test_z_score_computation_and_properties():
    """O Z-Score dos resíduos deve ter média 0 e desvio padrão amostral 1."""
    residuals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    z = z_score(residuals)
    std_e = np.std(residuals, ddof=1)  # sqrt(2.5)
    assert np.allclose(z, residuals / std_e, rtol=1e-12)
    assert np.isclose(np.mean(z), 0.0, atol=1e-12)
    assert np.isclose(np.std(z, ddof=1), 1.0, rtol=1e-12)


def test_half_life_ornstein_uhlenbeck_synthetic():
    """Valida o cálculo da Meia-Vida ln(2)/theta em processo de Ornstein-Uhlenbeck controlado."""
    rng = np.random.default_rng(7)
    n = 500
    true_theta = 0.15  # Meia-vida teórica = ln(2) / 0.15 = 4.621 dias
    e = np.zeros(n)
    for t in range(1, n):
        # Delta(e_t) = -true_theta * e_{t-1} + u_t
        e[t] = e[t - 1] - true_theta * e[t - 1] + rng.normal(0.0, 0.2)

    hl_discrete = half_life(e, method="discrete")
    expected_hl = np.log(2.0) / true_theta
    assert np.isclose(hl_discrete, expected_hl, rtol=0.15)
    assert hl_discrete > 0.0

    hl_exact = half_life(e, method="exact_ou")
    assert hl_exact > 0.0


def test_stationarity_edge_cases_and_errors():
    # Série constante -> erro em dickey_fuller, z_score e half_life
    const = np.ones(20)
    with pytest.raises(DegenerateSeriesError):
        dickey_fuller(const)
    with pytest.raises(DegenerateSeriesError):
        z_score(const)
    with pytest.raises(DegenerateSeriesError):
        half_life(const)

    # Série divergente (gamma > 0) -> half_life levanta DegenerateSeriesError por padrão
    divergent = np.array([1.0, 2.0, 4.5, 9.0, 19.0, 40.0, 82.0])
    with pytest.raises(DegenerateSeriesError, match="reversão à média"):
        half_life(divergent, allow_non_stationary=False)
    assert np.isinf(half_life(divergent, allow_non_stationary=True))

    # Dados insuficientes e NaN
    with pytest.raises(InsufficientDataError):
        dickey_fuller([1.0, 2.0, 3.0])
    with pytest.raises(InvalidSeriesError):
        z_score([1.0, np.nan, 2.0])
