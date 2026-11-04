"""Testes de integração e contrato para pair_stats(prices_a, prices_b, n_periods)."""

import numpy as np
import pandas as pd
import pytest

from statistical_engine import (
    DegenerateSeriesError,
    InsufficientDataError,
    InvalidSeriesError,
    beta_confidence_interval,
    beta_rotation,
    coefficient_of_variation,
    coefficient_t_test,
    dickey_fuller,
    fisher_test,
    half_life,
    historical_volatility,
    linear_regression,
    pair_stats,
    pearson_correlation,
    z_score,
)

REQUIRED_CONTRACT_KEYS = {
    # Identificação da configuração
    "n_periods",
    "with_time",
    "conf",
    "rolling_window",
    # Coeficientes e regressão
    "beta",
    "alpha",
    "t_coef",
    "se_alpha",
    "se_beta",
    "se_time",
    "df",
    "r_squared",
    "fitted_values",
    "residuals",
    "std_residual",
    # Intervalos de confiança e testes t
    "ic_beta_min",
    "ic_beta_max",
    "ic_alpha_min",
    "ic_alpha_max",
    "ic_time_min",
    "ic_time_max",
    "t_beta",
    "t_alpha",
    "t_time",
    "t_beta_level",
    "t_alpha_level",
    "t_time_level",
    # Correlação e Fisher
    "corr",
    "fisher_min",
    "fisher_max",
    # Estacionariedade e reversão à média
    "dickey_fuller",
    "cointegration_level",
    "is_cointegrated",
    "z_score_current",
    "z_scores",
    "half_life",
    # Estabilidade do Beta e Volatilidade
    "beta_rotation",
    "beta_rotation_mean",
    "beta_rotation_std",
    "cv_beta",
    "volatility_a",
    "volatility_b",
    "volatility_beta_rotation",
    "historical_volatility",
    # Diagnóstico e validade
    "validity",
    "is_valid",
    "warnings",
}


def test_pair_stats_cointegrated_synthetic_pair_and_module_consistency():
    """Valida pair_stats em um par sintético cointegrado controlado e verifica consistência exata com as funções individuais."""
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

    # 1. Verifica presença de todas as chaves do contrato e validade global
    assert REQUIRED_CONTRACT_KEYS.issubset(stats_dict.keys())
    assert stats_dict["is_valid"] is True
    assert all(stats_dict["validity"].values())
    assert stats_dict["warnings"] == []
    assert stats_dict["n_periods"] == n_periods
    assert stats_dict["with_time"] is False

    # 2. Verifica consistência exata com cada módulo individual na mesma amostra [-n_periods:]
    win_a = prices_a[-n_periods:]
    win_b = prices_b[-n_periods:]
    reg = linear_regression(win_a, win_b, with_time=False)
    ic_b_min, ic_b_max = beta_confidence_interval(
        reg["beta"], reg["se_beta"], n=n_periods, k=1, conf=0.99
    )
    t_b = coefficient_t_test(
        reg["beta"], reg["se_beta"], n=n_periods, k=1, tail="right", conf=0.99
    )
    corr = pearson_correlation(win_a, win_b)
    f_min, f_max = fisher_test(corr, n=n_periods, conf=0.99)
    df_res = dickey_fuller(reg["residuals"])
    z_ser = z_score(reg["residuals"])
    hl = half_life(reg["residuals"])
    b_rot = beta_rotation(win_a, win_b, window=60, with_time=False)
    cv_b = coefficient_of_variation(b_rot, ddof=1)
    vol_a = historical_volatility(win_a, window=(10, 21, 42, 63), scale="annualized")

    assert np.isclose(stats_dict["beta"], reg["beta"], rtol=1e-12)
    assert np.isclose(stats_dict["alpha"], reg["alpha"], rtol=1e-12)
    assert np.allclose(stats_dict["fitted_values"], reg["fitted_values"], rtol=1e-12)
    assert np.allclose(stats_dict["residuals"], reg["residuals"], rtol=1e-12)
    assert np.isclose(stats_dict["ic_beta_min"], ic_b_min, rtol=1e-12)
    assert np.isclose(stats_dict["ic_beta_max"], ic_b_max, rtol=1e-12)
    assert np.isclose(stats_dict["t_beta"], t_b["t_stat"], rtol=1e-12)
    assert np.isclose(stats_dict["corr"], corr, rtol=1e-12)
    assert np.isclose(stats_dict["fisher_min"], f_min, rtol=1e-12)
    assert np.isclose(stats_dict["fisher_max"], f_max, rtol=1e-12)
    assert np.isclose(stats_dict["dickey_fuller"], df_res["tau"], rtol=1e-12)
    assert stats_dict["cointegration_level"] == df_res["cointegration_level"] == 99
    assert stats_dict["is_cointegrated"] is True
    assert np.isclose(stats_dict["z_score_current"], z_ser[-1], rtol=1e-12)
    assert np.isclose(stats_dict["half_life"], hl, rtol=1e-12)
    assert np.allclose(stats_dict["beta_rotation"], b_rot, rtol=1e-12)
    assert np.isclose(stats_dict["cv_beta"], cv_b, rtol=1e-12)
    for w in (10, 21, 42, 63):
        assert np.isclose(stats_dict["volatility_a"][w], vol_a[w], rtol=1e-12)

    # 3. Verifica convenção fixa A (dependente Y) vs B (independente X): A x B != B x A
    stats_inverted = pair_stats(prices_b, prices_a, n_periods=n_periods, with_time=False)
    assert not np.isclose(stats_dict["beta"], stats_inverted["beta"])
    assert np.isclose(stats_dict["beta"], true_beta, atol=0.05)


def test_pair_stats_non_cointegrated_and_invalid_half_life():
    """Valida pair_stats em séries não cointegradas com resíduo divergente (gamma >= 0 -> half_life indefinida)."""
    rng = np.random.default_rng(77)
    n = 120
    prices_b = 50.0 + np.cumsum(rng.normal(0.0, 0.5, n))
    # Resíduo não-estacionário com raiz explosiva (gamma > 0 na regressão de Dickey-Fuller)
    non_stationary_spread = np.linspace(0.0, 25.0, n) ** 1.5 + np.cumsum(
        rng.normal(0.2, 0.1, n)
    )
    prices_a = 10.0 + 0.5 * prices_b + non_stationary_spread

    stats_dict = pair_stats(prices_a, prices_b, n_periods=n)
    assert stats_dict["dickey_fuller"] > -3.28
    assert stats_dict["cointegration_level"] == 0
    assert stats_dict["is_cointegrated"] is False

    # Como gamma >= 0, a meia-vida não pode ser estimada validamente -> None (nunca 0.0!)
    assert stats_dict["half_life"] is None
    assert stats_dict["validity"]["half_life"] is False
    assert stats_dict["is_valid"] is False
    assert any("half_life" in w for w in stats_dict["warnings"])


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
    assert stats_dict["with_time"] is True
    assert stats_dict["t_coef"] is not None
    assert np.isclose(stats_dict["t_coef"], 0.08, atol=0.02)
    assert stats_dict["t_time"] is not None
    assert stats_dict["t_time"] > 2.39
    assert stats_dict["t_time_level"] == 99
    assert stats_dict["ic_time_min"] is not None
    assert stats_dict["ic_time_max"] is not None


def test_pair_stats_correlation_near_plus_or_minus_one():
    """Valida comportamento numérico quando a correlação está muito próxima de +1 ou -1."""
    rng = np.random.default_rng(321)
    n = 100
    prices_b = np.linspace(50.0, 100.0, n) + rng.normal(0.0, 0.5, n)
    tiny_stationary_noise = rng.normal(0.0, 0.01, n)

    # Correlação muito próxima de +1 (0.9999 < rho < 1.0)
    prices_a_pos = 10.0 + 2.0 * prices_b + tiny_stationary_noise
    res_pos = pair_stats(prices_a_pos, prices_b, n_periods=n)
    assert 0.999 < res_pos["corr"] < 1.0
    assert res_pos["validity"]["fisher_test"] is True
    assert res_pos["fisher_min"] is not None and res_pos["fisher_min"] > 0.99
    assert res_pos["fisher_max"] is not None and res_pos["fisher_max"] <= 1.0

    # Correlação muito próxima de -1 (-1.0 < rho < -0.9999)
    prices_a_neg = 250.0 - 2.0 * prices_b + tiny_stationary_noise
    res_neg = pair_stats(prices_a_neg, prices_b, n_periods=n)
    assert -1.0 < res_neg["corr"] < -0.999
    assert res_neg["validity"]["fisher_test"] is True
    assert res_neg["fisher_min"] is not None and res_neg["fisher_min"] >= -1.0
    assert res_neg["fisher_max"] is not None and res_neg["fisher_max"] < -0.99


def test_pair_stats_auxiliary_stats_unavailability_and_warnings():
    """Valida tratamento explícito quando janelas móveis ou janelas de volatilidade não possuem dados suficientes."""
    rng = np.random.default_rng(808)
    n = 40  # Menor que rolling_window padrão (60) e menor que janelas de VH 42 e 63
    prices_b = 30.0 + np.cumsum(rng.normal(0.05, 0.3, n))
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.4 * e[i - 1] + rng.normal(0.0, 0.2)
    prices_a = 5.0 + 0.9 * prices_b + e

    res = pair_stats(prices_a, prices_b, n_periods=n, rolling_window=60)

    # Regressão e Dickey-Fuller funcionam normalmente para n=40
    assert res["validity"]["regression"] is True
    assert res["validity"]["dickey_fuller"] is True
    assert res["validity"]["z_score"] is True

    # Beta rotation (janela 60 > 40) e cv_beta ficam None (NUNCA 0.0!)
    assert res["beta_rotation"] is None
    assert res["cv_beta"] is None
    assert res["validity"]["beta_rotation"] is False
    assert res["validity"]["cv_beta"] is False

    # Volatilidade histórica disponível para 10 e 21, mas None para 42 e 63
    assert res["volatility_a"][10] is not None
    assert res["volatility_a"][21] is not None
    assert res["volatility_a"][42] is None
    assert res["volatility_a"][63] is None
    assert res["validity"]["historical_volatility"] is False

    # O par não é classificado como totalmente válido quando faltam estatísticas auxiliares
    assert res["is_valid"] is False
    assert any("beta_rotation" in w for w in res["warnings"])
    assert any("cv_beta" in w for w in res["warnings"])
    assert any("volatility_a[42]" in w for w in res["warnings"])

    # Se rolling_window == n_periods (ex: 40), beta_rotation tem 1 ponto, mas cv_beta exige >= 2 pontos -> None
    res_single_rot = pair_stats(prices_a, prices_b, n_periods=n, rolling_window=n)
    assert res_single_rot["validity"]["beta_rotation"] is True
    assert res_single_rot["beta_rotation"].shape == (1,)
    assert res_single_rot["cv_beta"] is None
    assert res_single_rot["validity"]["cv_beta"] is False


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
    """Valida rejeição explícita para séries constantes, desvio-padrão residual zero, NaN/Inf, índices incompatíveis e dados insuficientes."""
    rng = np.random.default_rng(1)
    valid_x = 20.0 + np.cumsum(rng.normal(0.0, 0.5, 80))
    valid_y = 5.0 + 0.7 * valid_x + rng.normal(0.0, 0.2, 80)

    # 1. Série constante em B ou em A
    with pytest.raises(DegenerateSeriesError):
        pair_stats(valid_y, np.full(80, 10.0), n_periods=70)
    with pytest.raises(DegenerateSeriesError):
        pair_stats(np.full(80, 10.0), valid_x, n_periods=70)

    # 2. Desvio-padrão dos resíduos igual a zero (relação linear sem ruído)
    exact_y = 5.0 + 0.7 * valid_x
    with pytest.raises(DegenerateSeriesError, match="zero"):
        pair_stats(exact_y, valid_x, n_periods=70)

    # 3. Dados insuficientes para n_periods ou n_periods muito curto
    with pytest.raises(InsufficientDataError):
        pair_stats(valid_y[:50], valid_x[:50], n_periods=70)
    with pytest.raises(InsufficientDataError):
        pair_stats(valid_y, valid_x, n_periods=3)

    # 4. Valores ausentes (NaN) ou infinitos (Inf)
    y_with_nan = valid_y.copy()
    y_with_nan[10] = np.nan
    with pytest.raises(InvalidSeriesError, match="NaN"):
        pair_stats(y_with_nan, valid_x, n_periods=70)

    x_with_inf = valid_x.copy()
    x_with_inf[5] = np.inf
    with pytest.raises(InvalidSeriesError, match="Inf"):
        pair_stats(valid_y, x_with_inf, n_periods=70)

    # 5. Comprimentos ou índices incompatíveis
    with pytest.raises(InvalidSeriesError, match="mesmo comprimento"):
        pair_stats(valid_y, valid_x[:-1], n_periods=70)

    idx_a = pd.date_range("2025-01-01", periods=80, freq="B")
    idx_b = pd.date_range("2025-01-02", periods=80, freq="B")
    ser_a = pd.Series(valid_y, index=idx_a)
    ser_b = pd.Series(valid_x, index=idx_b)
    with pytest.raises(InvalidSeriesError, match="índices"):
        pair_stats(ser_a, ser_b, n_periods=70)
