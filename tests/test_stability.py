"""Testes unitários para o módulo de Estabilidade e Volatilidade (Beta Rotation, CV e Volatilidade Histórica)."""

import numpy as np
import pytest

from statistical_engine.exceptions import (
    DegenerateSeriesError,
    InsufficientDataError,
    InvalidSeriesError,
)
from statistical_engine.stability import (
    beta_rotation,
    coefficient_of_variation,
    historical_volatility,
)


def test_manual_example_coefficient_of_variation_page_65():
    """Valida as duas tabelas numéricas de Coeficiente de Variação da página 65 do Manual:
    Tabela 1 (estável): [3.68, 3.65, 3.60, 3.51, 3.48, 3.42, 3.46, 3.50, 3.06, 4.11] -> ~7,01% (ddof=0) / 7,38% (ddof=1)
    Tabela 2 (instável): [1.17, 1.19, 1.40, 1.19, 1.18, 1.20, 1.15, 0.88, 0.33, 0.38] -> ~34,40% (ddof=0) / 36,31% (ddof=1)
    """
    betas_stable = [3.68, 3.65, 3.60, 3.51, 3.48, 3.42, 3.46, 3.50, 3.06, 4.11]
    betas_unstable = [1.17, 1.19, 1.40, 1.19, 1.18, 1.20, 1.15, 0.88, 0.33, 0.38]

    cv1_pop = coefficient_of_variation(betas_stable, ddof=0, as_percentage=True)
    cv2_pop = coefficient_of_variation(betas_unstable, ddof=0, as_percentage=True)
    assert np.isclose(cv1_pop, 7.003, atol=0.05)
    assert np.isclose(cv2_pop, 34.446, atol=0.06)

    cv1_sample = coefficient_of_variation(betas_stable, ddof=1)
    cv2_sample = coefficient_of_variation(betas_unstable, ddof=1)
    assert cv1_sample < 0.15  # Baixa dispersão (< 15%)
    assert cv2_sample > 0.30  # Alta dispersão (> 30%)


def test_manual_example_historical_volatility_page_52():
    """Valida a tabela completa de 22 preços (21 retornos LN) da página 52-53 do Manual:
    Datas: 13/01/2010 a 12/02/2010
    VH diária (10 dias) = 2,40%
    VH diária (21 dias) = 2,04%
    VH anualizada (21 dias) = 2,04% * sqrt(252) = 32,36%
    """
    prices = [
        46.51,
        46.38,
        46.14,
        46.70,
        47.25,
        46.41,
        44.76,
        43.75,
        42.60,
        41.96,
        42.15,
        42.14,
        43.60,
        43.66,
        43.52,
        41.25,
        40.80,
        41.25,
        42.23,
        42.20,
        43.00,
        42.74,
    ]

    vh_daily = historical_volatility(prices, window=(10, 21), scale="daily")
    assert np.isclose(vh_daily[10], 0.024046, atol=1e-4)  # 2,40%
    assert np.isclose(vh_daily[21], 0.020388, atol=1e-4)  # 2,04%

    vh_ann_21 = historical_volatility(prices, window=21, scale="annualized")
    assert np.isclose(vh_ann_21, 0.323646, atol=1e-3)  # 32,36%

    vh_period = historical_volatility(prices, window=(10, 21), scale="period")
    assert np.isclose(vh_period[10], vh_daily[10] * np.sqrt(10), rtol=1e-12)
    assert np.isclose(vh_period[21], vh_daily[21] * np.sqrt(21), rtol=1e-12)


def test_beta_rotation_rolling_window():
    """Em relação linear exata Y = alpha + beta * X, o beta_rotation é constante igual a beta."""
    rng = np.random.default_rng(99)
    n = 120
    window = 70
    x = np.linspace(20.0, 35.0, n) + rng.normal(0.0, 0.5, n)
    y = 3.0 + 0.75 * x

    rolling_betas = beta_rotation(y, x, window=window, with_time=False)
    assert rolling_betas.shape == (n - window + 1,)
    assert np.allclose(rolling_betas, 0.75, atol=1e-12)


def test_stability_edge_cases_and_errors():
    # beta_rotation com janela maior que a série
    with pytest.raises(InsufficientDataError):
        beta_rotation(np.arange(10.0), np.arange(10.0), window=20)

    # coefficient_of_variation com média zero
    with pytest.raises(DegenerateSeriesError, match="média da série é zero"):
        coefficient_of_variation([-1.0, 0.0, 1.0])

    # historical_volatility com valores negativos ou zero
    with pytest.raises(InvalidSeriesError, match="positivos"):
        historical_volatility([10.0, -5.0, 12.0, 11.0], window=2)
    with pytest.raises(InsufficientDataError):
        historical_volatility([10.0, 11.0, 12.0], window=10)
