"""Testes unitários para o módulo de Inferência: IC(Beta), Testes t, Correlação de Pearson e Fisher."""

import numpy as np
import pytest

from statistical_engine.exceptions import (
    DegenerateSeriesError,
    InsufficientDataError,
    InvalidSeriesError,
)
from statistical_engine.inference import (
    beta_confidence_interval,
    coefficient_t_test,
    fisher_test,
    fisher_transform,
    inverse_fisher_transform,
    pearson_correlation,
)


def test_manual_example_beta_confidence_interval_page_34():
    """Valida o exemplo numérico da página 34 do Manual:
    beta = 0,811; EP(beta) = 0,091; 99% de confiança (t_60gl = 2,66)
    IC(beta; 99%) = [0,811 - 2,66 * 0,091 ; 0,811 + 2,66 * 0,091] = [0,5689 ; 1,0530]
    """
    ic_min_manual, ic_max_manual = beta_confidence_interval(
        beta=0.811,
        se=0.091,
        n=62,
        k=1,
        conf=0.99,
        use_manual_critical_values=True,
    )
    assert np.isclose(ic_min_manual, 0.56894, atol=1e-4)
    assert np.isclose(ic_max_manual, 1.05306, atol=1e-4)

    # Também valida com distribuição t exata (scipy.stats.t com df = 60)
    ic_min_exact, ic_max_exact = beta_confidence_interval(
        beta=0.811,
        se=0.091,
        n=62,
        k=1,
        conf=0.99,
        use_manual_critical_values=False,
    )
    assert np.isclose(ic_min_exact, 0.5689, atol=1e-4)
    assert np.isclose(ic_max_exact, 1.0531, atol=1e-4)


def test_manual_example_t_test_beta_page_35():
    """Valida o exemplo da página 35 do Manual:
    beta = 0,81184; ep(beta) = 0,09113; 99% de confiança (unilateral à direita, t_crit = 2,390)
    t_calc = 0,81184 / 0,09113 = 8,90859 > 2,39 => significativo a 99%.
    """
    res = coefficient_t_test(
        coef=0.81184,
        se=0.09113,
        n=120,
        k=2,
        tail="right",
        conf=0.99,
        use_manual_critical_values=True,
    )
    assert np.isclose(res["t_stat"], 0.81184 / 0.09113, rtol=1e-10)
    assert np.isclose(res["t_stat"], 8.9086, atol=1e-3)
    assert np.isclose(res["critical_value"], 2.390, atol=1e-6)
    assert res["is_significant"] is True
    assert res["confidence_level"] == 99


def test_manual_example_t_test_time_page_36():
    """Valida o exemplo da página 36 do Manual:
    T = 0,018074844; ep(T) = 0,010169671; 99% de confiança (unilateral à direita, t_crit = 2,390)
    t_calc = 1,7773 < 2,39 => não significativo a 99% (porém > 1,671 => significativo a 95%).
    """
    res = coefficient_t_test(
        coef=0.018074844,
        se=0.010169671,
        n=120,
        k=2,
        tail="right",
        conf=0.99,
        use_manual_critical_values=True,
    )
    assert np.isclose(res["t_stat"], 1.777328, atol=1e-4)
    assert res["is_significant"] is False
    assert res["confidence_level"] == 95


def test_manual_example_t_test_alpha_page_37():
    """Valida o exemplo da página 37 do Manual:
    alpha = -9,62; ep(alpha) = 4,14; 95% de confiança (bilateral, t_crit = 2,00)
    t_calc = -9,62 / 4,14 = -2,3236 => pertence à RC ]-inf, -2,00] U [2,00, +inf[ => significativo a 95%.
    """
    res = coefficient_t_test(
        coef=-9.62,
        se=4.14,
        n=120,
        k=2,
        tail="two-sided",
        conf=0.95,
        use_manual_critical_values=True,
    )
    assert np.isclose(res["t_stat"], -9.62 / 4.14, rtol=1e-10)
    assert np.isclose(res["t_stat"], -2.3237, atol=1e-3)
    assert np.isclose(res["critical_value"], 2.00, atol=1e-6)
    assert res["is_significant"] is True
    assert res["confidence_level"] == 95


def test_manual_example_fisher_transform_page_32():
    """Valida o exemplo numérico da página 32 do Manual:
    Correlação rho = 0,2552; Período = 120; Confiança = 99%
    F(rho) = 0,2609
    IC(F(rho), 99%) = [0,0228 ; 0,4989]
    IC(rho, 99%) = [2,28% ; 46,13%]
    """
    f_val = fisher_transform(0.2552)
    assert np.isclose(f_val, 0.260947, atol=1e-4)
    assert np.isclose(inverse_fisher_transform(f_val), 0.2552, atol=1e-10)

    f_min, f_max = fisher_test(
        corr=0.2552, n=120, conf=0.99, use_manual_critical_values=True
    )
    assert np.isclose(f_min, 0.0228, atol=1e-3)
    assert np.isclose(f_max, 0.4613, atol=1e-3)

    f_min_exact, f_max_exact = fisher_test(
        corr=0.2552, n=120, conf=0.99, use_manual_critical_values=False
    )
    assert np.isclose(f_min_exact, 0.0228, atol=1e-3)
    assert np.isclose(f_max_exact, 0.4614, atol=1e-3)


def test_pearson_correlation_direct_and_log_returns():
    x = np.array([10.0, 12.0, 15.0, 14.0, 18.0, 20.0])
    y = 2.0 * x + 5.0
    assert np.isclose(pearson_correlation(y, x, use_log_returns=False), 1.0, atol=1e-12)

    # Em relação exponencial proporcional, log-retornos têm correlação 1.0
    y_prop = 3.0 * x
    assert np.isclose(
        pearson_correlation(y_prop, x, use_log_returns=True), 1.0, atol=1e-12
    )


def test_inference_edge_cases_and_errors():
    # Erro padrão zero
    with pytest.raises(DegenerateSeriesError):
        beta_confidence_interval(beta=0.5, se=0.0, n=60, k=1)
    with pytest.raises(DegenerateSeriesError):
        coefficient_t_test(coef=0.5, se=0.0, n=60, k=1)

    # Graus de liberdade insuficientes
    with pytest.raises(InsufficientDataError):
        beta_confidence_interval(beta=0.5, se=0.1, n=2, k=1)

    # Correlação constante -> erro
    with pytest.raises(DegenerateSeriesError):
        pearson_correlation([10.0, 10.0, 10.0], [1.0, 2.0, 3.0])

    # Fisher com |rho| = 1.0 -> divisão por zero / infinito
    with pytest.raises(DegenerateSeriesError):
        fisher_transform(1.0)
    with pytest.raises(DegenerateSeriesError):
        fisher_transform(-1.0)
    with pytest.raises(InvalidSeriesError):
        fisher_transform(1.2)

    # Fisher com n <= 3 -> erro de amostra insuficiente
    with pytest.raises(InsufficientDataError):
        fisher_test(corr=0.5, n=3, conf=0.99)
