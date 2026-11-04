"""Testes unitários para o módulo de Regressão Linear Simples e Múltipla (OLS)."""

import numpy as np
import pytest

from statistical_engine.exceptions import (
    DegenerateSeriesError,
    InsufficientDataError,
    InvalidSeriesError,
)
from statistical_engine.regression import (
    compute_fitted_values,
    compute_residuals,
    linear_regression,
)


def test_manual_example_simple_regression_residual_page_22():
    """Valida o exemplo numérico explícito da página 22 do Manual:
    BBAS3 = -2,45 + 0,65 * CSAN3.
    Para CSAN3 = 42,00 => BBAS3 estimado = 24,85.
    Se BBAS3 real = 25,00 => Resíduo = 25,00 - 24,85 = 0,15.
    """
    fitted = compute_fitted_values([42.00], alpha=-2.45, beta=0.65)
    assert np.isclose(fitted[0], 24.85, atol=1e-12)

    residual = compute_residuals([25.00], [42.00], alpha=-2.45, beta=0.65)
    assert np.isclose(residual[0], 0.15, atol=1e-12)


def test_simple_linear_regression_exact_synthetic_relation():
    """Valida Y = alpha + beta * X + e contra as fórmulas analíticas de MQO (p. 22)."""
    rng = np.random.default_rng(42)
    n = 120
    x = np.linspace(30.0, 45.0, n) + rng.normal(0.0, 0.5, n)
    true_alpha = -2.45
    true_beta = 0.65
    noise = rng.normal(0.0, 0.2, n)
    y = true_alpha + true_beta * x + noise

    res = linear_regression(y, x, with_time=False)

    # Fórmula analítica do manual (p. 22):
    # beta = (T * sum(X*Y) - sum(X)*sum(Y)) / (T * sum(X^2) - (sum(X))^2)
    beta_manual = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
        n * np.sum(x**2) - (np.sum(x)) ** 2
    )
    alpha_manual = np.mean(y) - beta_manual * np.mean(x)
    fitted_manual = alpha_manual + beta_manual * x
    resid_manual = y - fitted_manual

    assert np.isclose(res["beta"], beta_manual, rtol=1e-12)
    assert np.isclose(res["alpha"], alpha_manual, rtol=1e-12)
    assert res["t_coef"] is None
    assert res["se_time"] is None
    assert np.allclose(res["fitted_values"], fitted_manual, rtol=1e-12)
    assert np.allclose(res["residuals"], resid_manual, rtol=1e-12)
    assert np.isclose(np.mean(res["residuals"]), 0.0, atol=1e-12)
    assert np.isclose(res["std_residual"], np.std(resid_manual, ddof=1), rtol=1e-12)
    assert res["n_obs"] == 120
    assert res["k"] == 1
    assert res["df"] == 118


def test_multiple_linear_regression_with_time_trend():
    """Valida Y_t = alpha + beta * X_t + T * t + e_t com t = 1..N (p. 23-24)."""
    rng = np.random.default_rng(123)
    n = 120
    t = np.arange(1, n + 1, dtype=np.float64)
    x = 40.0 + np.cumsum(rng.normal(0.0, 0.4, n))
    true_alpha = -9.624436273
    true_beta = 0.811841144
    true_t_coef = 0.018074844
    noise = rng.normal(0.0, 0.05, n)
    y = true_alpha + true_beta * x + true_t_coef * t + noise

    res = linear_regression(y, x, with_time=True)

    # Validação algébrica matricial: theta_hat = (X^T X)^(-1) X^T Y
    design = np.column_stack((np.ones(n), x, t))
    theta_hat = np.linalg.inv(design.T @ design) @ (design.T @ y)
    resid_matrix = y - design @ theta_hat
    s2 = float(np.sum(resid_matrix**2) / (n - 3))
    cov_theta = s2 * np.linalg.inv(design.T @ design)
    se_matrix = np.sqrt(np.diag(cov_theta))

    assert np.isclose(res["alpha"], theta_hat[0], rtol=1e-10)
    assert np.isclose(res["beta"], theta_hat[1], rtol=1e-10)
    assert np.isclose(res["t_coef"], theta_hat[2], rtol=1e-10)
    assert np.isclose(res["se_alpha"], se_matrix[0], rtol=1e-10)
    assert np.isclose(res["se_beta"], se_matrix[1], rtol=1e-10)
    assert np.isclose(res["se_time"], se_matrix[2], rtol=1e-10)
    assert np.allclose(res["residuals"], resid_matrix, atol=1e-10)
    assert res["k"] == 2
    assert res["df"] == 117


def test_linear_regression_rejects_constant_x():
    y = np.linspace(10.0, 20.0, 50)
    x = np.full(50, 15.0)
    with pytest.raises(DegenerateSeriesError, match="constante"):
        linear_regression(y, x, with_time=False)


def test_linear_regression_rejects_constant_y():
    y = np.full(50, 15.0)
    x = np.linspace(10.0, 20.0, 50)
    with pytest.raises(DegenerateSeriesError, match="constante"):
        linear_regression(y, x, with_time=False)


def test_linear_regression_rejects_collinear_x_and_time():
    """Se X cresce de forma perfeitamente linear com t, a regressão múltipla é singular."""
    t = np.arange(1, 51, dtype=np.float64)
    x = 10.0 + 2.0 * t
    y = 5.0 + 1.5 * x + np.sin(t)
    with pytest.raises(DegenerateSeriesError, match="colinear|singular"):
        linear_regression(y, x, with_time=True)


def test_linear_regression_rejects_insufficient_and_nan_series():
    with pytest.raises(InsufficientDataError):
        linear_regression([1.0, 2.0], [3.0, 4.0], with_time=False)
    with pytest.raises(InsufficientDataError):
        linear_regression([1.0, 2.0, 3.0], [3.0, 4.0, 6.0], with_time=True)
    with pytest.raises(InvalidSeriesError, match="NaN"):
        linear_regression([1.0, np.nan, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
