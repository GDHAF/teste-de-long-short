"""Módulo de Regressão Linear Simples e Múltipla (OLS) e cálculo de resíduos."""

from __future__ import annotations

from typing import Any

import numpy as np
import statsmodels.api as sm

from .exceptions import DegenerateSeriesError, InsufficientDataError
from .validation import validate_paired_series


def compute_fitted_values(
    x: Any,
    alpha: float,
    beta: float,
    t_coef: float | None = None,
) -> np.ndarray:
    """Calcula os valores estimados (Y_hat) pela equação da regressão.

    Equação:
        Simples: Y_hat_t = alpha + beta * X_t
        Múltipla: Y_hat_t = alpha + beta * X_t + t_coef * t, onde t = 1, ..., N
    """
    arr_x = np.asarray(x, dtype=np.float64)
    fitted = float(alpha) + float(beta) * arr_x
    if t_coef is not None:
        time_index = np.arange(1, arr_x.size + 1, dtype=np.float64)
        fitted = fitted + float(t_coef) * time_index
    return fitted


def compute_residuals(
    y: Any,
    x: Any,
    alpha: float,
    beta: float,
    t_coef: float | None = None,
) -> np.ndarray:
    """Calcula os resíduos estatísticos da regressão: e_t = Y_t - Y_hat_t."""
    arr_y, arr_x = validate_paired_series(y, x, name_a="y", name_b="x", min_length=1)
    fitted = compute_fitted_values(arr_x, alpha=alpha, beta=beta, t_coef=t_coef)
    return arr_y - fitted


def linear_regression(
    y: Any,
    x: Any,
    with_time: bool = False,
) -> dict[str, Any]:
    """Executa Regressão Linear Mínimos Quadrados Ordinários (OLS) simples ou múltipla.

    Convenção do TCC:
        - y (Ativo A): variável dependente (Y_t)
        - x (Ativo B): variável independente (X_t)
        - Quando with_time=False (k=1):
            Y_t = alpha + beta * X_t + e_t
        - Quando with_time=True (k=2):
            Y_t = alpha + beta * X_t + T * t + e_t, com t = 1, 2, ..., N

    Args:
        y: Série temporal de preços do Ativo A (variável dependente).
        x: Série temporal de preços do Ativo B (variável independente).
        with_time: Se True, inclui variável de tendência temporal linear (1..N).

    Returns:
        Dicionário contendo:
            - 'alpha': float (coeficiente linear / intercepto)
            - 'beta': float (coeficiente angular da variável independente x)
            - 't_coef': float | None (coeficiente angular do tempo T se with_time=True, senão None)
            - 'fitted_values': np.ndarray (valores estimados Y_hat_t)
            - 'residuals': np.ndarray (resíduos e_t = Y_t - Y_hat_t)
            - 'se_alpha': float (erro padrão do intercepto alpha)
            - 'se_beta': float (erro padrão do coeficiente angular beta)
            - 'se_time': float | None (erro padrão do coeficiente T se with_time=True, senão None)
            - 'std_residual': float (desvio padrão amostral dos resíduos, ddof=1)
            - 'r_squared': float (coeficiente de determinação R²)
            - 'n_obs': int (número de observações N)
            - 'k': int (número de variáveis explicativas: 1 ou 2)
            - 'df': int (graus de liberdade do resíduo: N - k - 1)

    Raises:
        InvalidSeriesError: Se houver NaN, Inf, dimensões inválidas ou tamanhos diferentes.
        InsufficientDataError: Se N <= k + 1 (graus de liberdade <= 0).
        DegenerateSeriesError: Se X tiver variância zero ou se a matriz de desenho for singular/colinear.
    """
    k = 2 if with_time else 1
    min_obs = k + 2  # Garante df = N - k - 1 >= 1

    arr_y, arr_x = validate_paired_series(
        y, x, name_a="y", name_b="x", min_length=min_obs
    )
    n_obs = int(arr_y.size)
    df = n_obs - k - 1
    if df <= 0:
        raise InsufficientDataError(
            f"Observações insuficientes ({n_obs}) para regressão com k={k} variáveis explicativas."
        )

    # Verifica variância da variável independente X
    if np.allclose(arr_x, arr_x[0], rtol=0.0, atol=1e-15):
        raise DegenerateSeriesError(
            "A série independente 'x' é constante (variância zero); beta é indeterminado."
        )

    # Monta a matriz de regressores [1, X] ou [1, X, t]
    ones = np.ones(n_obs, dtype=np.float64)
    if with_time:
        time_index = np.arange(1, n_obs + 1, dtype=np.float64)
        design_matrix = np.column_stack((ones, arr_x, time_index))
    else:
        design_matrix = np.column_stack((ones, arr_x))

    # Verifica posto completo (ausência de colinearidade perfeita, ex: X perfeitamente linear com t)
    rank = np.linalg.matrix_rank(design_matrix)
    if rank < design_matrix.shape[1]:
        raise DegenerateSeriesError(
            "A matriz de regressores é singular (colinearidade perfeita entre as variáveis explicativas)."
        )

    # Verifica também condicionamento numérico para colinearidade quase perfeita
    if with_time:
        corr_xt = np.corrcoef(arr_x, time_index)[0, 1]
        if np.isnan(corr_xt) or abs(corr_xt) >= 1.0 - 1e-12:
            raise DegenerateSeriesError(
                "A série independente 'x' é perfeitamente colinear com a tendência temporal 't'."
            )

    model = sm.OLS(arr_y, design_matrix)
    results = model.fit()

    params = np.asarray(results.params, dtype=np.float64)
    bse = np.asarray(results.bse, dtype=np.float64)
    fitted_values = np.asarray(results.fittedvalues, dtype=np.float64)
    residuals = arr_y - fitted_values

    alpha = float(params[0])
    beta = float(params[1])
    t_coef = float(params[2]) if with_time else None

    se_alpha = float(bse[0])
    se_beta = float(bse[1])
    se_time = float(bse[2]) if with_time else None

    std_residual = float(np.std(residuals, ddof=1))

    # R²: se Y for constante, TSS=0 -> r_squared = 0.0
    tss = float(np.sum((arr_y - np.mean(arr_y)) ** 2))
    if tss <= 1e-30:
        r_squared = 0.0
    else:
        r_squared = float(results.rsquared)

    return {
        "alpha": alpha,
        "beta": beta,
        "t_coef": t_coef,
        "fitted_values": fitted_values,
        "residuals": residuals,
        "se_alpha": se_alpha,
        "se_beta": se_beta,
        "se_time": se_time,
        "std_residual": std_residual,
        "r_squared": r_squared,
        "n_obs": n_obs,
        "k": k,
        "df": df,
    }
