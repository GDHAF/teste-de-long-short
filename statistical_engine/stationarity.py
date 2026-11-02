"""Módulo de Estacionariedade e Reversão à Média: Dickey-Fuller (Engle-Granger), Z-Score e Meia-Vida (Ornstein-Uhlenbeck)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .exceptions import (
    DegenerateSeriesError,
    InsufficientDataError,
    InvalidSeriesError,
)
from .validation import validate_1d_series, validate_scalar_float

# Valores críticos do teste t de cointegração de Engle-Granger especificados no Manual (p. 41-42)
ENGLE_GRANGER_CRITICAL_VALUES: dict[int, float] = {
    99: -4.32,
    95: -3.67,
    90: -3.28,
}


def classify_cointegration_level(tau: float) -> int:
    """Classifica o grau de cointegração (99, 95, 90 ou 0) a partir da estatística Tau (p. 41-42).

    Regras do Manual:
        - tau <= -4.32 -> 99% de confiança
        - -4.32 < tau <= -3.67 -> 95% de confiança
        - -3.67 < tau <= -3.28 -> 90% de confiança
        - tau > -3.28 -> 0% (não estacionário / não cointegrado)
    """
    tau_val = validate_scalar_float(tau, "tau")
    if tau_val <= ENGLE_GRANGER_CRITICAL_VALUES[99]:
        return 99
    if tau_val <= ENGLE_GRANGER_CRITICAL_VALUES[95]:
        return 95
    if tau_val <= ENGLE_GRANGER_CRITICAL_VALUES[90]:
        return 90
    return 0


def _fit_ar1_residual_differences(residuals: Any) -> dict[str, Any]:
    """Ajusta a regressão auxiliar AR(1): Delta(e_t) = alpha_df + gamma * e_{t-1} + u_t."""
    arr_e = validate_1d_series(residuals, name="residuals", min_length=4)

    e_lag = arr_e[:-1]
    delta_e = arr_e[1:] - arr_e[:-1]
    n_df = int(e_lag.size)  # N - 1 observações na regressão auxiliar
    df_resid = n_df - 2

    if df_resid <= 0:
        raise InsufficientDataError(
            f"Série de resíduos insuficiente para regressão AR(1) (n_df={n_df}, df={df_resid})."
        )

    e_lag_centered = e_lag - np.mean(e_lag)
    sxx = float(np.sum(e_lag_centered**2))
    if sxx <= 1e-30:
        raise DegenerateSeriesError(
            "A série de resíduos é constante (variância zero em e_{t-1}); regressão AR(1) indeterminada."
        )

    delta_centered = delta_e - np.mean(delta_e)
    sxy = float(np.sum(e_lag_centered * delta_centered))

    gamma = sxy / sxx
    alpha_df = float(np.mean(delta_e) - gamma * np.mean(e_lag))
    u_t = delta_e - (alpha_df + gamma * e_lag)

    ssr = float(np.sum(u_t**2))
    se_gamma = float(np.sqrt(ssr / df_resid) / np.sqrt(sxx))

    return {
        "gamma": float(gamma),
        "alpha_df": alpha_df,
        "se_gamma": se_gamma,
        "u_t": u_t,
        "n_df": n_df,
        "df": df_resid,
    }


def dickey_fuller(residuals: Any) -> dict[str, Any]:
    """Executa o Teste de Dickey-Fuller sobre os resíduos pela metodologia de Engle-Granger (p. 39-42).

    Regressão auxiliar:
        Delta(e_t) = e_t - e_{t-1} = alpha + gamma * e_{t-1} + u_t

    Estatística de teste:
        Tau (tau) = gamma / EP(gamma)

    Args:
        residuals: Série temporal de resíduos (e_t) da regressão de cointegração.

    Returns:
        Dicionário contendo:
            - 'tau': float (estatística Dickey-Fuller Tau)
            - 'dickey_fuller': float (alias para 'tau')
            - 'gamma': float (coeficiente angular de e_{t-1})
            - 'se_gamma': float (erro padrão de gamma)
            - 'alpha_df': float (intercepto da regressão auxiliar)
            - 'cointegration_level': int (99, 95, 90 ou 0)
            - 'is_stationary': bool (True se cointegration_level >= 90)
            - 'critical_values': dict[int, float] ({99: -4.32, 95: -3.67, 90: -3.28})

    Raises:
        InvalidSeriesError: Se houver NaN, Inf ou dimensões inválidas.
        InsufficientDataError: Se houver menos de 4 observações nos resíduos.
        DegenerateSeriesError: Se o resíduo for constante ou se EP(gamma) == 0.
    """
    ar1 = _fit_ar1_residual_differences(residuals)
    gamma = ar1["gamma"]
    se_gamma = ar1["se_gamma"]

    if se_gamma <= 1e-15:
        raise DegenerateSeriesError(
            "Erro padrão EP(gamma) nulo na regressão de Dickey-Fuller (ajuste determinístico sem ruído)."
        )

    tau = float(gamma / se_gamma)
    level = classify_cointegration_level(tau)

    return {
        "tau": tau,
        "dickey_fuller": tau,
        "gamma": gamma,
        "se_gamma": se_gamma,
        "alpha_df": ar1["alpha_df"],
        "cointegration_level": level,
        "is_stationary": bool(level >= 90),
        "critical_values": dict(ENGLE_GRANGER_CRITICAL_VALUES),
    }


def z_score(residuals: Any, ddof: int = 1) -> np.ndarray:
    """Calcula a série padronizada Z-Score dos resíduos (p. 43-44).

    Fórmula:
        Z_t = (e_t - media(e)) / desvio_padrao(e)

    Args:
        residuals: Série temporal de resíduos (e_t).
        ddof: Graus de liberdade do desvio padrão (padrão 1, amostral).

    Returns:
        np.ndarray 1D contendo o Z-Score para cada instante t.

    Raises:
        InvalidSeriesError: Se houver NaN/Inf ou tipo inválido.
        InsufficientDataError: Se houver menos de 2 observações.
        DegenerateSeriesError: Se o desvio padrão dos resíduos for zero.
    """
    arr_e = validate_1d_series(residuals, name="residuals", min_length=2)
    if arr_e.size <= ddof:
        raise InsufficientDataError(
            f"Número de observações ({arr_e.size}) insuficiente para ddof={ddof}."
        )

    mean_e = float(np.mean(arr_e))
    std_e = float(np.std(arr_e, ddof=ddof))

    if std_e <= 1e-15:
        raise DegenerateSeriesError(
            "Desvio padrão dos resíduos é zero; Z-Score indefinido (divisão por zero)."
        )

    return (arr_e - mean_e) / std_e


def half_life(
    residuals: Any,
    method: str = "discrete",
    allow_non_stationary: bool = False,
) -> float:
    """Calcula a Meia-Vida (Half-Life) da reversão à média pelo modelo de Ornstein-Uhlenbeck (p. 47).

    Fórmula:
        Na regressão Delta(e_t) = alpha + gamma * e_{t-1} + u_t:
        - method='discrete' (padrão): velocidade de reversão theta = -gamma => HL = ln(2) / (-gamma)
        - method='exact_ou': theta = -ln(1 + gamma) => HL = -ln(2) / ln(1 + gamma)

    Args:
        residuals: Série temporal de resíduos (e_t).
        method: 'discrete' (padrão do manual/roteiro) ou 'exact_ou'.
        allow_non_stationary: Se False (padrão), levanta DegenerateSeriesError quando gamma >= 0
            (ausência de reversão à média). Se True, retorna float('inf').

    Returns:
        Meia-vida (float > 0) em número de períodos (dias úteis).
    """
    if method not in ("discrete", "exact_ou"):
        raise InvalidSeriesError(
            f"Método 'method' inválido: '{method}'. Use 'discrete' ou 'exact_ou'."
        )

    ar1 = _fit_ar1_residual_differences(residuals)
    gamma = ar1["gamma"]

    if gamma >= 0.0:
        if allow_non_stationary:
            return float("inf")
        raise DegenerateSeriesError(
            f"A série de resíduos não apresenta reversão à média (gamma={gamma:.6f} >= 0); meia-vida indefinida."
        )

    if method == "discrete":
        theta = -gamma
        return float(np.log(2.0) / theta)

    # method == 'exact_ou'
    if gamma <= -1.0:
        if allow_non_stationary:
            return float("inf")
        raise DegenerateSeriesError(
            f"Coeficiente AR(1) gamma={gamma:.6f} <= -1.0 invalida log(1 + gamma) no modelo contínuo."
        )
    theta = -float(np.log(1.0 + gamma))
    return float(np.log(2.0) / theta)
