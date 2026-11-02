"""Módulo agregador da Statistical Engine: pair_stats(prices_a, prices_b, n_periods)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .exceptions import InsufficientDataError, InvalidSeriesError
from .inference import (
    beta_confidence_interval,
    coefficient_t_test,
    fisher_test,
    pearson_correlation,
)
from .regression import linear_regression
from .stability import beta_rotation, coefficient_of_variation
from .stationarity import dickey_fuller, half_life, z_score
from .validation import validate_paired_series


def pair_stats(
    prices_a: Any,
    prices_b: Any,
    n_periods: int,
    with_time: bool = False,
    conf: float = 0.99,
    rolling_window: int | None = None,
    corr_on_log_returns: bool = False,
    use_manual_critical_values: bool = False,
) -> dict[str, Any]:
    """Calcula e agrega todas as métricas estatísticas de cointegração para o par (A, B).

    Convenção (Manual e Roteiro do TCC):
        - prices_a (Ativo A): variável dependente (Y_t)
        - prices_b (Ativo B): variável independente (X_t)
        - O cálculo utiliza os últimos `n_periods` pontos das séries fornecidas
          (ordenados do mais antigo para o mais recente).

    Args:
        prices_a: Série temporal de preços do Ativo A (dependente).
        prices_b: Série temporal de preços do Ativo B (independente).
        n_periods: Tamanho da janela amostral N (ex: 70, 100, 120, ..., 250).
        with_time: Se True, ajusta regressão linear múltipla incluindo tendência temporal t=1..N.
        conf: Nível de confiança para IC(beta), Teste de Fisher e Testes t (padrão 0.99).
        rolling_window: Tamanho da janela móvel para Beta Rotation (padrão: 60 quando N > 60,
            ou max(k + 2, N // 2) para janelas menores em testes sintéticos).
        corr_on_log_returns: Se True, calcula a correlação de Pearson sobre log-retornos ln(P_t/P_{t-1}).
        use_manual_critical_values: Se True, utiliza valores críticos fixos de 60gl / z=2.575 do manual.

    Returns:
        Dicionário contendo todas as métricas exigidas pelo contrato da Parte 2:
            - 'alpha': float
            - 'beta': float
            - 't_coef': float | None
            - 'se_alpha': float
            - 'se_beta': float
            - 'se_time': float | None
            - 'r_squared': float
            - 'fitted_values': np.ndarray
            - 'residuals': np.ndarray
            - 'std_residual': float
            - 'dickey_fuller': float (estatística Tau)
            - 'cointegration_level': int (99, 95, 90 ou 0)
            - 'z_score_current': float (último valor do Z-Score dos resíduos)
            - 'z_scores': np.ndarray (série completa de Z-Score dos resíduos)
            - 'half_life': float (meia-vida de Ornstein-Uhlenbeck em períodos; inf se gamma >= 0)
            - 'corr': float (correlação de Pearson rho)
            - 'fisher_min': float
            - 'fisher_max': float
            - 'ic_beta_min': float
            - 'ic_beta_max': float
            - 't_beta': float (estatística t calculada para beta)
            - 't_alpha': float (estatística t calculada para alpha)
            - 't_time': float | None (estatística t calculada para T, ou None se with_time=False)
            - 't_beta_level': int (nível de confiança atingido no teste t de beta: 99, 95 ou 0)
            - 't_alpha_level': int (nível de confiança atingido no teste t de alpha: 99, 95 ou 0)
            - 't_time_level': int | None (nível de confiança atingido no teste t de T: 99, 95, 0 ou None)
            - 'beta_rotation': np.ndarray (série de beta móvel)
            - 'cv_beta': float (coeficiente de variação de beta_rotation)
            - 'n_periods': int
    """
    if isinstance(n_periods, bool) or not isinstance(n_periods, (int, np.integer)):
        raise InvalidSeriesError("O parâmetro 'n_periods' deve ser um inteiro.")

    n_win = int(n_periods)
    k = 2 if with_time else 1
    min_required = max(4, k + 2)
    if n_win < min_required:
        raise InsufficientDataError(
            f"O parâmetro 'n_periods' ({n_win}) é insuficiente; mínimo exigido é {min_required}."
        )

    arr_a_full, arr_b_full = validate_paired_series(
        prices_a,
        prices_b,
        name_a="prices_a",
        name_b="prices_b",
        min_length=n_win,
        require_positive=corr_on_log_returns,
    )

    # Seleciona os últimos N períodos (do mais antigo ao mais recente dentro da janela N)
    arr_a = arr_a_full[-n_win:]
    arr_b = arr_b_full[-n_win:]

    # 1. Regressão Linear OLS (Simples ou Múltipla) e Resíduos
    reg = linear_regression(arr_a, arr_b, with_time=with_time)
    alpha = reg["alpha"]
    beta = reg["beta"]
    t_coef = reg["t_coef"]
    se_alpha = reg["se_alpha"]
    se_beta = reg["se_beta"]
    se_time = reg["se_time"]
    residuals = reg["residuals"]
    fitted_values = reg["fitted_values"]
    std_residual = reg["std_residual"]

    # 2. Intervalo de Confiança para Beta e Testes t de Significância
    ic_beta_min, ic_beta_max = beta_confidence_interval(
        beta=beta,
        se=se_beta,
        n=n_win,
        k=k,
        conf=conf,
        use_manual_critical_values=use_manual_critical_values,
    )

    t_beta_res = coefficient_t_test(
        coef=beta,
        se=se_beta,
        n=n_win,
        k=k,
        tail="right",
        conf=conf,
        use_manual_critical_values=use_manual_critical_values,
    )
    t_alpha_res = coefficient_t_test(
        coef=alpha,
        se=se_alpha,
        n=n_win,
        k=k,
        tail="two-sided",
        conf=conf,
        use_manual_critical_values=use_manual_critical_values,
    )

    if with_time and t_coef is not None and se_time is not None:
        t_time_res = coefficient_t_test(
            coef=t_coef,
            se=se_time,
            n=n_win,
            k=k,
            tail="right",
            conf=conf,
            use_manual_critical_values=use_manual_critical_values,
        )
        t_time_stat: float | None = t_time_res["t_stat"]
        t_time_level: int | None = t_time_res["confidence_level"]
    else:
        t_time_stat = None
        t_time_level = None

    # 3. Correlação de Pearson e Transformada de Fisher
    corr = pearson_correlation(arr_a, arr_b, use_log_returns=corr_on_log_returns)
    fisher_min, fisher_max = fisher_test(
        corr=corr,
        n=n_win,
        conf=conf,
        use_manual_critical_values=use_manual_critical_values,
    )

    # 4. Teste de Dickey-Fuller (Engle-Granger), Z-Score e Meia-Vida (Ornstein-Uhlenbeck)
    df_res = dickey_fuller(residuals)
    z_series = z_score(residuals)
    z_score_current = float(z_series[-1])
    hl = half_life(residuals, method="discrete", allow_non_stationary=True)

    # 5. Beta Rotation e Coeficiente de Variação do Beta
    if rolling_window is not None:
        eff_rolling_window = int(rolling_window)
    else:
        eff_rolling_window = 60 if n_win > 60 else max(k + 2, n_win // 2)

    b_rot = beta_rotation(
        arr_a,
        arr_b,
        window=eff_rolling_window,
        with_time=with_time,
    )
    if b_rot.size >= 2:
        cv_beta = coefficient_of_variation(b_rot, ddof=1)
    else:
        cv_beta = 0.0

    return {
        "alpha": alpha,
        "beta": beta,
        "t_coef": t_coef,
        "se_alpha": se_alpha,
        "se_beta": se_beta,
        "se_time": se_time,
        "r_squared": reg["r_squared"],
        "fitted_values": fitted_values,
        "residuals": residuals,
        "std_residual": std_residual,
        "dickey_fuller": df_res["tau"],
        "cointegration_level": df_res["cointegration_level"],
        "z_score_current": z_score_current,
        "z_scores": z_series,
        "half_life": hl,
        "corr": corr,
        "fisher_min": fisher_min,
        "fisher_max": fisher_max,
        "ic_beta_min": ic_beta_min,
        "ic_beta_max": ic_beta_max,
        "t_beta": t_beta_res["t_stat"],
        "t_alpha": t_alpha_res["t_stat"],
        "t_time": t_time_stat,
        "t_beta_level": t_beta_res["confidence_level"],
        "t_alpha_level": t_alpha_res["confidence_level"],
        "t_time_level": t_time_level,
        "beta_rotation": b_rot,
        "cv_beta": cv_beta,
        "n_periods": n_win,
    }
