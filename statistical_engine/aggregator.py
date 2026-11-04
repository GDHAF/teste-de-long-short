"""Módulo agregador da Statistical Engine: pair_stats(prices_a, prices_b, n_periods)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .exceptions import (
    DegenerateSeriesError,
    InsufficientDataError,
    InvalidSeriesError,
    StatisticalEngineError,
)
from .inference import (
    beta_confidence_interval,
    coefficient_t_test,
    fisher_test,
    pearson_correlation,
)
from .regression import linear_regression
from .stability import (
    beta_rotation,
    coefficient_of_variation,
    historical_volatility,
)
from .stationarity import dickey_fuller, half_life, z_score
from .validation import validate_paired_series

DEFAULT_VOLATILITY_WINDOWS: tuple[int, ...] = (10, 21, 42, 63)
DEFAULT_ROLLING_WINDOW: int = 60


def _normalize_windows(windows: int | Iterable[int]) -> list[int]:
    if isinstance(windows, (int, np.integer)) and not isinstance(windows, bool):
        return [int(windows)]
    if isinstance(windows, Iterable) and not isinstance(windows, (str, bytes)):
        win_list: list[int] = []
        for w in windows:
            if isinstance(w, bool) or not isinstance(w, (int, np.integer)) or int(w) < 2:
                raise InvalidSeriesError(
                    "As janelas de volatilidade devem ser inteiros >= 2."
                )
            win_list.append(int(w))
        if not win_list:
            raise InvalidSeriesError("A coleção de janelas de volatilidade não pode ser vazia.")
        return win_list
    raise InvalidSeriesError(
        "O parâmetro 'volatility_windows' deve ser um inteiro ou iterável de inteiros."
    )


def pair_stats(
    prices_a: Any,
    prices_b: Any,
    n_periods: int,
    with_time: bool = False,
    conf: float = 0.99,
    rolling_window: int | None = DEFAULT_ROLLING_WINDOW,
    volatility_windows: int | Iterable[int] = DEFAULT_VOLATILITY_WINDOWS,
    volatility_scale: str = "annualized",
    corr_on_log_returns: bool = False,
    use_manual_critical_values: bool = False,
) -> dict[str, Any]:
    """Calcula e agrega todas as métricas estatísticas de cointegração para o par (A, B).

    Convenção fixa do projeto (Manual e Roteiro do TCC):
        - prices_a (Ativo A): variável dependente (Y_t)
        - prices_b (Ativo B): variável independente (X_t)
        - Equação da regressão: Y_t = alpha + beta * X_t (+ T * t) + e_t
        - O cálculo utiliza os últimos `n_periods` pontos das séries alinhadas
          (ordenados cronologicamente do mais antigo para o mais recente).

    Args:
        prices_a: Série temporal de preços do Ativo A (variável dependente Y).
        prices_b: Série temporal de preços do Ativo B (variável independente X).
        n_periods: Tamanho da janela amostral N (ex: 70, 100, 120, ..., 250).
        with_time: Se True, ajusta regressão linear múltipla incluindo tendência temporal t=1..N.
        conf: Nível de confiança para IC(beta), Teste de Fisher e Testes t (padrão 0.99).
        rolling_window: Tamanho da janela móvel para Beta Rotation (padrão 60).
        volatility_windows: Janelas para cálculo de volatilidade histórica (padrão (10, 21, 42, 63)).
        volatility_scale: Escala da volatilidade histórica ('annualized', 'period' ou 'daily').
        corr_on_log_returns: Se True, calcula a correlação de Pearson sobre log-retornos ln(P_t/P_{t-1}).
        use_manual_critical_values: Se True, utiliza valores críticos fixos de 60gl / z=2.575 do manual.

    Returns:
        Dicionário estruturado contendo:
            - Configuração: 'n_periods', 'with_time', 'conf', 'rolling_window'
            - Regressão e resíduos: 'alpha', 'beta', 't_coef', 'fitted_values', 'residuals',
              'std_residual', 'r_squared', 'se_alpha', 'se_beta', 'se_time', 'df'
            - Intervalos de confiança: 'ic_beta_min', 'ic_beta_max', 'ic_alpha_min', 'ic_alpha_max',
              'ic_time_min', 'ic_time_max'
            - Testes t: 't_beta', 't_alpha', 't_time', 't_beta_level', 't_alpha_level', 't_time_level',
              't_beta_p_value', 't_alpha_p_value', 't_time_p_value'
            - Correlação e Fisher: 'corr', 'fisher_min', 'fisher_max'
            - Cointegração e estacionariedade: 'dickey_fuller', 'cointegration_level',
              'is_cointegrated', 'df_gamma', 'df_se_gamma', 'df_critical_values'
            - Z-Score e Meia-vida: 'z_score_current', 'z_scores', 'half_life'
            - Estabilidade do Beta: 'beta_rotation', 'beta_rotation_mean', 'beta_rotation_std', 'cv_beta'
            - Volatilidade histórica: 'volatility_a', 'volatility_b', 'volatility_beta_rotation',
              'historical_volatility'
            - Diagnóstico e validade: 'validity', 'is_valid', 'warnings'

    Raises:
        InvalidSeriesError: Se houver NaN, Inf, tamanhos/índices incompatíveis ou parâmetros inválidos.
        InsufficientDataError: Se as séries forem curtas demais para `n_periods` ou para a regressão base.
        DegenerateSeriesError: Se `prices_a` ou `prices_b` for constante, a regressão for singular/colinear,
            ou o desvio-padrão dos resíduos for igual a zero.
    """
    if isinstance(n_periods, bool) or not isinstance(n_periods, (int, np.integer)):
        raise InvalidSeriesError("O parâmetro 'n_periods' deve ser um inteiro.")
    if not isinstance(with_time, bool):
        raise InvalidSeriesError("O parâmetro 'with_time' deve ser um booleano.")

    n_win = int(n_periods)
    k = 2 if with_time else 1
    min_required = max(4, k + 2)
    if n_win < min_required:
        raise InsufficientDataError(
            f"O parâmetro 'n_periods' ({n_win}) é insuficiente; mínimo exigido é {min_required}."
        )

    vol_windows = _normalize_windows(volatility_windows)

    arr_a_full, arr_b_full = validate_paired_series(
        prices_a,
        prices_b,
        name_a="prices_a",
        name_b="prices_b",
        min_length=n_win,
        require_positive=corr_on_log_returns,
    )

    # Seleciona exatamente os últimos N períodos (mantendo a ordem do mais antigo ao mais recente)
    arr_a = arr_a_full[-n_win:]
    arr_b = arr_b_full[-n_win:]

    warnings_list: list[str] = []
    validity: dict[str, bool] = {
        "regression": False,
        "beta_ci": False,
        "t_tests": False,
        "correlation": False,
        "fisher_test": False,
        "dickey_fuller": False,
        "z_score": False,
        "half_life": False,
        "beta_rotation": False,
        "cv_beta": False,
        "historical_volatility": False,
    }

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
    df_reg = reg["df"]

    scale_y = max(1.0, float(np.std(arr_a, ddof=1)))
    if std_residual <= 1e-12 * scale_y or se_beta <= 1e-12 * scale_y:
        raise DegenerateSeriesError(
            "Desvio-padrão dos resíduos ou erro-padrão de beta é igual a zero (ajuste determinístico sem ruído); "
            "estatísticas de inferência, estacionariedade e z-score são indefinidas."
        )

    validity["regression"] = True

    # 2. Intervalos de Confiança dos Coeficientes e Testes t de Significância
    ic_beta_min, ic_beta_max = beta_confidence_interval(
        beta=beta,
        se=se_beta,
        n=n_win,
        k=k,
        conf=conf,
        use_manual_critical_values=use_manual_critical_values,
    )
    ic_alpha_min, ic_alpha_max = beta_confidence_interval(
        beta=alpha,
        se=se_alpha,
        n=n_win,
        k=k,
        conf=conf,
        use_manual_critical_values=use_manual_critical_values,
    )
    validity["beta_ci"] = True

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
        ic_time_min, ic_time_max = beta_confidence_interval(
            beta=t_coef,
            se=se_time,
            n=n_win,
            k=k,
            conf=conf,
            use_manual_critical_values=use_manual_critical_values,
        )
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
        t_time_p_val: float | None = t_time_res["p_value"]
    else:
        ic_time_min = None
        ic_time_max = None
        t_time_stat = None
        t_time_level = None
        t_time_p_val = None

    validity["t_tests"] = True

    # 3. Correlação de Pearson e Transformada de Fisher
    corr = pearson_correlation(arr_a, arr_b, use_log_returns=corr_on_log_returns)
    validity["correlation"] = True

    n_corr = n_win - 1 if corr_on_log_returns else n_win
    try:
        fisher_min_val, fisher_max_val = fisher_test(
            corr=corr,
            n=n_corr,
            conf=conf,
            use_manual_critical_values=use_manual_critical_values,
        )
        fisher_min: float | None = fisher_min_val
        fisher_max: float | None = fisher_max_val
        validity["fisher_test"] = True
    except StatisticalEngineError as exc:
        fisher_min = None
        fisher_max = None
        validity["fisher_test"] = False
        warnings_list.append(f"fisher_test: {exc}")

    # 4. Teste de Dickey-Fuller (Engle-Granger) sobre os Resíduos da Regressão
    df_res = dickey_fuller(residuals)
    validity["dickey_fuller"] = True

    # 5. Z-Score dos Resíduos
    z_series = z_score(residuals)
    z_score_current = float(z_series[-1])
    validity["z_score"] = True

    # 6. Meia-Vida (Ornstein-Uhlenbeck)
    try:
        hl_val: float | None = half_life(
            residuals, method="discrete", allow_non_stationary=False
        )
        validity["half_life"] = True
    except StatisticalEngineError as exc:
        hl_val = None
        validity["half_life"] = False
        warnings_list.append(f"half_life: {exc}")

    # 7. Beta Rotation e Coeficiente de Variação do Beta
    eff_rolling_window = (
        DEFAULT_ROLLING_WINDOW if rolling_window is None else int(rolling_window)
    )
    try:
        b_rot: np.ndarray | None = beta_rotation(
            arr_a,
            arr_b,
            window=eff_rolling_window,
            with_time=with_time,
        )
        validity["beta_rotation"] = True
        beta_rot_mean: float | None = float(np.mean(b_rot))
    except StatisticalEngineError as exc:
        b_rot = None
        beta_rot_mean = None
        validity["beta_rotation"] = False
        warnings_list.append(f"beta_rotation: {exc}")

    if b_rot is not None:
        try:
            cv_beta_val: float | None = coefficient_of_variation(b_rot, ddof=1)
            beta_rot_std: float | None = float(np.std(b_rot, ddof=1))
            validity["cv_beta"] = True
        except StatisticalEngineError as exc:
            cv_beta_val = None
            beta_rot_std = None
            validity["cv_beta"] = False
            warnings_list.append(f"cv_beta: {exc}")
    else:
        cv_beta_val = None
        beta_rot_std = None
        validity["cv_beta"] = False
        warnings_list.append(
            "cv_beta: indisponível porque a série beta_rotation não pôde ser calculada."
        )

    # 8. Volatilidade Histórica nas janelas solicitadas
    volatility_a: dict[int, float | None] = {}
    volatility_b: dict[int, float | None] = {}
    volatility_beta_rot: dict[int, float | None] = {}
    all_price_vols_ok = True

    for w in vol_windows:
        try:
            vol_a_w = historical_volatility(arr_a, window=w, scale=volatility_scale)
            volatility_a[w] = float(vol_a_w)  # type: ignore[arg-type]
        except StatisticalEngineError as exc:
            volatility_a[w] = None
            all_price_vols_ok = False
            warnings_list.append(f"volatility_a[{w}]: {exc}")

        try:
            vol_b_w = historical_volatility(arr_b, window=w, scale=volatility_scale)
            volatility_b[w] = float(vol_b_w)  # type: ignore[arg-type]
        except StatisticalEngineError as exc:
            volatility_b[w] = None
            all_price_vols_ok = False
            warnings_list.append(f"volatility_b[{w}]: {exc}")

        if b_rot is not None:
            try:
                vol_br_w = historical_volatility(
                    b_rot, window=w, scale=volatility_scale
                )
                volatility_beta_rot[w] = float(vol_br_w)  # type: ignore[arg-type]
            except StatisticalEngineError:
                volatility_beta_rot[w] = None
        else:
            volatility_beta_rot[w] = None

    validity["historical_volatility"] = all_price_vols_ok
    is_valid = bool(all(validity.values()))

    return {
        # Identificação da configuração
        "n_periods": n_win,
        "with_time": with_time,
        "conf": float(conf),
        "rolling_window": eff_rolling_window,
        # Coeficientes e ajuste da regressão
        "alpha": alpha,
        "beta": beta,
        "t_coef": t_coef,
        "se_alpha": se_alpha,
        "se_beta": se_beta,
        "se_time": se_time,
        "df": df_reg,
        "r_squared": reg["r_squared"],
        "fitted_values": fitted_values,
        "residuals": residuals,
        "std_residual": std_residual,
        # Intervalos de confiança dos coeficientes
        "ic_beta_min": ic_beta_min,
        "ic_beta_max": ic_beta_max,
        "ic_alpha_min": ic_alpha_min,
        "ic_alpha_max": ic_alpha_max,
        "ic_time_min": ic_time_min,
        "ic_time_max": ic_time_max,
        # Testes t dos coeficientes
        "t_beta": t_beta_res["t_stat"],
        "t_alpha": t_alpha_res["t_stat"],
        "t_time": t_time_stat,
        "t_beta_level": t_beta_res["confidence_level"],
        "t_alpha_level": t_alpha_res["confidence_level"],
        "t_time_level": t_time_level,
        "t_beta_p_value": t_beta_res["p_value"],
        "t_alpha_p_value": t_alpha_res["p_value"],
        "t_time_p_value": t_time_p_val,
        # Correlação de Pearson e Transformada de Fisher
        "corr": corr,
        "fisher_min": fisher_min,
        "fisher_max": fisher_max,
        # Teste de Dickey-Fuller (Engle-Granger)
        "dickey_fuller": df_res["tau"],
        "cointegration_level": df_res["cointegration_level"],
        "is_cointegrated": df_res["is_stationary"],
        "df_gamma": df_res["gamma"],
        "df_se_gamma": df_res["se_gamma"],
        "df_critical_values": df_res["critical_values"],
        # Z-Score e Meia-vida dos resíduos
        "z_score_current": z_score_current,
        "z_scores": z_series,
        "half_life": hl_val,
        # Estabilidade do Beta
        "beta_rotation": b_rot,
        "beta_rotation_mean": beta_rot_mean,
        "beta_rotation_std": beta_rot_std,
        "cv_beta": cv_beta_val,
        # Volatilidade Histórica
        "volatility_a": volatility_a,
        "volatility_b": volatility_b,
        "volatility_beta_rotation": volatility_beta_rot,
        "historical_volatility": {
            "prices_a": volatility_a,
            "prices_b": volatility_b,
            "beta_rotation": volatility_beta_rot,
        },
        # Diagnóstico de validade e avisos
        "validity": validity,
        "is_valid": is_valid,
        "warnings": warnings_list,
    }
