"""Módulo de Estabilidade e Volatilidade: Beta Rotation, Coeficiente de Variação e Volatilidade Histórica."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .exceptions import (
    DegenerateSeriesError,
    InsufficientDataError,
    InvalidSeriesError,
)
from .regression import linear_regression
from .validation import validate_1d_series, validate_paired_series


def beta_rotation(
    prices_a: Any,
    prices_b: Any,
    window: int,
    with_time: bool = False,
) -> np.ndarray:
    """Calcula a série de Beta Móvel (Beta Rotation / Rolling Beta) em janela deslizante (p. 48-50).

    Para séries de comprimento N e janela móvel W (com N >= W), calcula o coeficiente
    angular beta para cada sub-janela contígua [i : i + W], do período mais antigo
    ao mais recente, retornando N - W + 1 valores.

    Args:
        prices_a: Série temporal de preços do Ativo A (variável dependente Y).
        prices_b: Série temporal de preços do Ativo B (variável independente X).
        window: Tamanho da janela móvel (ex: 60, 70, 120).
        with_time: Se True, utiliza regressão múltipla com tendência temporal em cada janela.

    Returns:
        np.ndarray 1D contendo os valores de beta móvel ao longo do tempo.

    Raises:
        InvalidSeriesError: Se window não for inteiro válido ou se as séries contiverem NaN/Inf.
        InsufficientDataError: Se window <= k + 1 ou se o tamanho das séries for menor que window.
        DegenerateSeriesError: Se alguma janela apresentar variância zero em X ou colinearidade.
    """
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise InvalidSeriesError("O parâmetro 'window' deve ser um inteiro.")

    w = int(window)
    k = 2 if with_time else 1
    if w <= k + 1:
        raise InsufficientDataError(
            f"O tamanho da janela 'window' ({w}) é insuficiente para regressão com k={k} (mínimo {k + 2})."
        )

    arr_a, arr_b = validate_paired_series(
        prices_a,
        prices_b,
        name_a="prices_a",
        name_b="prices_b",
        min_length=w,
    )

    n_total = int(arr_a.size)
    n_windows = n_total - w + 1
    betas = np.empty(n_windows, dtype=np.float64)

    for i in range(n_windows):
        sub_y = arr_a[i : i + w]
        sub_x = arr_b[i : i + w]
        reg = linear_regression(sub_y, sub_x, with_time=with_time)
        betas[i] = reg["beta"]

    return betas


def coefficient_of_variation(
    series: Any,
    ddof: int = 1,
    as_percentage: bool = False,
) -> float:
    """Calcula o Coeficiente de Variação (CV = sigma / |mu|) de uma série (p. 65).

    Args:
        series: Série numérica 1D (ex: série de betas entre janelas ou beta_rotation).
        ddof: Graus de liberdade do desvio padrão (padrão 1 amostral; 0 reproduz DESVPAD.P da p. 65).
        as_percentage: Se True, multiplica o resultado por 100.0 (em %).

    Returns:
        Valor escalar float do Coeficiente de Variação.

    Raises:
        InvalidSeriesError: Se houver NaN/Inf ou tipo inválido.
        InsufficientDataError: Se houver menos de max(2, ddof + 1) observações.
        DegenerateSeriesError: Se a média da série for zero (divisão por zero).
    """
    if isinstance(ddof, bool) or not isinstance(ddof, (int, np.integer)) or ddof < 0:
        raise InvalidSeriesError("O parâmetro 'ddof' deve ser um inteiro >= 0.")

    min_len = max(2, int(ddof) + 1)
    arr = validate_1d_series(series, name="series", min_length=min_len)

    mean_val = float(np.mean(arr))
    if abs(mean_val) <= 1e-15:
        raise DegenerateSeriesError(
            "Coeficiente de Variação indefinido: a média da série é zero (divisão por zero)."
        )

    std_val = float(np.std(arr, ddof=int(ddof)))
    cv = std_val / abs(mean_val)
    return float(cv * 100.0) if as_percentage else float(cv)


def historical_volatility(
    series: Any,
    window: int | Iterable[int] = (10, 21, 42, 63),
    scale: str = "annualized",
    ddof: int = 1,
) -> float | dict[int, float]:
    """Calcula a Volatilidade Histórica (VH) sobre os log-retornos r_t = ln(P_t / P_{t-1}) (p. 51-53).

    Fórmulas:
        r_t = ln(series_t / series_{t-1})
        VH_diaria(w) = DesvioPadrao(ultimos w retornos, ddof=1)
        - scale='daily': VH_diaria(w)
        - scale='period': VH_diaria(w) * sqrt(w)  (conforme p. 52 do Manual)
        - scale='annualized': VH_diaria(w) * sqrt(252)  (conforme p. 13 do Roteiro e p. 52-53 do Manual)

    Args:
        series: Série temporal de valores estritamente positivos (> 0).
        window: Inteiro único (retorna float) ou iterável de inteiros (retorna dict[int, float]).
        scale: 'annualized' (padrão), 'period' ou 'daily'.
        ddof: Graus de liberdade do desvio padrão (padrão 1).

    Returns:
        float (se window for int) ou dict[int, float] mapeando cada janela para sua VH.

    Raises:
        InvalidSeriesError: Se a série contiver valores <= 0, NaN/Inf ou se scale/window forem inválidos.
        InsufficientDataError: Se a série não possuir pelo menos max(window) + 1 observações.
    """
    if scale not in ("annualized", "period", "daily"):
        raise InvalidSeriesError(
            f"Escala 'scale' inválida: '{scale}'. Use 'annualized', 'period' ou 'daily'."
        )

    single_window = isinstance(window, (int, np.integer)) and not isinstance(
        window, bool
    )
    if single_window:
        windows_list = [int(window)]
    elif isinstance(window, Iterable) and not isinstance(window, (str, bytes)):
        windows_list = []
        for w in window:
            if isinstance(w, bool) or not isinstance(w, (int, np.integer)):
                raise InvalidSeriesError(
                    "Todos os elementos de 'window' devem ser inteiros."
                )
            windows_list.append(int(w))
        if not windows_list:
            raise InvalidSeriesError("A lista de janelas 'window' não pode ser vazia.")
    else:
        raise InvalidSeriesError(
            "O parâmetro 'window' deve ser um inteiro ou iterável de inteiros."
        )

    for w in windows_list:
        if w <= ddof or w < 2:
            raise InsufficientDataError(
                f"A janela de volatilidade ({w}) deve ser maior que ddof ({ddof}) e >= 2."
            )

    max_w = max(windows_list)
    arr = validate_1d_series(
        series,
        name="series",
        min_length=max_w + 1,
        require_positive=True,
    )

    log_returns = np.log(arr[1:] / arr[:-1])

    results: dict[int, float] = {}
    for w in windows_list:
        r_slice = log_returns[-w:]
        daily_vol = float(np.std(r_slice, ddof=ddof))
        if scale == "daily":
            vol = daily_vol
        elif scale == "period":
            vol = float(daily_vol * np.sqrt(float(w)))
        else:  # annualized
            vol = float(daily_vol * np.sqrt(252.0))
        results[w] = vol

    if single_window:
        return results[windows_list[0]]
    return results
