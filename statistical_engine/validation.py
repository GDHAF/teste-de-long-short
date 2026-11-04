"""Funções de validação de entrada e contratos de dados para a Statistical Engine."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .exceptions import (
    DegenerateSeriesError,
    InsufficientDataError,
    InvalidSeriesError,
)


def validate_1d_series(
    series: Any,
    name: str = "series",
    min_length: int = 1,
    require_positive: bool = False,
) -> np.ndarray:
    """Converte e valida uma série unidimensional numérica sem NaN ou Inf.

    Args:
        series: Objeto iterável (list, tuple, np.ndarray, pd.Series).
        name: Nome identificador da série para mensagens de erro.
        min_length: Número mínimo de observações exigido.
        require_positive: Se True, exige que todos os valores sejam > 0.

    Returns:
        np.ndarray 1D do tipo float64.

    Raises:
        InvalidSeriesError: Se a série for None, escalar, multidimensional,
            não-numérica, contiver NaN/Inf ou violar require_positive.
        InsufficientDataError: Se o tamanho da série for menor que min_length.
    """
    if series is None:
        raise InvalidSeriesError(f"A série '{name}' não pode ser None.")

    if isinstance(series, pd.DataFrame):
        raise InvalidSeriesError(
            f"A série '{name}' deve ser unidimensional (1D), mas recebeu DataFrame."
        )

    if isinstance(series, pd.Series):
        if series.index.isna().any():
            raise InvalidSeriesError(
                f"O índice da série '{name}' contém valores ausentes (NaN/NaT)."
            )
        if series.index.has_duplicates:
            raise InvalidSeriesError(
                f"O índice da série '{name}' contém rótulos duplicados."
            )

    try:
        arr = np.asarray(series, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise InvalidSeriesError(
            f"A série '{name}' contém valores não numéricos."
        ) from exc

    if arr.ndim != 1:
        raise InvalidSeriesError(
            f"A série '{name}' deve ser unidimensional (1D), mas possui {arr.ndim} dimensões."
        )

    if arr.size == 0:
        raise InsufficientDataError(f"A série '{name}' está vazia.")

    if np.isnan(arr).any():
        raise InvalidSeriesError(
            f"A série '{name}' contém valores ausentes (NaN)."
        )

    if np.isinf(arr).any():
        raise InvalidSeriesError(
            f"A série '{name}' contém valores infinitos (Inf)."
        )

    if arr.size < min_length:
        raise InsufficientDataError(
            f"A série '{name}' possui {arr.size} observações, mas o mínimo exigido é {min_length}."
        )

    if require_positive and np.any(arr <= 0.0):
        raise InvalidSeriesError(
            f"A série '{name}' exige valores estritamente positivos (> 0)."
        )

    return arr


def validate_paired_series(
    series_a: Any,
    series_b: Any,
    name_a: str = "prices_a",
    name_b: str = "prices_b",
    min_length: int = 2,
    require_positive: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Valida um par de séries unidimensionais alinhadas e de mesmo comprimento.

    Args:
        series_a: Série da variável dependente (Ativo A / Y).
        series_b: Série da variável independente (Ativo B / X).
        name_a: Nome identificador da primeira série.
        name_b: Nome identificador da segunda série.
        min_length: Número mínimo de observações exigido.
        require_positive: Se True, exige que ambas as séries tenham valores > 0.

    Returns:
        Tupla (arr_a, arr_b) como np.ndarray 1D float64 de mesmo tamanho.

    Raises:
        InvalidSeriesError: Se houver tamanhos diferentes, NaN/Inf ou tipo inválido.
        InsufficientDataError: Se o número de observações for menor que min_length.
    """
    arr_a = validate_1d_series(
        series_a, name=name_a, min_length=1, require_positive=require_positive
    )
    arr_b = validate_1d_series(
        series_b, name=name_b, min_length=1, require_positive=require_positive
    )

    if arr_a.size != arr_b.size:
        raise InvalidSeriesError(
            f"As séries '{name_a}' ({arr_a.size}) e '{name_b}' ({arr_b.size}) devem ter o mesmo comprimento."
        )

    if isinstance(series_a, pd.Series) and isinstance(series_b, pd.Series):
        if not series_a.index.equals(series_b.index):
            raise InvalidSeriesError(
                f"Os índices das séries '{name_a}' e '{name_b}' são incompatíveis ou desalinhados."
            )
    elif isinstance(series_a, pd.Series) or isinstance(series_b, pd.Series):
        ser = series_a if isinstance(series_a, pd.Series) else series_b
        ser_name = name_a if isinstance(series_a, pd.Series) else name_b
        default_index = pd.RangeIndex(start=0, stop= len(ser), step=1)
        if not ser.index.equals(default_index):
            raise InvalidSeriesError(
                f"A série '{ser_name}' possui índice customizado incompatível com array sem índice explícito."
            )

    if arr_a.size < min_length:
        raise InsufficientDataError(
            f"As séries possuem {arr_a.size} observações, mas o mínimo exigido é {min_length}."
        )

    return arr_a, arr_b


def validate_scalar_float(
    value: Any,
    name: str,
    allow_zero: bool = True,
    require_positive: bool = False,
) -> float:
    """Valida um valor escalar numérico finito."""
    if value is None or isinstance(value, bool):
        raise InvalidSeriesError(f"O parâmetro '{name}' deve ser um número válido.")
    try:
        val = float(value)
    except (ValueError, TypeError) as exc:
        raise InvalidSeriesError(
            f"O parâmetro '{name}' deve ser numérico."
        ) from exc

    if np.isnan(val) or np.isinf(val):
        raise InvalidSeriesError(
            f"O parâmetro '{name}' não pode ser NaN ou Inf."
        )

    if not allow_zero and val == 0.0:
        raise DegenerateSeriesError(f"O parâmetro '{name}' não pode ser zero.")

    if require_positive and val <= 0.0:
        if val == 0.0:
            raise DegenerateSeriesError(
                f"O parâmetro '{name}' deve ser estritamente positivo (> 0), recebeu 0."
            )
        raise InvalidSeriesError(
            f"O parâmetro '{name}' deve ser estritamente positivo (> 0), recebeu {val}."
        )

    return val
