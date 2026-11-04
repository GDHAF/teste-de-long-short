"""Testes unitários de validação de entradas e exceções da Statistical Engine."""

import numpy as np
import pandas as pd
import pytest

from statistical_engine.exceptions import (
    DegenerateSeriesError,
    InsufficientDataError,
    InvalidSeriesError,
)
from statistical_engine.validation import (
    validate_1d_series,
    validate_paired_series,
    validate_scalar_float,
)


def test_validate_1d_series_accepts_valid_inputs():
    for src in ([1.0, 2.0, 3.0], np.array([1, 2, 3]), pd.Series([1.0, 2.0, 3.0])):
        arr = validate_1d_series(src, min_length=3)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert arr.shape == (3,)


def test_validate_1d_series_rejects_none_and_2d():
    with pytest.raises(InvalidSeriesError):
        validate_1d_series(None)
    with pytest.raises(InvalidSeriesError):
        validate_1d_series([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(InvalidSeriesError):
        validate_1d_series(pd.DataFrame({"a": [1.0, 2.0]}))


def test_validate_1d_series_rejects_nan_and_inf():
    with pytest.raises(InvalidSeriesError, match="NaN"):
        validate_1d_series([1.0, np.nan, 3.0])
    with pytest.raises(InvalidSeriesError, match="Inf"):
        validate_1d_series([1.0, np.inf, 3.0])


def test_validate_1d_series_rejects_insufficient_length():
    with pytest.raises(InsufficientDataError):
        validate_1d_series([], min_length=1)
    with pytest.raises(InsufficientDataError):
        validate_1d_series([1.0, 2.0], min_length=5)


def test_validate_1d_series_require_positive():
    with pytest.raises(InvalidSeriesError, match="positivos"):
        validate_1d_series([10.0, 0.0, 12.0], require_positive=True)
    with pytest.raises(InvalidSeriesError, match="positivos"):
        validate_1d_series([10.0, -1.0, 12.0], require_positive=True)


def test_validate_paired_series_mismatched_lengths():
    with pytest.raises(InvalidSeriesError, match="mesmo comprimento"):
        validate_paired_series([1.0, 2.0, 3.0], [1.0, 2.0])


def test_validate_paired_series_incompatible_indices():
    s_a = pd.Series([10.0, 11.0, 12.0], index=["2024-01-01", "2024-01-02", "2024-01-03"])
    s_b = pd.Series([20.0, 21.0, 22.0], index=["2024-01-02", "2024-01-03", "2024-01-04"])
    with pytest.raises(InvalidSeriesError, match="índices"):
        validate_paired_series(s_a, s_b)

    # Série com índice duplicado
    s_dup = pd.Series([10.0, 11.0, 12.0], index=[1, 1, 2])
    with pytest.raises(InvalidSeriesError, match="duplicados"):
        validate_1d_series(s_dup)


def test_validate_scalar_float():
    assert validate_scalar_float(1.5, "x") == 1.5
    with pytest.raises(InvalidSeriesError):
        validate_scalar_float(np.nan, "x")
    with pytest.raises(DegenerateSeriesError):
        validate_scalar_float(0.0, "x", allow_zero=False)
