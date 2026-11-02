"""Pacote público da Statistical Engine do sistema Long & Short por Cointegração."""

from .aggregator import pair_stats
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
    fisher_transform,
    inverse_fisher_transform,
    pearson_correlation,
)
from .regression import compute_fitted_values, compute_residuals, linear_regression
from .stability import (
    beta_rotation,
    coefficient_of_variation,
    historical_volatility,
)
from .stationarity import (
    ENGLE_GRANGER_CRITICAL_VALUES,
    classify_cointegration_level,
    dickey_fuller,
    half_life,
    z_score,
)
from .validation import (
    validate_1d_series,
    validate_paired_series,
    validate_scalar_float,
)

__all__ = [
    # Exceções
    "StatisticalEngineError",
    "InvalidSeriesError",
    "InsufficientDataError",
    "DegenerateSeriesError",
    # Validação
    "validate_1d_series",
    "validate_paired_series",
    "validate_scalar_float",
    # Regressão
    "linear_regression",
    "compute_fitted_values",
    "compute_residuals",
    # Inferência e Correlação
    "beta_confidence_interval",
    "coefficient_t_test",
    "pearson_correlation",
    "fisher_transform",
    "inverse_fisher_transform",
    "fisher_test",
    # Estacionariedade e Reversão à Média
    "ENGLE_GRANGER_CRITICAL_VALUES",
    "classify_cointegration_level",
    "dickey_fuller",
    "z_score",
    "half_life",
    # Estabilidade e Volatilidade
    "beta_rotation",
    "coefficient_of_variation",
    "historical_volatility",
    # Agregador
    "pair_stats",
]
