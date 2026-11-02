"""Exceções de domínio da Statistical Engine."""


class StatisticalEngineError(ValueError):
    """Exceção base para erros de validação ou cálculo na Statistical Engine."""


class InvalidSeriesError(StatisticalEngineError):
    """Série de entrada inválida (NaN, Inf, dimensões incorretas, tamanhos divergentes ou valores fora do domínio)."""


class InsufficientDataError(StatisticalEngineError):
    """Quantidade de observações insuficiente para o cálculo estatístico ou graus de liberdade inválidos."""


class DegenerateSeriesError(StatisticalEngineError):
    """Série degenerada ou estatisticamente inválida (variância zero, colinearidade perfeita, divisão por zero ou ausência de reversão à média)."""
