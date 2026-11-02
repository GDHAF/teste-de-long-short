"""Módulo de Inferência Estatística: IC de Beta, Testes t, Correlação de Pearson e Transformada de Fisher."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from .exceptions import (
    DegenerateSeriesError,
    InsufficientDataError,
    InvalidSeriesError,
)
from .validation import validate_paired_series, validate_scalar_float


def _validate_confidence(conf: float) -> float:
    val = validate_scalar_float(conf, "conf")
    if not (0.0 < val < 1.0):
        raise InvalidSeriesError(
            f"O nível de confiança 'conf' deve estar no intervalo aberto (0, 1), recebeu {val}."
        )
    return val


def _validate_df(n: int, k: int) -> int:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise InvalidSeriesError("O parâmetro 'n' deve ser um inteiro.")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise InvalidSeriesError("O parâmetro 'k' deve ser um inteiro >= 1.")
    df = int(n) - int(k) - 1
    if df <= 0:
        raise InsufficientDataError(
            f"Graus de liberdade insuficientes (n - k - 1 = {df} <= 0) para n={n}, k={k}."
        )
    return df


def _get_t_critical(
    df: int,
    conf: float,
    tail: str,
    use_manual_critical_values: bool = False,
) -> float:
    """Obtém o valor crítico da distribuição t-Student (exato ou tabela 60gl do manual)."""
    alpha = 1.0 - conf
    if tail in ("two-sided", "both"):
        if use_manual_critical_values:
            if np.isclose(conf, 0.95, atol=1e-4):
                return 2.00
            if np.isclose(conf, 0.99, atol=1e-4):
                return 2.66
            return float(stats.t.ppf(1.0 - alpha / 2.0, df=60))
        return float(stats.t.ppf(1.0 - alpha / 2.0, df=df))

    if tail in ("right", "left"):
        if use_manual_critical_values:
            if np.isclose(conf, 0.95, atol=1e-4):
                return 1.671
            if np.isclose(conf, 0.99, atol=1e-4):
                return 2.390
            return float(stats.t.ppf(1.0 - alpha, df=60))
        return float(stats.t.ppf(1.0 - alpha, df=df))

    raise InvalidSeriesError(
        f"Tipo de cauda 'tail' inválido: '{tail}'. Use 'right', 'left' ou 'two-sided'."
    )


def beta_confidence_interval(
    beta: float,
    se: float,
    n: int,
    k: int = 1,
    conf: float = 0.99,
    use_manual_critical_values: bool = False,
) -> tuple[float, float]:
    """Calcula o Intervalo de Confiança bilateral para o coeficiente angular Beta (p. 33-34).

    Fórmula:
        IC(beta; 1 - alpha) = [beta - t_(alpha/2, n-k-1) * EP(beta) ; beta + t_(alpha/2, n-k-1) * EP(beta)]

    Args:
        beta: Coeficiente angular estimado.
        se: Erro padrão de beta EP(beta) (> 0).
        n: Número de observações da regressão.
        k: Número de variáveis explicativas (1 para simples, 2 para múltipla com tempo).
        conf: Nível de confiança (ex: 0.99 ou 0.95).
        use_manual_critical_values: Se True, usa valores fixos de 60gl do manual (2.00 p/ 95%, 2.66 p/ 99%).

    Returns:
        Tupla (ic_beta_min, ic_beta_max).
    """
    beta_val = validate_scalar_float(beta, "beta")
    se_val = validate_scalar_float(se, "se", allow_zero=False, require_positive=True)
    conf_val = _validate_confidence(conf)
    df = _validate_df(n, k)

    t_crit = _get_t_critical(
        df=df,
        conf=conf_val,
        tail="two-sided",
        use_manual_critical_values=use_manual_critical_values,
    )
    margin = t_crit * se_val
    return (float(beta_val - margin), float(beta_val + margin))


def coefficient_t_test(
    coef: float,
    se: float,
    n: int,
    k: int = 1,
    tail: str = "right",
    conf: float = 0.99,
    use_manual_critical_values: bool = False,
) -> dict[str, Any]:
    """Realiza o teste t de significância para os coeficientes da regressão (p. 34-37).

    Regras do Manual:
        - Para Beta (Ativo B) e Tempo (T): teste unilateral à direita (tail='right'),
          H0: coef <= 0 vs H1: coef > 0.
        - Para Alpha (Intercepto): teste bilateral (tail='two-sided'),
          H0: alpha = 0 vs H1: alpha != 0.

    Args:
        coef: Valor estimado do coeficiente (beta, T ou alpha).
        se: Erro padrão do coeficiente (> 0).
        n: Número de observações da regressão.
        k: Número de variáveis explicativas (1 ou 2).
        tail: 'right' (unilateral à direita), 'left' (unilateral à esquerda) ou 'two-sided' (bilateral).
        conf: Nível de confiança principal para 'critical_value' e 'is_significant' (padrão 0.99).
        use_manual_critical_values: Se True, usa os valores críticos tabelados para 60gl do manual.

    Returns:
        Dicionário contendo:
            - 't_stat': float (estatística t calculada = coef / se)
            - 'p_value': float (p-valor do teste)
            - 'critical_value': float (valor crítico positivo para o nível conf)
            - 'is_significant': bool (se rejeita H0 no nível conf)
            - 'confidence_level': int (99, 95 ou 0, conforme classificação do painel operacional p. 55)
    """
    coef_val = validate_scalar_float(coef, "coef")
    se_val = validate_scalar_float(se, "se", allow_zero=False, require_positive=True)
    conf_val = _validate_confidence(conf)
    df = _validate_df(n, k)

    if tail not in ("right", "left", "two-sided", "both"):
        raise InvalidSeriesError(
            f"Parâmetro 'tail' inválido: '{tail}'. Valores aceitos: 'right', 'left', 'two-sided'."
        )

    t_stat = float(coef_val / se_val)
    crit_main = _get_t_critical(
        df=df,
        conf=conf_val,
        tail=tail,
        use_manual_critical_values=use_manual_critical_values,
    )
    crit_95 = _get_t_critical(
        df=df,
        conf=0.95,
        tail=tail,
        use_manual_critical_values=use_manual_critical_values,
    )
    crit_99 = _get_t_critical(
        df=df,
        conf=0.99,
        tail=tail,
        use_manual_critical_values=use_manual_critical_values,
    )

    if tail == "right":
        p_value = float(1.0 - stats.t.cdf(t_stat, df=df))
        is_sig = bool(t_stat >= crit_main)
        if t_stat >= crit_99:
            conf_level = 99
        elif t_stat >= crit_95:
            conf_level = 95
        else:
            conf_level = 0
    elif tail == "left":
        p_value = float(stats.t.cdf(t_stat, df=df))
        is_sig = bool(t_stat <= -crit_main)
        if t_stat <= -crit_99:
            conf_level = 99
        elif t_stat <= -crit_95:
            conf_level = 95
        else:
            conf_level = 0
    else:
        p_value = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df)))
        is_sig = bool(abs(t_stat) >= crit_main)
        if abs(t_stat) >= crit_99:
            conf_level = 99
        elif abs(t_stat) >= crit_95:
            conf_level = 95
        else:
            conf_level = 0

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "critical_value": crit_main,
        "is_significant": is_sig,
        "confidence_level": conf_level,
    }


def pearson_correlation(
    series_a: Any,
    series_b: Any,
    use_log_returns: bool = False,
) -> float:
    """Calcula o Coeficiente de Correlação de Pearson (rho) entre duas séries (p. 28-30).

    Fórmula:
        rho(X, Y) = COV(X, Y) / (sigma_X * sigma_Y)

    Args:
        series_a: Série do Ativo A (Y).
        series_b: Série do Ativo B (X).
        use_log_returns: Se True, calcula a correlação sobre os retornos logarítmicos
            ln(P_t / P_{t-1}) conforme nota da p. 30 do manual; se False (padrão),
            calcula diretamente sobre as séries fornecidas conforme Roteiro p. 10/13.

    Returns:
        Coeficiente de correlação de Pearson rho no intervalo [-1.0, 1.0].

    Raises:
        InvalidSeriesError: Se houver NaN/Inf, tamanhos diferentes ou preços <= 0 quando use_log_returns=True.
        InsufficientDataError: Se houver menos de 2 observações efetivas.
        DegenerateSeriesError: Se alguma das séries tiver desvio padrão igual a zero.
    """
    min_len = 3 if use_log_returns else 2
    arr_a, arr_b = validate_paired_series(
        series_a,
        series_b,
        name_a="series_a",
        name_b="series_b",
        min_length=min_len,
        require_positive=use_log_returns,
    )

    if use_log_returns:
        arr_a = np.log(arr_a[1:] / arr_a[:-1])
        arr_b = np.log(arr_b[1:] / arr_b[:-1])

    std_a = float(np.std(arr_a, ddof=1))
    std_b = float(np.std(arr_b, ddof=1))

    if std_a <= 1e-15 or std_b <= 1e-15:
        raise DegenerateSeriesError(
            "Correlação de Pearson indefinida: pelo menos uma das séries possui variância zero (constante)."
        )

    cov_ab = float(np.sum((arr_a - np.mean(arr_a)) * (arr_b - np.mean(arr_b))) / (arr_a.size - 1))
    corr = cov_ab / (std_a * std_b)
    return float(np.clip(corr, -1.0, 1.0))


def fisher_transform(corr: float) -> float:
    """Aplica a Transformada de Fisher sobre o coeficiente de correlação (p. 31).

    Fórmula:
        F(rho) = 0.5 * ln((1 + rho) / (1 - rho))
    """
    rho = validate_scalar_float(corr, "corr")
    if abs(rho) > 1.0:
        raise InvalidSeriesError(
            f"O coeficiente de correlação deve estar em [-1, 1], recebeu {rho}."
        )
    if abs(rho) == 1.0:
        raise DegenerateSeriesError(
            f"A Transformada de Fisher é indefinida (divisão por zero / infinito) para |rho| = 1.0 (recebeu {rho})."
        )
    return float(0.5 * np.log((1.0 + rho) / (1.0 - rho)))


def inverse_fisher_transform(f_val: float) -> float:
    """Aplica a Transformada Inversa de Fisher (p. 32).

    Fórmula:
        rho = (exp(2 * F) - 1) / (exp(2 * F) + 1) = tanh(F)
    """
    val = validate_scalar_float(f_val, "f_val")
    return float(np.tanh(val))


def fisher_test(
    corr: float,
    n: int,
    conf: float = 0.99,
    use_manual_critical_values: bool = False,
) -> tuple[float, float]:
    """Calcula o Intervalo de Confiança para o Coeficiente de Correlação via Teste de Fisher (p. 31-32).

    Fórmulas:
        F(rho) = 0.5 * ln((1 + rho) / (1 - rho))
        EP = 1 / sqrt(n - 3)
        IC(F; 1 - alpha) = [F(rho) - z_(alpha/2) * EP ; F(rho) + z_(alpha/2) * EP]
        [min, max] = Transformada Inversa de Fisher sobre IC(F)

    Args:
        corr: Coeficiente de correlação de Pearson rho em (-1, 1).
        n: Tamanho amostral N (exige n > 3).
        conf: Nível de confiança (padrão 0.99; aceita ex: 0.95, 0.99).
        use_manual_critical_values: Se True, usa z=1.96 (95%) e z=2.575 (99%) conforme p. 32 do manual.

    Returns:
        Tupla (fisher_min, fisher_max) na escala original de correlação (-1, 1).
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise InvalidSeriesError("O tamanho amostral 'n' deve ser um inteiro.")
    if int(n) <= 3:
        raise InsufficientDataError(
            f"O Teste de Fisher exige tamanho amostral n > 3 (EP = 1 / sqrt(n - 3)), recebeu n={n}."
        )

    conf_val = _validate_confidence(conf)
    f_rho = fisher_transform(corr)
    se_fisher = 1.0 / np.sqrt(float(n) - 3.0)

    alpha = 1.0 - conf_val
    if use_manual_critical_values:
        if np.isclose(conf_val, 0.95, atol=1e-4):
            z_crit = 1.96
        elif np.isclose(conf_val, 0.99, atol=1e-4):
            z_crit = 2.575
        else:
            z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    else:
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    f_min = f_rho - z_crit * se_fisher
    f_max = f_rho + z_crit * se_fisher

    return (inverse_fisher_transform(f_min), inverse_fisher_transform(f_max))
