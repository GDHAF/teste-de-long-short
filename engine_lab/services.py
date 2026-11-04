"""Serviço de geração de dados dummy determinísticos e integração com a statistical_engine."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from statistical_engine import (
    coefficient_of_variation,
    pair_stats,
    pearson_correlation,
)

# Constantes fixas do cenário dummy único de demonstração
DUMMY_SEED: int = 42
DUMMY_TICKER_A: str = "BBAS3"
DUMMY_TICKER_B: str = "CSAN3"
DUMMY_TOTAL_OBSERVATIONS: int = 260
DUMMY_PRIMARY_PERIODS: int = 250
DUMMY_ROLLING_WINDOW: int = 60
DUMMY_CONFIDENCE_LEVEL: float = 0.99

# Os 10 períodos padrão do Painel Operacional do Manual do TCC (p. 54-55)
MANUAL_PERIODS: tuple[int, ...] = (250, 240, 220, 200, 180, 160, 140, 120, 100, 70)
VOLATILITY_WINDOWS: tuple[int, ...] = (10, 21, 42, 63)


def generate_dummy_pair_data() -> dict[str, Any]:
    """Gera um único conjunto de dados dummy determinístico para validação do processo matemático.

    Constrói duas séries de preços diários positivamente correlacionadas e cointegradas:
        - Ativo B (CSAN3, variável independente X_t): passeio aleatório com variações diárias realistas.
        - Resíduo estacionário (e_t): processo Ornstein-Uhlenbeck / AR(1) com reversão rápida (phi = 0.62)
          e distorção controlada na última observação para demonstrar gatilho operacional (|Z| > 2).
        - Ativo A (BBAS3, variável dependente Y_t): Y_t = alpha_true + beta_true * X_t + e_t.

    Returns:
        Dicionário contendo:
            - 'dates': list[str] com 260 datas de dias úteis (formato YYYY-MM-DD)
            - 'dates_br': list[str] com 260 datas no formato DD/MM/YYYY
            - 'prices_a': pd.Series indexada por DatetimeIndex (Ativo A - Dependente Y)
            - 'prices_b': pd.Series indexada por DatetimeIndex (Ativo B - Independente X)
            - 'params': dict com os parâmetros geradores verdadeiros (seed, alpha_true, beta_true, phi_true)
    """
    rng = np.random.default_rng(DUMMY_SEED)
    n = DUMMY_TOTAL_OBSERVATIONS

    bdate_index = pd.bdate_range(end="2025-10-31", periods=n)
    dates_iso = [d.strftime("%Y-%m-%d") for d in bdate_index]
    dates_br = [d.strftime("%d/%m/%Y") for d in bdate_index]

    # Série X_t (Ativo B - Independente): passeio aleatório suave em torno de R$ 38,00 a R$ 48,00
    steps_x = rng.normal(loc=0.015, scale=0.42, size=n)
    raw_x = 38.50 + np.cumsum(steps_x)
    prices_b_arr = np.maximum(raw_x, 15.0)

    # Processo estacionário AR(1) / Ornstein-Uhlenbeck para o resíduo: e_t = phi * e_{t-1} + u_t
    phi_true = 0.62
    sigma_u = 0.28
    e_arr = np.zeros(n, dtype=np.float64)
    shocks = rng.normal(loc=0.0, scale=sigma_u, size=n)
    for t in range(1, n):
        e_arr[t] = phi_true * e_arr[t - 1] + shocks[t]

    # Distorção controlada no último pregão para demonstrar o sinal de entrada (Z-Score < -2.0)
    e_arr[-2] = -0.35
    e_arr[-1] = -0.92

    alpha_true = 4.20
    beta_true = 0.72
    prices_a_arr = alpha_true + beta_true * prices_b_arr + e_arr

    series_a = pd.Series(prices_a_arr, index=bdate_index, name=DUMMY_TICKER_A, dtype=np.float64)
    series_b = pd.Series(prices_b_arr, index=bdate_index, name=DUMMY_TICKER_B, dtype=np.float64)

    return {
        "dates": dates_iso,
        "dates_br": dates_br,
        "prices_a": series_a,
        "prices_b": series_b,
        "params": {
            "seed": DUMMY_SEED,
            "alpha_true": alpha_true,
            "beta_true": beta_true,
            "phi_true": phi_true,
            "sigma_u": sigma_u,
            "total_observations": n,
        },
    }


def build_dummy_dashboard_context() -> dict[str, Any]:
    """Executa a statistical_engine sobre o par dummy e monta o contexto somente leitura do Dashboard.

    Toda estatística exibida provém diretamente de `statistical_engine.pair_stats`.
    """
    dummy = generate_dummy_pair_data()
    series_a: pd.Series = dummy["prices_a"]
    series_b: pd.Series = dummy["prices_b"]
    dates_br: list[str] = dummy["dates_br"]

    # 1. Execução principal da engine para a janela de referência (N = 250 períodos)
    primary = pair_stats(
        prices_a=series_a,
        prices_b=series_b,
        n_periods=DUMMY_PRIMARY_PERIODS,
        with_time=False,
        conf=DUMMY_CONFIDENCE_LEVEL,
        rolling_window=DUMMY_ROLLING_WINDOW,
        volatility_windows=VOLATILITY_WINDOWS,
    )

    # Execução complementar com tendência temporal (Regressão Múltipla) apenas para exibição comparativa da equação
    multiple_reg_stats = pair_stats(
        prices_a=series_a,
        prices_b=series_b,
        n_periods=DUMMY_PRIMARY_PERIODS,
        with_time=True,
        conf=DUMMY_CONFIDENCE_LEVEL,
        rolling_window=DUMMY_ROLLING_WINDOW,
        volatility_windows=VOLATILITY_WINDOWS,
    )

    # Execução do par invertido (B x A) na janela principal (Manual p. 62)
    inverse_primary = pair_stats(
        prices_a=series_b,
        prices_b=series_a,
        n_periods=DUMMY_PRIMARY_PERIODS,
        with_time=False,
        conf=DUMMY_CONFIDENCE_LEVEL,
        rolling_window=DUMMY_ROLLING_WINDOW,
        volatility_windows=VOLATILITY_WINDOWS,
    )

    # 2. Painel Operacional Multi-Período (10 períodos do Manual: 250 a 70 dias)
    multi_period_rows: list[dict[str, Any]] = []
    cointegrated_count = 0
    entry_signal_count = 0
    b_x_a_cointegrated_count = 0

    for period in MANUAL_PERIODS:
        st_ab = pair_stats(
            prices_a=series_a,
            prices_b=series_b,
            n_periods=period,
            with_time=False,
            conf=DUMMY_CONFIDENCE_LEVEL,
            rolling_window=min(DUMMY_ROLLING_WINDOW, period - 10),
            volatility_windows=VOLATILITY_WINDOWS,
        )
        st_ba = pair_stats(
            prices_a=series_b,
            prices_b=series_a,
            n_periods=period,
            with_time=False,
            conf=DUMMY_CONFIDENCE_LEVEL,
            rolling_window=min(DUMMY_ROLLING_WINDOW, period - 10),
            volatility_windows=VOLATILITY_WINDOWS,
        )

        price_a_now = float(series_a.iloc[-1])
        price_b_now = float(series_b.iloc[-1])
        beta_p = float(st_ab["beta"])
        z_now = float(st_ab["z_score_current"])

        # Relação Financeiro % conforme Manual (p. 27 / Roteiro p. 15): |beta * P_B| / P_A * 100
        # Quando comprado em A (Z < 0), Comprado = P_A e Vendido = beta * P_B -> (P_A / (beta * P_B)) * 100
        # Quando vendido em A (Z > 0), Comprado = beta * P_B e Vendido = P_A -> ((beta * P_B) / P_A) * 100
        if z_now < 0:
            financeiro_pct = (price_a_now / (abs(beta_p) * price_b_now)) * 100.0
        else:
            financeiro_pct = ((abs(beta_p) * price_b_now) / price_a_now) * 100.0

        is_coint = bool(st_ab["is_cointegrated"])
        is_ba_coint = bool(st_ba["is_cointegrated"])
        is_entry = abs(z_now) >= 2.0

        if is_coint:
            cointegrated_count += 1
        if is_entry:
            entry_signal_count += 1
        if is_ba_coint:
            b_x_a_cointegrated_count += 1

        multi_period_rows.append(
            {
                "n_periods": period,
                "dickey_fuller": st_ab["dickey_fuller"],
                "cointegration_level": st_ab["cointegration_level"],
                "is_cointegrated": is_coint,
                "beta": beta_p,
                "alpha": st_ab["alpha"],
                "z_score_current": z_now,
                "is_entry_signal": is_entry,
                "fisher_min_pct": st_ab["fisher_min"] * 100.0 if st_ab["fisher_min"] is not None else None,
                "fisher_max_pct": st_ab["fisher_max"] * 100.0 if st_ab["fisher_max"] is not None else None,
                "financeiro_pct": financeiro_pct,
                "half_life": st_ab["half_life"],
                "ic_beta_min": st_ab["ic_beta_min"],
                "ic_beta_max": st_ab["ic_beta_max"],
                "t_beta_level": st_ab["t_beta_level"],
                "t_alpha_level": st_ab["t_alpha_level"],
                "bx_a_cointegrated": is_ba_coint,
                "bx_a_level": st_ba["cointegration_level"],
            }
        )

    # 3. Interpretação do sinal na janela principal (N = 250) segundo o Manual (p. 44 e p. 67)
    z_curr = float(primary["z_score_current"])
    beta_curr = float(primary["beta"])
    alpha_curr = float(primary["alpha"])
    std_resid = float(primary["std_residual"])
    price_a_curr = float(series_a.iloc[-1])
    price_b_curr = float(series_b.iloc[-1])

    if z_curr <= -2.0:
        signal_status = "SINAL DE ENTRADA ATIVO (Z ≤ -2,00 DP)"
        signal_badge_class = "badge-signal-long"
        action_a = f"COMPRA (Long) na Dependente ({DUMMY_TICKER_A})"
        action_b = f"VENDA (Short) na Independente ({DUMMY_TICKER_B})"
        financeiro_primary_pct = (price_a_curr / (abs(beta_curr) * price_b_curr)) * 100.0
        # Alvos ilustrativos para 1,0 desvio de gain e 1,0 desvio de loss (Manual p. 67-68)
        target_z_gain = z_curr + 1.0
        target_z_loss = z_curr - 1.0
        est_price_a_gain = price_a_curr + (1.0 * std_resid)
        est_price_a_loss = price_a_curr - (1.0 * std_resid)
    elif z_curr >= 2.0:
        signal_status = "SINAL DE ENTRADA ATIVO (Z ≥ +2,00 DP)"
        signal_badge_class = "badge-signal-short"
        action_a = f"VENDA (Short) na Dependente ({DUMMY_TICKER_A})"
        action_b = f"COMPRA (Long) na Independente ({DUMMY_TICKER_B})"
        financeiro_primary_pct = ((abs(beta_curr) * price_b_curr) / price_a_curr) * 100.0
        target_z_gain = z_curr - 1.0
        target_z_loss = z_curr + 1.0
        est_price_a_gain = price_a_curr - (1.0 * std_resid)
        est_price_a_loss = price_a_curr + (1.0 * std_resid)
    else:
        signal_status = "AGUARDANDO GATILHO (|Z| < 2,00 DP)"
        signal_badge_class = "badge-neutral"
        action_a = f"Neutro em {DUMMY_TICKER_A}"
        action_b = f"Neutro em {DUMMY_TICKER_B}"
        financeiro_primary_pct = (price_a_curr / (abs(beta_curr) * price_b_curr)) * 100.0
        target_z_gain = 0.0
        target_z_loss = 3.0 if z_curr >= 0 else -3.0
        est_price_a_gain = float(primary["fitted_values"][-1])
        est_price_a_loss = price_a_curr

    # Quantidades ilustrativas para lote base de 1.000 ações de A (Manual p. 26)
    base_qty_a = 1000
    raw_qty_b = base_qty_a * beta_curr
    rounded_qty_b = int(round(raw_qty_b / 100.0) * 100)

    # 4. Séries da janela principal (N = 250) para tabelas e gráficos
    dates_window = dates_br[-DUMMY_PRIMARY_PERIODS:]
    prices_a_win = series_a.to_numpy(dtype=np.float64)[-DUMMY_PRIMARY_PERIODS:]
    prices_b_win = series_b.to_numpy(dtype=np.float64)[-DUMMY_PRIMARY_PERIODS:]
    fitted_win = primary["fitted_values"]
    residuals_win = primary["residuals"]
    z_scores_win = primary["z_scores"]
    beta_rot_arr = primary["beta_rotation"]

    beta_rot_list = [float(v) for v in beta_rot_arr] if beta_rot_arr is not None else []
    beta_rot_dates = dates_window[-len(beta_rot_list) :] if beta_rot_list else []
    beta_rot_mean = float(np.mean(beta_rot_arr)) if beta_rot_arr is not None else 0.0
    beta_rot_std = float(np.std(beta_rot_arr, ddof=1)) if beta_rot_arr is not None else 0.0

    # Últimas 12 observações para inspeção tabular direta dos cálculos
    sample_table_rows: list[dict[str, Any]] = []
    start_idx = DUMMY_PRIMARY_PERIODS - 12
    for idx in range(start_idx, DUMMY_PRIMARY_PERIODS):
        sample_table_rows.append(
            {
                "t": idx + 1,
                "date": dates_window[idx],
                "price_a": float(prices_a_win[idx]),
                "price_b": float(prices_b_win[idx]),
                "fitted_a": float(fitted_win[idx]),
                "residual": float(residuals_win[idx]),
                "z_score": float(z_scores_win[idx]),
                "is_last": idx == (DUMMY_PRIMARY_PERIODS - 1),
            }
        )

    # Volatilidade Histórica formatada em percentual para as janelas 10, 21, 42 e 63 dias
    volatility_rows: list[dict[str, Any]] = []
    for w in VOLATILITY_WINDOWS:
        vol_a = primary["volatility_a"].get(w)
        vol_b = primary["volatility_b"].get(w)
        vol_beta = primary["volatility_beta_rotation"].get(w)
        volatility_rows.append(
            {
                "window": w,
                "vol_a_pct": float(vol_a * 100.0) if vol_a is not None else None,
                "vol_b_pct": float(vol_b * 100.0) if vol_b is not None else None,
                "vol_beta_pct": float(vol_beta * 100.0) if vol_beta is not None else None,
            }
        )

    # Payload JSON para os gráficos em Canvas HTML5
    chart_payload = {
        "dates": dates_window,
        "prices_a": [round(float(v), 4) for v in prices_a_win],
        "prices_b": [round(float(v), 4) for v in prices_b_win],
        "fitted_values": [round(float(v), 4) for v in fitted_win],
        "residuals": [round(float(v), 4) for v in residuals_win],
        "z_scores": [round(float(v), 4) for v in z_scores_win],
        "beta_rotation_dates": beta_rot_dates,
        "beta_rotation": [round(float(v), 4) for v in beta_rot_list],
        "beta_rotation_mean": round(beta_rot_mean, 4),
        "beta_rotation_upper": round(beta_rot_mean + 2.0 * beta_rot_std, 4),
        "beta_rotation_lower": round(beta_rot_mean - 2.0 * beta_rot_std, 4),
    }

    cv_beta_pct = (
        float(primary["cv_beta"]) * 100.0 if primary["cv_beta"] is not None else 0.0
    )
    cv_across_periods_pct = float(
        coefficient_of_variation(
            [r["beta"] for r in multi_period_rows], ddof=0, as_percentage=True
        )
    )
    if cv_beta_pct < 15.0:
        cv_classification = "Baixa dispersão, homogênea, estável (< 15%)"
    elif cv_beta_pct <= 30.0:
        cv_classification = "Moderada dispersão (15% a 30%)"
    else:
        cv_classification = "Alta dispersão, heterogênea (> 30%)"

    corr_log_ret = float(
        pearson_correlation(prices_a_win, prices_b_win, use_log_returns=True)
    )

    return {
        "meta": {
            "ticker_a": DUMMY_TICKER_A,
            "ticker_b": DUMMY_TICKER_B,
            "n_periods": DUMMY_PRIMARY_PERIODS,
            "total_observations": DUMMY_TOTAL_OBSERVATIONS,
            "rolling_window": DUMMY_ROLLING_WINDOW,
            "conf_pct": int(round(DUMMY_CONFIDENCE_LEVEL * 100)),
            "seed": DUMMY_SEED,
            "last_date": dates_window[-1],
            "first_date": dates_window[0],
            "equation_simple": (
                f"{DUMMY_TICKER_A}_t = {alpha_curr:.4f} + {beta_curr:.4f} · {DUMMY_TICKER_B}_t + e_t"
            ),
            "equation_multiple": (
                f"{DUMMY_TICKER_A}_t = {multiple_reg_stats['alpha']:.4f} + "
                f"{multiple_reg_stats['beta']:.4f} · {DUMMY_TICKER_B}_t + "
                f"{multiple_reg_stats['t_coef']:.5f} · Tempo_t + e_t"
            ),
            "true_params": dummy["params"],
        },
        "primary_stats": primary,
        "multiple_reg_stats": multiple_reg_stats,
        "inverse_primary": inverse_primary,
        "derived_display": {
            "corr_pct": float(primary["corr"]) * 100.0,
            "corr_log_returns_pct": corr_log_ret * 100.0,
            "fisher_min_pct": (
                float(primary["fisher_min"]) * 100.0 if primary["fisher_min"] is not None else None
            ),
            "fisher_max_pct": (
                float(primary["fisher_max"]) * 100.0 if primary["fisher_max"] is not None else None
            ),
            "r_squared_pct": float(primary["r_squared"]) * 100.0,
            "residual_current": float(residuals_win[-1]),
            "fitted_current": float(fitted_win[-1]),
            "beta_rotation_current": float(beta_rot_list[-1]) if beta_rot_list else None,
            "beta_rotation_mean": beta_rot_mean,
            "beta_rotation_std": beta_rot_std,
            "cv_beta_pct": cv_beta_pct,
            "cv_across_periods_pct": cv_across_periods_pct,
            "cv_classification": cv_classification,
        },
        "signal_summary": {
            "status_text": signal_status,
            "badge_class": signal_badge_class,
            "is_entry_signal": abs(z_curr) >= 2.0,
            "z_score_current": z_curr,
            "action_a": action_a,
            "action_b": action_b,
            "price_a_current": price_a_curr,
            "price_b_current": price_b_curr,
            "base_qty_a": base_qty_a,
            "raw_qty_b": raw_qty_b,
            "rounded_qty_b": rounded_qty_b,
            "financeiro_pct": financeiro_primary_pct,
            "target_z_gain": target_z_gain,
            "target_z_loss": target_z_loss,
            "est_price_a_gain": est_price_a_gain,
            "est_price_a_loss": est_price_a_loss,
        },
        "panel_summary": {
            "total_periods": len(MANUAL_PERIODS),
            "cointegrated_count": cointegrated_count,
            "entry_signal_count": entry_signal_count,
            "b_x_a_cointegrated_count": b_x_a_cointegrated_count,
        },
        "multi_period_rows": multi_period_rows,
        "volatility_rows": volatility_rows,
        "sample_table_rows": sample_table_rows,
        "chart_payload_json": json.dumps(chart_payload),
    }
