# Especificação Técnica (Spec) — Statistical Engine (Núcleo Estatístico)

## 1. Objetivo
Implementar o núcleo matemático e econométrico (**Parte 2 — Núcleo Estatístico**) do sistema de análise de operações **Long & Short por Cointegração**, baseado na metodologia de Engle-Granger descrita no Manual de Referência e no Roteiro do Projeto de TCC.

A engine deve ser composta por funções puras em Python (utilizando `numpy`, `pandas`, `scipy` e `statsmodels`), sem efeitos colaterais e completamente independente de Django, banco de dados SQLite/ORM, interface gráfica ou leitura de arquivos externos (CSV/Parquet/APIs).

---

## 2. Análise do Código Existente (`long_e_short.py`)
O arquivo existente [`long_e_short.py`](../../long_e_short.py) implementa a classe `LongEShort` com o método `calcular(preco_a, preco_b, coef_a, coef_b, tolerancia)`:
- **Fórmula atual**: $\text{spread} = A \cdot P_A + B \cdot P_B$.
- **Regra atual**:
  - Se $|\text{spread}| \le \text{tolerancia} \implies \{'A': 0, 'B': 0\}$.
  - Se $\text{spread} > \text{tolerancia} \implies \{'A': -|A|, 'B': |B|\}$ (venda em A, compra em B).
  - Se $\text{spread} < -\text{tolerancia} \implies \{'A': |A|, 'B': -|B|\}$ (compra em A, venda em B).
- **Diagnóstico de Compatibilidade**:
  - **O que é compatível**: A convenção de direção operacional na presença de desvio positivo/negativo (quando o ativo dependente A está caro frente a B, vende-se A e compra-se B na proporção dos coeficientes; quando A está barato, compra-se A e vende-se B).
  - **Limitações frente ao TCC**: Não estima a regressão ($\alpha, \beta, T$), omite o intercepto $\alpha$ no cálculo do resíduo ($e_t = Y_t - (\alpha + \beta X_t)$), utiliza tolerância absoluta em vez de padronização por Z-Score ($Z_t = (e_t - \bar{e})/\sigma_e$) e não realiza os testes econométricos (Dickey-Fuller, Fisher, IC de $\beta$, testes $t$, meia-vida de Ornstein-Uhlenbeck, Beta Rotation, CV($\beta$) e Volatilidade Histórica).
  - **Decisão**: Preservar [`long_e_short.py`](../../long_e_short.py) intacto e coberto por testes automatizados de regressão, construindo o pacote modular `statistical_engine/` para o núcleo econométrico completo.

---

## 3. Convenções Matemáticas e Regras de Negócio

### 3.1. Convenção de Variáveis
- **Ativo A (`prices_a` / $Y_t$)**: Variável **dependente** (eixo $Y$ da regressão). É quem comanda o sinal da operação.
- **Ativo B (`prices_b` / $X_t$)**: Variável **independente** (eixo $X$ da regressão). Seu coeficiente angular $\beta$ define a proporção (hedge ratio) da operação ($Q_B = \beta \cdot Q_A$).
- **Ordenação Temporal**: Todas as séries devem estar ordenadas cronologicamente do dado **mais antigo** ($t = 1$) para o **mais recente** ($t = N$).

### 3.2. Regressão Linear OLS Simples e Múltipla (`linear_regression`)
- **Simples (`with_time=False`, $k = 1$)**:
  $$Y_t = \alpha + \beta X_t + e_t, \quad gl = N - 2$$
- **Múltipla com Tendência Temporal (`with_time=True`, $k = 2$)**:
  $$Y_t = \alpha + \beta X_t + T \cdot t + e_t, \quad t \in \{1, 2, \dots, N\}, \quad gl = N - 3$$
- **Valores Estimados e Resíduos**:
  $$\hat{Y}_t = \alpha + \beta X_t (+ T \cdot t), \quad e_t = Y_t - \hat{Y}_t$$
- **Erros Padrão**: Extraídos da diagonal da matriz de covariância dos estimadores OLS $\widehat{\text{Cov}}(\hat{\theta}) = s^2 (X^T X)^{-1}$, onde $s^2 = \frac{\sum_{t=1}^N e_t^2}{N - k - 1}$.
- **Desvio Padrão do Resíduo ($\sigma_e$)**: Desvio padrão amostral (`ddof=1`) da série $e_t$:
  $$\sigma_e = \sqrt{\frac{1}{N - 1} \sum_{t=1}^N (e_t - \bar{e})^2}$$

### 3.3. Intervalo de Confiança para $\beta$ (`beta_confidence_interval`)
- **Fórmula**:
  $$IC(\beta; 1 - \alpha_{\text{sig}}) = \left[\beta - t_{\alpha_{\text{sig}}/2, (N-k-1)} \cdot EP(\beta) \;;\; \beta + t_{\alpha_{\text{sig}}/2, (N-k-1)} \cdot EP(\beta)\right]$$
- Filtro de validade no manual: ambos os limites (`ic_beta_min`, `ic_beta_max`) devem ser estritamente positivos.

### 3.4. Teste $t$ de Significância dos Coeficientes (`coefficient_t_test`)
- **Estatística**:
  $$t_{\text{calc}} = \frac{\hat{\theta}}{EP(\hat{\theta})} \sim t_{(N - k - 1)}$$
- **Hipóteses por coeficiente (conforme Manual p. 35–37)**:
  - **Coeficiente Angular $\beta$**: Teste **unilateral à direita** ($H_0: \beta \le 0$ vs. $H_1: \beta > 0$). Significativo se $t_{\text{calc}} > t_{\text{crit}}$.
  - **Coeficiente Temporal $T$**: Teste **unilateral à direita** ($H_0: T \le 0$ vs. $H_1: T > 0$). Significativo se $t_{\text{calc}} > t_{\text{crit}}$.
  - **Coeficiente Linear $\alpha$ (Intercepto)**: Teste **bilateral** ($H_0: \alpha = 0$ vs. $H_1: \alpha \ne 0$). Significativo se $|t_{\text{calc}}| \ge t_{\text{crit}}$.
- **Classificação de Nível de Confiança (`confidence_level`)**: Retorna `99` (significativo a 99%), `95` (significativo a 95%) ou `0` (não significativo).

### 3.5. Teste de Dickey-Fuller para Cointegração de Engle-Granger (`dickey_fuller`)
- **Modelo AR(1) sobre os Resíduos $e_t$ (Manual p. 39–41)**:
  $$\Delta e_t = e_t - e_{t-1} = \alpha_{df} + \gamma e_{t-1} + u_t, \quad t = 2, \dots, N$$
- **Estatística $\tau$ (Tau)**:
  $$\tau = \frac{\gamma}{EP(\gamma)}, \quad EP(\gamma) = \frac{\sqrt{\frac{\sum u_t^2}{(N - 1) - 2}}}{\sqrt{\sum (e_{t-1} - \overline{e_{t-1}})^2}}$$
- **Valores Críticos de Engle-Granger (Manual p. 41–42)**:
  - $\tau < -4,32 \implies$ Estacionário / Cointegrado com **99%** de confiança (`cointegration_level = 99`).
  - $-4,32 \le \tau < -3,67 \implies$ Estacionário / Cointegrado com **95%** de confiança (`cointegration_level = 95`).
  - $-3,67 \le \tau < -3,28 \implies$ Estacionário / Cointegrado com **90%** de confiança (`cointegration_level = 90`).
  - $\tau \ge -3,28 \implies$ Não estacionário / Não cointegrado (`cointegration_level = 0`).

### 3.6. Correlação de Pearson e Transformada de Fisher (`pearson_correlation`, `fisher_test`)
- **Correlação de Pearson ($\rho$)**:
  $$\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]$$
- **Transformada de Fisher**:
  $$F(\rho) = \frac{1}{2} \ln\left(\frac{1 + \rho}{1 - \rho}\right) = \operatorname{arctanh}(\rho), \quad EP = \frac{1}{\sqrt{N - 3}}$$
- **Intervalo de Confiança em escala Fisher e Inversa**:
  $$IC_F = \left[F(\rho) - z_{\alpha/2} \cdot \frac{1}{\sqrt{N - 3}} \;;\; F(\rho) + z_{\alpha/2} \cdot \frac{1}{\sqrt{N - 3}}\right]$$
  $$\rho_{\min}, \rho_{\max} = \frac{e^{2 F} - 1}{e^{2 F} + 1} = \tanh(F)$$

### 3.7. Z-Score dos Resíduos (`z_score`)
- **Fórmula (Manual p. 43)**:
  $$Z_t = \frac{e_t - \bar{e}}{\sigma_e}$$
  Onde $\bar{e}$ é a média amostral do resíduo na janela e $\sigma_e$ é o desvio padrão amostral (`ddof=1`). O Z-Score atual (`z_score_current`) corresponde ao último elemento $Z_N$.

### 3.8. Meia-Vida pelo Modelo de Ornstein-Uhlenbeck (`half_life`)
- **Fórmula (Manual p. 47, Roteiro p. 9 e 13)**:
  - A partir da regressão $\Delta e_t = \alpha_{df} + \gamma e_{t-1} + u_t$, a velocidade de reversão à média discreta é $\theta = -\gamma$.
  - Para $\gamma < 0$ ($\theta > 0$):
    $$\text{HL} = \frac{\ln(2)}{\theta} = -\frac{\ln(2)}{\gamma}$$
  - Se $\gamma \ge 0$, o processo não apresenta reversão à média (é explosivo ou passeio aleatório).

### 3.9. Beta Rotation por Janela Móvel (`beta_rotation`)
- Calcula o coeficiente angular $\beta$ em janelas móveis deslizantes de tamanho `window` sobre as séries `prices_a` e `prices_b`.
- Para séries de tamanho $M$ e janela $W$ ($M \ge W$), produz $M - W + 1$ estimativas $\beta_t$, correspondentes a cada sub-janela contígua `[i : i + W]`.

### 3.10. Coeficiente de Variação (`coefficient_of_variation`)
- **Fórmula (Manual p. 65, Roteiro p. 9)**:
  $$\text{CV} = \frac{\sigma}{|\mu|}$$
  (Retornado em razão decimal e/ou percentual conforme contrato; na agregação e nos testes documentamos claramente a escala).

### 3.11. Volatilidade Histórica (`historical_volatility`)
- **Retorno Logarítmico (Manual p. 52)**:
  $$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$
- **Escalas suportadas para os últimos $w$ retornos** ($w \in \{10, 21, 42, 63\}$):
  - Diária (`scale="daily"`): $\sigma(r_{[-w:]})$ (`ddof=1`).
  - No período (`scale="period"`, Manual p. 52): $\sigma(r_{[-w:]}) \times \sqrt{w}$.
  - Anualizada (`scale="annualized"`, Roteiro p. 13 e Manual p. 52–53): $\sigma(r_{[-w:]}) \times \sqrt{252}$.

---

## 4. Ambiguidades Identificadas nos Documentos de Referência e Tratamento Proposto

1. **Correlação de Pearson sobre Preços vs. Log-Retornos**:
   - *Manual (p. 30)* orienta calcular sobre log-retornos $\ln(P_t/P_{t-1})$, mas na *p. 32* usa $N=120$ no denominador $\sqrt{N-3}$ da Transformada de Fisher. Já o *Roteiro do TCC (p. 10 e 13)* define $\rho$ diretamente entre $Y_t$ e $X_t$ com tamanho amostral $N$.
   - *Solução*: Suportar o parâmetro `use_log_returns: bool = False` em `pearson_correlation` e `corr_on_log_returns: bool = False` em `pair_stats`.
2. **Valores Críticos Fixos ($gl=60$ / $z=2,575$) vs. Quantis Exatos (`scipy.stats`)**:
   - *Manual (p. 32–37)* utiliza valores tabelados fixos ($z_{99\%}=2,575$; $t_{60gl, 99\%}=2,66$ bilateral e $2,390$ unilateral; $t_{60gl, 95\%}=2,00$ bilateral e $1,671$ unilateral). O *Roteiro (p. 13–14)* passa `n` e `k` para cálculo com $gl = n - k - 1$ via `scipy.stats`.
   - *Solução*: Calcular por padrão via `scipy.stats` com $gl = n - k - 1$ exatos, oferecendo `use_manual_critical_values: bool = False` para reproduzir exatamente os exemplos manuais.
3. **Coeficiente de Variação com `ddof=1` vs. `ddof=0`**:
   - *Manual (p. 28)* define desvio padrão com $n-1$ (`ddof=1`), mas os exemplos da *p. 65* (`7,01%` e `34,40%`) foram gerados com desvio padrão populacional (`ddof=0`).
   - *Solução*: Parametrizar `ddof: int = 1` em `coefficient_of_variation(series, ddof=1, as_percentage=False)`, testando ambos os modos.
4. **Meia-Vida quando $\gamma \ge 0$ e Estatísticas Auxiliares Indisponíveis em `pair_stats`**:
   - Quando o resíduo não reverte à média ($\gamma \ge 0$), $-\ln(2)/\gamma$ seria negativo ou divisão por zero.
   - *Solução*: Na função isolada `half_life(residuals)`, levantar `DegenerateSeriesError` se $\gamma \ge 0$. No agregador `pair_stats`, quando uma estimativa derivada/auxiliar (`half_life`, `fisher_test` em $|\rho|=1$, `beta_rotation` / `cv_beta` quando $N < W$, ou janelas específicas de `historical_volatility`) não pode ser calculada validamente, seu valor retorna `None` (nunca `0.0`), sua entrada em `validity` recebe `False`, `is_valid` recebe `False`, e a causa exata é registrada em `warnings`.
5. **Compatibilidade de Índices (`pd.Series`)**:
   - Quando `prices_a` e `prices_b` são objetos indexados (`pd.Series`), `validate_paired_series` verifica igualdade estrita de índices (`series_a.index.equals(series_b.index)`), ausência de `NaN`/`NaT` e ausência de duplicatas no índice, levantando `InvalidSeriesError` sem alinhar ou descartar observações silenciosamente.
6. **Valores Críticos de Engle-Granger na Regressão com Tendência Temporal (`with_time=True`)**:
   - O Manual (p. 41 e p. 55) utiliza a tabela fixa `{-4,32 (99%), -3,67 (95%), -3,28 (90%)}` (derivada para regressão de cointegração com 2 variáveis). Na literatura econométrica (MacKinnon 1991/2010), a inclusão de tendência determinística linear ($T \cdot t$) desloca a distribuição assintótica de $\tau$ mais para a esquerda. Por fidelidade ao Manual, `classify_cointegration_level` utiliza os valores documentados por padrão, mantendo a estatística contínua `dickey_fuller` ($\tau$) separada da classificação `cointegration_level`.

---

## 5. Contratos de Entrada, Saída e Casos Extremos

| Função | Entradas | Saída | Casos Extremos / Exceções |
|---|---|---|---|
| `linear_regression(y, x, with_time=False)` | `y`, `x` (1D array-like), `with_time: bool` | `dict` com `alpha`, `beta`, `t_coef`, `fitted_values`, `residuals`, `se_alpha`, `se_beta`, `se_time`, `std_residual`, `r_squared`, `n_obs`, `k`, `df` | `InvalidSeriesError` (NaN/Inf, tamanhos diferentes, dim $\ne 1$); `InsufficientDataError` ($N \le k + 1$); `DegenerateSeriesError` ($\text{Var}(x) = 0$ ou matriz singular) |
| `beta_confidence_interval(beta, se, n, k=1, conf=0.99, ...)` | `beta: float`, `se: float`, `n: int`, `k: int`, `conf: float` | `tuple[float, float]` (`ic_min`, `ic_max`) | `InsufficientDataError` ($n - k - 1 \le 0$); `InvalidSeriesError` ($se < 0$, $\text{conf} \notin (0,1)$, NaN) |
| `coefficient_t_test(coef, se, n, k=1, tail="right", conf=0.99, ...)` | `coef: float`, `se: float`, `n: int`, `k: int`, `tail: str` | `dict` com `t_stat`, `p_value`, `critical_value`, `is_significant`, `confidence_level` (`99`, `95`, `0`) | `DegenerateSeriesError` ($se \le 0$); `InsufficientDataError` ($n - k - 1 \le 0$); ` ValueError` (`tail` inválido) |
| `dickey_fuller(residuals)` | `residuals` (1D array-like) | `dict` com `tau`, `gamma`, `se_gamma`, `alpha_df`, `cointegration_level` (`99`, `95`, `90`, `0`), `critical_values` | `InvalidSeriesError` (NaN/Inf); `InsufficientDataError` ($N < 4$); `DegenerateSeriesError` (resíduo constante / $EP(\gamma) = 0$) |
| `pearson_correlation(series_a, series_b, use_log_returns=False)` | `series_a`, `series_b` (1D array-like) | `float` ($\rho \in [-1, 1]$) | `InvalidSeriesError` (NaN/Inf, $\le 0$ com log-retornos); `InsufficientDataError`; `DegenerateSeriesError` (variância zero) |
| `fisher_test(corr, n, conf=0.99, ...)` | `corr: float`, `n: int`, `conf: float` | `tuple[float, float]` (`fisher_min`, `fisher_max`) | `InsufficientDataError` ($n \le 3$); `DegenerateSeriesError` ($|\text{corr}| \ge 1$); `InvalidSeriesError` ($|\text{corr}| > 1$ ou NaN) |
| `z_score(residuals)` | `residuals` (1D array-like) | `np.ndarray` (série padronizada $Z_t$) | `InvalidSeriesError` (NaN/Inf); `InsufficientDataError` ($N < 2$); `DegenerateSeriesError` ($\sigma_e = 0$) |
| `half_life(residuals, method="discrete", allow_non_stationary=False)` | `residuals` (1D array-like) | `float` (meia-vida em períodos) | `InvalidSeriesError` (NaN/Inf); `InsufficientDataError` ($N < 4$); `DegenerateSeriesError` ($\gamma \ge 0$ quando `allow_non_stationary=False`) |
| `beta_rotation(prices_a, prices_b, window, with_time=False)` | `prices_a`, `prices_b`, `window: int` | `np.ndarray` (série de $\beta_t$ móvel) | `InvalidSeriesError`; `InsufficientDataError` ($N < \text{window}$ ou $\text{window} \le k + 1$); `DegenerateSeriesError` |
| `coefficient_of_variation(series, ddof=1, as_percentage=False)` | `series` (1D array-like) | `float` ($\sigma / |\mu|$) | `InvalidSeriesError` (NaN/Inf); `InsufficientDataError` ($N < 2$); `DegenerateSeriesError` ($\mu = 0$) |
| `historical_volatility(series, window=(10, 21, 42, 63), scale="annualized")` | `series` (1D array-like de valores $> 0$), `window: int \| Iterable[int]` | `float` ou `dict[int, float]` | `InvalidSeriesError` (valores $\le 0$, NaN/Inf); `InsufficientDataError` ($N < w + 1$) |
| `pair_stats(prices_a, prices_b, n_periods, ...)` | `prices_a`, `prices_b`, `n_periods: int` | `dict` consolidado com todas as chaves exigidas pelo Roteiro | Propaga validações de entrada explicitamente |

---

## 6. Critérios Verificáveis de Aceitação
1. Todas as funções validam entradas e levantam exceções específicas (`InvalidSeriesError`, `InsufficientDataError`, `DegenerateSeriesError`) sem produzir resultados numéricos falsos ou silenciosos.
2. `linear_regression` reproduz com erro relativo $< 10^{-10}$ os estimadores de `statsmodels.api.OLS` tanto na regressão simples quanto na múltipla com tendência temporal $t = 1 \dots N$.
3. Os exemplos numéricos autocontidos do Manual (p. 22, 32, 34, 35, 36, 37, 42, 52, 65) são reproduzidos exatamente nos testes automatizados.
4. Séries sintéticas controladas validam:
   - Pares cointegrados ($\tau < -4,32 \implies 99\%$, meia-vida finita próxima ao valor teórico simulado);
   - Pares não cointegrados / passeios aleatórios independentes ($\tau \ge -3,28 \implies 0\%$);
   - Séries constantes, insuficientes e com `NaN`/`Inf` (levantamento de exceções apropriadas).
5. `pair_stats(prices_a, prices_b, n_periods)` retorna todas as chaves exigidas pelo contrato da Parte 2 do Roteiro do TCC.
