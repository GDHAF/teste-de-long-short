# Especificação Técnica: Protótipo Inicial de Dashboard (Tela de Teste com Dados Dummy)

## 1. Objetivo
Construir uma tela web simples de demonstração e teste em Django (protótipo inicial do Dashboard do sistema Long & Short por Cointegração) que execute o processo matemático real do pacote `statistical_engine` sobre um único par de séries sintéticas determinísticas (*valores dummy*) e apresente os resultados de forma clara, estruturada e auditável.

Conforme restrição explícita de escopo:
- **Sem campos de inserção ou alteração**: A interface é estritamente de leitura (*read-only*), sem formulários (`<form>`), caixas de texto (`<input>`, `<textarea>`) ou seletores (`<select>`).
- **Sem múltiplos cenários**: A tela apresenta uma única situação demonstrativa coesa de um par cointegrado com distorção estatística no último período, servindo como validação visual ponta a ponta do motor matemático e protótipo do Dashboard do projeto.

---

## 2. Arquitetura e Separação de Responsabilidades

1. **Camada Matemática (`statistical_engine/`)**:
   - Permanece 100% independente de Django, banco de dados ou interface web.
   - Realiza todos os cálculos econométricos através de `pair_stats(prices_a, prices_b, n_periods, ...)`.
2. **Camada de Serviço de Demonstração (`engine_lab/services.py`)**:
   - Gera em memória, com semente fixa (`DUMMY_SEED = 42`), um par determinístico de 260 pregões (`BBAS3` como variável dependente $Y$ e `CSAN3` como variável independente $X$).
   - Invoca `pair_stats` para a janela principal de análise ($N = 250$ períodos, `rolling_window = 60`, `conf = 0.99`) e para as 10 janelas padrão do Manual do TCC (`250, 240, 220, 200, 180, 160, 140, 120, 100, 70`), além da verificação inversa $B \times A$.
   - Formata o dicionário de contexto para apresentação no template Django sem recalcular estatísticas fora da engine.
3. **Camada de Apresentação (`engine_lab/views.py` e `engine_lab/templates/engine_lab/dashboard.html`)**:
   - View Django (`dashboard_view`) que responde a requisições `GET` em `/` (e `/dashboard/`, `/lab/`).
   - Renderiza o template HTML com os cards de KPIs, gráficos em Canvas HTML5 (Z-Score dos resíduos, Preços $Y_t \times \hat{Y}_t$ e Beta Rotation), tabela multi-período (10 janelas do Manual), tabela de volatilidade histórica (`10, 21, 42, 63` dias) e amostra auditável das observações dummy.

---

## 3. Entradas (Dados Dummy Determinísticos)

- **Identificação do Par Dummy**:
  - `ticker_a = "BBAS3"` (Ativo A — Variável Dependente $Y_t$)
  - `ticker_b = "CSAN3"` (Ativo B — Variável Independente $X_t$)
- **Processo Gerador Sintético (`generate_dummy_pair_data`)**:
  - Semente fixa: `np.random.default_rng(42)`.
  - Tamanho total da série gerada: `260` dias úteis (permitindo análise na janela máxima de `250` períodos e janelas de volatilidade histórica).
  - Série $X_t$ (`CSAN3`): passeio aleatório com drift suave $X_t = X_{t-1} + \epsilon_{x,t}$, iniciando em $\text{R\$ } 40,00$ e mantida estritamente positiva.
  - Resíduo estacionário $e_t$: processo Ornstein-Uhlenbeck / AR(1) $e_t = \phi e_{t-1} + \eta_t$ com $|\phi| < 1$ ($\phi = 0,72$), apresentando choque controlado na última observação ($t = 260$) para ilustrar um gatilho operacional ($|Z_N| > 2,0$).
  - Série $Y_t$ (`BBAS3`): $Y_t = \alpha_{\text{true}} + \beta_{\text{true}} X_t + e_t$, com $\alpha_{\text{true}} = 4,50$ e $\beta_{\text{true}} = 0,68$.

---

## 4. Saídas (Contrato de Contexto do Dashboard)

A função `build_dummy_dashboard_context()` retorna um dicionário estruturado contendo:
- `meta`: metadados fixos da demonstração (`ticker_a`, `ticker_b`, `n_periods=250`, `total_observations=260`, `seed=42`, `rolling_window=60`, `conf_pct=99`, `model_label`, `equation_display`).
- `primary_stats`: retorno integral de `pair_stats(prices_a, prices_b, n_periods=250, with_time=False, conf=0.99, rolling_window=60)`.
- `signal_summary`: resumo derivado exclusivamente de `primary_stats` (`z_score_current`, `direction_label`, `action_a`, `action_b`, `is_entry_signal`, `financeiro_pct`, `Current_price_a`, `Current_price_b`).
- `multi_period_rows`: lista com os resultados de `pair_stats` para cada uma das 10 janelas do Manual (`250, 240, 220, 200, 180, 160, 140, 120, 100, 70`), incluindo a verificação de cointegração reversa $B \times A$.
- `sample_table_rows`: últimas 10 observações da janela principal contendo `period_index`, `date_label`, `price_a`, `price_b`, `fitted_a`, `residual` e `z_score`.
- `chart_payload_json`: JSON serializado com as séries calculadas pela engine (`dates`, `prices_a`, `prices_b`, `fitted_values`, `residuals`, `z_scores`, `beta_rotation`, `beta_mean`, `beta_upper_2sd`, `beta_lower_2sd`) para plotagem fiel no Canvas.

---

## 5. Casos Extremos e Garantias
- **Determinismo**: Duas execuções consecutivas de `build_dummy_dashboard_context()` devem produzir exatamente os mesmos valores numéricos em todas as métricas.
- **Somente Leitura**: Requisições `POST`, `PUT`, `DELETE` na rota do dashboard devem retornar `405 Method Not Allowed`.
- **Fidelidade Matemática**: Todos os números exibidos no contexto do dashboard devem coincidir exatamente com a saída de `statistical_engine.pair_stats`.

---

## 6. Critérios Verificáveis de Aceitação
1. A requisição `GET /` retorna HTTP `200 OK` e renderiza o protótipo de Dashboard.
2. O HTML renderizado não contém elementos `<form`, `<input`, `<select` ou `<textarea`.
3. O HTML apresenta claramente o par dummy (`BBAS3` e `CSAN3`), os coeficientes $\alpha$ e $\beta$, estatística Dickey-Fuller ($\tau$) e nível de cointegração, Z-Score atual, Meia-Vida, Correlação de Pearson e intervalo de Fisher, Beta Rotation / $\text{CV}(\beta)$, Volatilidade Histórica (`10, 21, 42, 63` dias) e a tabela multi-período das 10 janelas do manual.
4. Os valores presentes no contexto da view são idênticos aos retornados diretamente por `statistical_engine.pair_stats`.
