# Long & Short Analyzer

Sistema de análise quantitativa e econométrica para operações **Long & Short por Cointegração**, fundamentado na metodologia de Engle-Granger. O projeto contempla um núcleo estatístico puro (`statistical_engine`) e um laboratório web interativo em Django (`engine_lab`) para simulação, inspeção e validação matemática com séries sintéticas controladas.

---

## 1. Estrutura do Projeto

O repositório está organizado com separação estrita entre o motor matemático, a camada de aplicação web e a suíte de testes:

```text
long-short_analyzer/
├── config/                         # Configuração principal do projeto Django
│   ├── settings.py                 # Configurações de apps, templates, static e locale (pt-br)
│   ├── urls.py                     # Roteamento raiz apontando para o Engine Lab
│   ├── asgi.py / wsgi.py           # Pontos de entrada ASGI/WSGI
├── statistical_engine/             # Núcleo Estatístico e Econométrico (100% puro, sem dependência de Django)
│   ├── validation.py               # Validação estrita de séries 1D, pares indexados e escalares
│   ├── regression.py               # Regressão linear OLS simples e múltipla (com tendência temporal)
│   ├── inference.py                # IC de Beta, testes t de Student, correlação de Pearson e teste de Fisher
│   ├── stationarity.py             # Teste de Dickey-Fuller (Engle-Granger), Z-Score e Meia-Vida (Ornstein-Uhlenbeck)
│   ├── stability.py                # Beta Rotation (janela móvel), Coeficiente de Variação (CV) e Volatilidade Histórica
│   ├── aggregator.py               # Função central pair_stats() que consolida todas as métricas do par
│   ├── exceptions.py               # Exceções de domínio (InvalidSeriesError, InsufficientDataError, DegenerateSeriesError)
├── engine_lab/                     # Aplicação Web (Protótipo Inicial de Dashboard — Somente Leitura)
│   ├── services.py                 # Geração dummy determinística (seed=42) e integração com statistical_engine
│   ├── views.py                    # View Django somente leitura (dashboard_view)
│   ├── urls.py                     # Rotas da interface (/, /dashboard/, /lab/)
│   ├── templates/engine_lab/       # Template HTML do Dashboard (dashboard.html)
│   └── static/engine_lab/          # Estilos CSS e renderização de gráficos em Canvas HTML5
├── docs/specs/                     # Especificações técnicas detalhadas do sistema
│   ├── statistical_engine_spec.md  # Fórmulas, contratos e regras de negócio do núcleo estatístico
│   └── dashboard_prototype_spec.md # Especificação do protótipo inicial de Dashboard com dados dummy
├── tests/                          # Suíte de testes automatizados (pytest)
│   ├── test_validation.py          # Testes de validação de entradas e exceções
│   ├── test_regression.py          # Testes de regressão OLS vs. statsmodels
│   ├── test_inference.py           # Testes de inferência, IC, teste t e Fisher
│   ├── test_stationarity.py        # Testes de Dickey-Fuller, Z-Score e meia-vida
│   ├── test_stability.py           # Testes de Beta Rotation, CV e volatilidade histórica
│   ├── test_pair_stats.py          # Testes integrados do agregador pair_stats()
│   ├── test_dashboard_prototype.py # Testes do serviço dummy e renderização somente leitura do Dashboard
│   └── test_existing_long_e_short.py # Testes de regressão do módulo legado
├── long_e_short.py                 # Módulo inicial de referência para regra de bolso de spread
├── manage.py                       # Utilitário de linha de comando do Django
└── pyproject.toml                  # Definição do projeto, dependências e configuração do pytest
```

---

## 2. Como Iniciar o Sistema

### Pré-requisitos
- **Python 3.11+**
- Pip ou gerenciador de ambientes virtuais (`venv`, `uv`, etc.)

### Passo 1: Criar e ativar o ambiente virtual

No Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

No Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Passo 2: Instalar as dependências

Instale o projeto em modo editável (que instalará `django`, `numpy`, `pandas`, `scipy` e `statsmodels`):
```bash
pip install -e .
```
*(Opcional)* Para executar a suíte de testes, instale também o `pytest`:
```bash
pip install pytest
```

### Passo 3: Iniciar o servidor web (Engine Lab)

Execute o servidor de desenvolvimento do Django:
```bash
python manage.py runserver
```

Acesse no navegador:
- **URL Principal**: `http://127.0.0.1:8000/` (ou `http://127.0.0.1:8000/lab/`)

> **Nota**: O **Engine Lab** opera 100% em memória com séries sintéticas geradas sob demanda; não é necessário rodar migrações de banco de dados nem configurar credenciais ou APIs externas para utilizá-lo.

### Passo 4: Executar os testes automatizados

Para validar o núcleo estatístico e a aplicação web:
```bash
pytest
```

---

## 3. Informações Importantes do Sistema

### 3.1. Convenções Matemáticas e Operacionais
- **Ativo Dependente vs. Independente**:
  - **Ativo A (`prices_a` / $Y_t$)**: Variável **dependente** (eixo $Y$ da regressão). Comanda o sinal da operação.
  - **Ativo B (`prices_b` / $X_t$)**: Variável **independente** (eixo $X$ da regressão). Seu coeficiente angular $\beta$ define a proporção (*hedge ratio*) da operação ($Q_B = \beta \cdot Q_A$).
- **Ordenação Cronológica**: Todas as séries devem estar ordenadas do período **mais antigo** ($t = 1$) para o **mais recente** ($t = N$). O Z-Score atual (`z_score_current`) refere-se sempre à última observação da janela ($Z_N$).
- **Modelos de Regressão**:
  - **Simples (`with_time=False`)**: $Y_t = \alpha + \beta X_t + e_t$ ($gl = N - 2$).
  - **Múltipla com Tendência (`with_time=True`)**: $Y_t = \alpha + \beta X_t + T \cdot t + e_t$ ($gl = N - 3$).
- **Testes de Cointegração (Engle-Granger / Dickey-Fuller)**:
  - Avalia a estacionariedade dos resíduos via $\Delta e_t = \alpha_{df} + \gamma e_{t-1} + u_t$.
  - Níveis críticos de confiança para a estatística $\tau$:
    - $\tau < -4,32 \rightarrow$ Cointegrado a **99%**
    - $-4,32 \le \tau < -3,67 \rightarrow$ Cointegrado a **95%**
    - $-3,67 \le \tau < -3,28 \rightarrow$ Cointegrado a **90%**
    - $\tau \ge -3,28 \rightarrow$ **Não cointegrado (0%)**
- **Meia-Vida (Ornstein-Uhlenbeck)**:
  - Calculada como $\text{HL} = -\ln(2)/\gamma$ quando $\gamma < 0$. Se $\gamma \ge 0$, o resíduo não apresenta reversão à média e a meia-vida é tratada como estatisticamente indefinida (`None`).

### 3.2. Arquitetura e Garantias de Confiabilidade
1. **Isolamento do Núcleo Estatístico (`statistical_engine`)**:
   - Nenhuma regra ou fórmula estatística é calculada no frontend JavaScript ou nas views do Django. Todo número exibido provém diretamente de `statistical_engine.pair_stats()`.
2. **Validação Estrita e Sem Falhas Silenciosas**:
   - Séries com tamanhos divergentes, valores `NaN`/`Inf`, variância nula ou dados insuficientes não geram zeros falsos: a engine levanta exceções específicas (`InvalidSeriesError`, `InsufficientDataError`, `DegenerateSeriesError`).
   - Quando `pd.Series` são fornecidas, exige-se igualdade estrita de índices sem alinhamento ou descarte silencioso de observações.
3. **Estados Explícitos no Engine Lab**:
   - Cada métrica exibida na interface possui um estado claro:
     - `COMPLETED` (*Cálculo concluído*)
     - `UNAVAILABLE` (*Cálculo indisponível* para a configuração/janela atual)
     - `UNDEFINED` (*Resultado estatisticamente indefinido*, ex.: meia-vida com $\gamma \ge 0$)
     - `INVALID_INPUT` (*Entrada inválida* capturada pelas validações de domínio)
     - `EXECUTION_ERROR` (*Falha inesperada da execução*)

### 3.3. Uso Programático da `statistical_engine`

Você pode utilizar a função agregadora `pair_stats` diretamente em scripts Python:

```python
import numpy as np
from statistical_engine import pair_stats

rng = np.random.default_rng(42)
x = np.cumsum(rng.normal(0, 1, 250)) + 100.0
e = np.zeros(250)
for t in range(1, 250):
    e[t] = 0.8 * e[t - 1] + rng.normal(0, 0.5)
y = 10.0 + 1.5 * x + e

resultado = pair_stats(
    prices_a=y,
    prices_b=x,
    n_periods=250,
    with_time=False,
    rolling_window=60,
)

print("Beta estimado:", resultado["beta"])
print("Tau (Dickey-Fuller):", resultado["dickey_fuller"])
print("Nível de Cointegração (%):", resultado["cointegration_level"])
print("Z-Score Atual:", resultado["z_score_current"])
print("Meia-Vida (períodos):", resultado["half_life"])
```

### 3.4. Documentação de Especificação
Para detalhes aprofundados sobre as fórmulas matemáticas, tratamento de ambiguidades do manual de referência e contratos de cada função, consulte:
- [`docs/specs/statistical_engine_spec.md`](docs/specs/statistical_engine_spec.md)
- [`docs/specs/dashboard_prototype_spec.md`](docs/specs/dashboard_prototype_spec.md)
