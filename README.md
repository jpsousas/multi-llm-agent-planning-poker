# MASPE — Planning Poker Multiagente com LLMs para Estimativa de Esforço

Artefato do Trabalho de Conclusão de Curso (TCC) do Instituto de Computação da
**Universidade Federal de Alagoas (UFAL)**, desenvolvido segundo o paradigma de
**Design Science Research (DSR)**.

Um comitê de **3 agentes** executa uma sessão de *Planning Poker* para estimar issues em
*Story Points* (Fibonacci). O comitê é orquestrado com **LangGraph** e debate em até 3 rodadas
com justificativas *Chain-of-Thought* (CoT). Os 3 agentes usam o **mesmo LLM local**, servido
por um endpoint compatível com a API da OpenAI (Ollama, LM Studio ou vLLM), e recebem como
contexto as issues históricas mais parecidas, recuperadas por **RAG semântico**.

---

## Desenho experimental

**Dataset:** SIS-ASTROS, exportação de issues do GitLab em CSV. O dataset é confidencial e
não faz parte do repositório.

- **Ground truth:** coluna `Time Estimate`, convertida para Story Points com 1 SP = 1 h.
  Só entram issues cujo valor está na escala `1, 2, 3, 5, 8, 13, 21, 34, 55`.
- **Split temporal 70/30** por `Created At`. Os 70% mais antigos formam o histórico, que é a
  base do RAG; os 30% mais recentes formam o teste. O manifesto do split fica em
  `results/split_sis_astros.json`.

**Fatorial 2 × 2:** especialização × protocolo de revelação.

| Condição | Personas | Protocolo | Papel na análise |
| :--- | :--- | :--- | :--- |
| `simultaneo_especializado` | Frontend, Backend, QA | Simultâneo (revelação cega) | Tratamento principal (H1, H2, H4, H5, H6) |
| `sequencial_especializado` | Frontend → Backend → QA | Sequencial | Ancoragem (H4); a âncora é o lance do Frontend, na posição 1 |
| `simultaneo_neutro` | 3 agentes sem persona | Simultâneo | Controle da especialização (H2) |
| `sequencial_neutro` | 3 agentes sem persona | Sequencial | Controle cruzado |

- **Papéis fixos:** não há rotação de personas entre issues. No sequencial, a ordem de fala é
  a mesma em todas as rodadas, e quem fala depois vê os lances e as justificativas de quem
  já falou.
- **Mesmo modelo nos 3 agentes:** o efeito medido vem da persona e do protocolo, não de
  modelos diferentes.
- **Regra de parada** ([src/consensus.py](src/consensus.py)):
  1. Unanimidade em qualquer rodada: encerra com o valor unânime.
  2. Estagnação na R2, quando cada agente repete o lance da R1: encerra com a mediana da R2.
  3. Teto na R3 sem unanimidade: encerra com a mediana da R3.

**RAG semântico** ([src/retrieval.py](src/retrieval.py)):
- Embeddings `paraphrase-multilingual-MiniLM-L12-v2`, com cache `.npy` em `results/cache/`.
- Recupera as *k* = 3 issues do histórico mais parecidas com a issue estimada.
- Exclui a própria issue e respeita a causalidade temporal: só entram issues criadas antes dela.
- Cada exemplo mostra a **Estimativa Planejada** (SP) e o **Esforço Real Gasto**, este em
  horas, vindo da coluna `Time Spent`.

### Questões de pesquisa e procedimento estatístico

| RQ / H | Comparação | Métricas e testes |
| :--- | :--- | :--- |
| RQ1 / H1 | Consenso vs. lance individual da R1 de cada papel e vs. mediana dos lances da R1 | MAE, MdAE, PRED(25), Spearman; Wilcoxon + Â12; correção de Holm |
| RQ2.1 / H2 | Especializado vs. neutro, nas mesmas issues | MAE; Wilcoxon + Â12 |
| RQ2.2 / H3 | Consenso vs. SBERT+LightGBM (Rossi, 2026) | MAE, Spearman; teste pareado com `--baseline-rossi` |
| RQ3 / H4 | Simultâneo vs. sequencial, com a âncora fixa na R1 | Proporção de revisões em direção à âncora, magnitude Σ\|ΔSP\|, MAE e trajetória do deslocamento |
| RQ4 / H5 | Dinâmica da convergência | σ por rodada, transições R1→R2 e R2→R3, motivos de encerramento |
| RQ4 / H6 | Análise qualitativa das justificativas CoT | Amostra de 10 sessões; Kappa de Cohen |

Convenções:
- **Wilcoxon:** pareado, bicaudal, `zero_method="pratt"`, α = 0,05.
- **Â12 de Vargha-Delaney:** calculado sobre todos os pares. Como a variável é o erro
  absoluto, Â12 < 0,50 indica que A erra menos que B.
- **PRED(25):** nesta escala, equivale à taxa de acerto exato.

---

## Como baixar, configurar e rodar

### 1. Pré-requisitos

- **Python 3.12 ou mais recente.** O projeto foi testado em 3.13.
- **Git.**
- **Um servidor de LLM local** compatível com a API da OpenAI:
  - [Ollama](https://ollama.com): URL `http://localhost:11434/v1`;
  - [LM Studio](https://lmstudio.ai): URL `http://localhost:1234/v1`;
  - ou vLLM.
- **Internet na primeira execução**, para baixar o modelo de embeddings (cerca de 470 MB).

### 2. Baixar o código e instalar as dependências

```powershell
git clone https://github.com/jpsousas/multi-llm-agent-planning-poker.git
cd multi-llm-agent-planning-poker

python -m venv .venv
.\.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

No Windows, para os acentos aparecerem corretamente no terminal, rode:

```powershell
$env:PYTHONIOENCODING="utf-8"
```

### 3. Colocar o dataset

1. Limpe o CSV exportado do GitLab:
   ```powershell
   python src/limpar.py caminho/do/export_gitlab.csv --out data/sis_astros/dataset_sis_astros.csv
   ```
2. O caminho padrão do CSV é definido em [src/config.py](src/config.py), na constante
   `CAMINHO_SIS_ASTROS`. Para usar outro arquivo, passe `--arquivo <caminho>` ao runner.
3. Cabeçalhos esperados: `IID, Title, State, Author, Assignee, Time Estimate, Time Spent,
   Created At, Closed At, Milestone, Labels, Description, URL`. Os obrigatórios são `IID`,
   `Title`, `Time Estimate` e `Created At`. Mais detalhes em
   [data/sis_astros/LEIA-ME.md](data/sis_astros/LEIA-ME.md).
4. Confira a leitura e o split:
   ```powershell
   python -m src.sis_astros
   ```

### 4. Configurar o modelo local

Baixe o modelo no servidor. Exemplo com o Ollama:

```powershell
ollama pull hf.co/Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M
```

O modelo e a URL são passados na linha de comando, com `--modelo` e `--url`. Como
alternativa, defina valores padrão num arquivo `.env`, que não é versionado:

```
LOCAL_LLM_URL=http://localhost:11434/v1
LOCAL_LLM_MODEL=qwen2.5:14b
```

Nas rodadas 2 e 3, o prompt junta o papel, as issues parecidas e as justificativas dos
colegas. Aumente o contexto do servidor para que nada seja cortado:
- **Ollama:** `OLLAMA_CONTEXT_LENGTH=8192`;
- **LM Studio:** "Context Length" ao carregar o modelo.

### 5. Rodar os testes (sem LLM e sem dataset)

```powershell
python -m unittest discover -s tests -t . -v
```

Os testes cobrem o parser do CSV, o split, os embeddings e o RAG, a regra de parada, o
isolamento dos protocolos (o que cada agente vê) e as estatísticas. O teste de integração
com o CSV real é pulado se o arquivo não existir.

### 6. Executar as sessões

```powershell
# Teste rápido: 1 condição, 3 issues
python -m src.runner --modelo ollama:hf.co/Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M --url http://localhost:11434/v1 --rag --limite 3

# As 4 condições
python -m src.runner --modelo <modelo> --url <url> --rag --condicao todas

# Uma condição específica
python -m src.runner --modelo <modelo> --url <url> --rag --condicao sequencial_especializado
```

| Opção | Descrição |
| :--- | :--- |
| `--modelo` | Nome do modelo no servidor. Aceita os prefixos `ollama:`, `lmstudio:`, `vllm:` e `local:` |
| `--url` | URL do endpoint compatível com a OpenAI |
| `--rag` / `--k-rag N` | Ativa o RAG semântico e define quantas issues recuperar (padrão: 3). Sem `--rag`, usa few-shot estratificado |
| `--condicao` | Uma das 4 condições, ou `todas` |
| `--limite N` | Processa no máximo N issues |
| `--temperatura`, `--seed` | Padrão: 0.0 e 42 |
| `--arquivo` | Outro CSV, em vez do definido em `config.py` |

A execução é **retomável**: as issues que já estão no `.jsonl` da condição são puladas.
Para refazer do zero, apague o arquivo correspondente em `results/raw/`.

### 7. Análise estatística

```powershell
# Relatório completo (JSON + tabela LaTeX)
python -m src.analysis --modelo <modelo> --contexto rag

# Exportar a amostra de 10 sessões para a codificação do H6
python -m src.analysis --modelo <modelo> --contexto rag --amostra-h6 results/h6_amostra.csv

# Kappa de Cohen entre dois avaliadores
python -m src.analysis --kappa avaliador_a.csv avaliador_b.csv

# H3: comparação com o baseline de Rossi (CSV com issue_id e predito)
python -m src.analysis --modelo <modelo> --contexto rag --baseline-rossi baseline.csv
```

O `--modelo` da análise precisa ser **exatamente** o mesmo texto usado no runner.

---

## Saídas

| Caminho | Conteúdo |
| :--- | :--- |
| `results/raw/SIS-ASTROS__<condicao>__<modelo>__<contexto>.jsonl` | Um registro por issue: texto da issue, issues de contexto com SP e horas reais, lances, justificativas, contexto recebido por agente, rodadas, motivo de parada, latência e tokens |
| `results/raw/...__falhas.jsonl` | Issues em que o modelo não devolveu JSON válido, com a resposta bruta |
| `results/transcripts/SIS-ASTROS/<condicao>/<issue>.md` | Transcrição legível da sessão. Não leva o nome do modelo: copie a pasta antes de trocar de modelo |
| `results/split_sis_astros.json` | Manifesto do split temporal |
| `results/analysis/relatorio__<modelo>__<contexto>.json` | Resultados de H1 a H5 |
| `results/latex/` | Tabelas para o TCC |

`results/raw/*.jsonl`, as transcrições, o cache e o CSV estão no `.gitignore`, porque contêm
texto do dataset confidencial.

---

## Estrutura do código

| Arquivo | Responsabilidade |
| :--- | :--- |
| [src/config.py](src/config.py) | Caminho do CSV, agentes, 4 condições, limite da descrição no prompt |
| [src/limpar.py](src/limpar.py) | Limpeza do CSV exportado do GitLab |
| [src/sis_astros.py](src/sis_astros.py) | Leitura do CSV, conversão de horas, split temporal e montagem do dataset |
| [src/retrieval.py](src/retrieval.py) | Embeddings, cache e busca semântica (RAG) |
| [src/prompts.py](src/prompts.py) | System prompt com personas e prompt da issue, do histórico e do debate |
| [src/agents.py](src/agents.py) | Chamada ao LLM, extração tolerante do JSON (`<think>`, vírgulas sobrando) e ajuste à escala |
| [src/consensus.py](src/consensus.py) | Mediana, σ, dispersão em degraus e regra de parada |
| [src/graph.py](src/graph.py) | `StateGraph`: fan-out paralelo (simultâneo), cadeia (sequencial) e laço de até 3 rodadas |
| [src/state.py](src/state.py) | Estado tipado do grafo |
| [src/runner.py](src/runner.py) | CLI de execução, gravação dos JSONL e das transcrições |
| [src/analysis.py](src/analysis.py) | Estatística de H1 a H6 e exportação LaTeX |
| [tests/](tests/) | Testes `unittest` |
