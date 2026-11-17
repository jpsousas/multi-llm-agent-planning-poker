# MASPE — Planning Poker Multiagente com LLMs para Estimativa de Esforço

Artefato do Trabalho de Conclusão de Curso (TCC) de Sistemas de Informação da
**Universidade Federal de Santa Maria (UFSM)**, desenvolvido segundo o paradigma de
**Design Science Research (DSR)**.

Um comitê de **3 agentes** executa uma sessão de *Planning Poker* para estimar issues em
horas de trabalho, na escala de Fibonacci. O comitê é orquestrado com **LangGraph** e debate em
até 3 rodadas com justificativas *Chain-of-Thought* (CoT). Os 3 agentes usam o **mesmo LLM local**
(**Qwen3-14B**, quantização Q4_K_M), servido por um endpoint compatível com a API da OpenAI
(Ollama, llama.cpp, LM Studio ou vLLM), e recebem como contexto as issues históricas mais parecidas,
recuperadas por **RAG semântico**.

---

## Comando para reproduzir a coleta

Com o servidor de LLM aberto e o modelo baixado (veja [Como baixar, configurar e rodar](#como-baixar-configurar-e-rodar)):

```powershell
python -m src.runner --rag --condicao todas --modelo qwen3:14b --url http://localhost:11434/v1
```

Isso executa as 4 condições sobre o bloco de **teste**. Nenhuma outra configuração é necessária:
o runner confere o contexto do servidor antes de começar e, no Ollama, ajusta-o sozinho
(veja [Contexto do servidor](#contexto-do-servidor-verificação-automática)).

---

## Desenho experimental

**Dataset:** SIS-ASTROS, exportação de issues do GitLab em CSV. O dataset é confidencial e
não faz parte do repositório.

- **Ground truth:** coluna `Time Estimate`, em horas (1 SP = 1 h). Só entram issues cujo valor está
  na escala `1, 2, 3, 5, 8, 13, 21, 34, 55`. Depois da limpeza, são 757 issues.
- **Split temporal 70/30** por `Created At`: os 529 itens mais antigos formam o **histórico**, que é a
  base do RAG, e os 228 mais recentes formam o **teste**. O manifesto do split fica em
  `results/split_sis_astros.json`.
- **Desenvolvimento e teste:** as decisões de configuração são calibradas **só dentro do histórico**
  (veja [Fase de desenvolvimento e fase de teste](#fase-de-desenvolvimento-e-fase-de-teste)).

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
- **Escala em horas:** o prompt pede a estimativa em horas. O lance do modelo é ajustado ao degrau
  mais próximo da escala, e o lance bruto fica registrado (`story_points_bruto`, `fora_da_escala`).

### RAG semântico

([src/retrieval.py](src/retrieval.py); valores fixados em [src/config.py](src/config.py))

- **Modelo de embeddings:** `paraphrase-multilingual-mpnet-base-v2` (768 dimensões), com cache `.npy`
  em `results/cache/`. O texto indexado é o título mais os 400 primeiros caracteres da descrição.
- **k = 3** issues do histórico mais parecidas (similaridade de cosseno) com a issue estimada.
- **Proteção contra vazamento**, nesta ordem:
  1. a própria issue (mesmo `IID`) nunca é retornada;
  2. só entram issues criadas **antes** da issue estimada (causalidade temporal);
  3. issues com título e descrição idênticos aos da estimada são excluídas, mesmo com outro `IID`;
  4. quase-duplicatas, com cosseno **≥ 0,95**, também são excluídas.
- O índice é construído só com a base de exemplos da fase em execução, então o bloco de teste nunca
  é indexado durante o desenvolvimento.
- Cada exemplo mostra a **Estimativa Planejada** (h) e o **Esforço Real Gasto** (h), este vindo da coluna
  `Time Spent` e omitido quando não há registro.
- Os mesmos exemplos vão para os 3 agentes e se repetem em todas as rodadas.

### Contexto do prompt

([src/sis_astros.py](src/sis_astros.py), [src/limpeza_texto.py](src/limpeza_texto.py))

O servidor é usado com **16.384 tokens por requisição**. O prompt segue uma ordem de prioridade:
instruções, depois a issue estimada, depois os exemplos.

- **Instruções:** sempre completas.
- **Issue estimada:** descrição **completa**, sem corte.
- **Exemplos:** cada descrição é truncada a um orçamento dinâmico, no máximo 2.500 tokens, que
  diminui quando a issue estimada é grande, para que `alvo + k × exemplo + reserva` caiba no contexto.
  A reserva de 5.000 tokens cobre as instruções, o debate das rodadas 2 e 3 e a resposta do modelo.
  Um exemplo truncado termina em `[... texto truncado ...]`.
- **Limpeza determinística (opcional, desligada por padrão):** com `LIMPAR_TEXTO=1`, links, imagens e
  trechos longos de código ou log são trocados por marcadores (`[link]`, `[imagem]`, `[código/log]`
  seguido de `[... N linhas omitidas ...]`, `…/arquivo`, `[token-longo]`). O prompt de sistema explica
  esses marcadores nas duas condições, para que o texto das issues seja o único fator que muda.
  A limpeza vale igualmente para a issue estimada e para os exemplos e não altera a busca do RAG.
- Os tokens são contados com o tokenizador do Qwen3. Sem ele, o sistema usa uma estimativa
  conservadora de caracteres por token.

### Contexto do servidor: verificação automática

O Ollama usa por padrão cerca de 4.096 tokens de contexto e **corta o começo do prompt sem avisar**,
justamente onde ficam as instruções. Para evitar estimativas inválidas, o runner confere o servidor
antes de começar ([src/servidor.py](src/servidor.py)):

| Servidor | O que o runner faz |
| :--- | :--- |
| **Ollama** | Lê o contexto do modelo. Se for menor que 16.384, cria uma cópia `maspe-<modelo>-ctx16384` com o contexto correto (não baixa nada, só na primeira vez) e usa essa cópia. Os arquivos de resultado continuam levando o nome que você digitou. |
| **llama.cpp (llama-server)** | Lê o contexto por slot. Se for menor, aborta e mostra o comando para subir o servidor. |
| **LM Studio, vLLM e outros** | Não consegue verificar. Avisa e segue; configure pelo menos 16.384 tokens de contexto. |

Há ainda uma rede de segurança por chamada: se o servidor local processar bem menos tokens de entrada
do que o prompt tem, a chamada falha com a mensagem de contexto truncado e entra em `__falhas.jsonl`,
em vez de gravar uma estimativa inválida.

O runner usa **uma chamada por vez** (`MAX_CONCORRENCIA=1`). Chamadas paralelas dividem a memória do
servidor, podem estourar o tempo limite de 180 s por chamada e tornam a saída menos reprodutível.
No protocolo simultâneo, os 3 agentes continuam votando de forma cega e independente; só a
execução é sequencial.

### Fase de desenvolvimento e fase de teste

- **`--fase dev`:** os alvos são os últimos 30% do histórico (159 issues) e a base de exemplos são os
  70% mais antigos dele (370 issues). O bloco de teste **não é lido** nem como exemplo.
  Use para calibrar o modelo de embeddings, k, o limiar de duplicatas, os orçamentos de tokens, a
  limpeza e o prompt. As saídas vão para `results/dev/` e `results/transcripts_dev/`, que o
  `src/analysis.py` não lê.
- **`--fase teste` (padrão):** os alvos são os 228 itens mais recentes e a base de exemplos é todo o
  histórico. Só deve ser executada depois de congelar a configuração.
- Cada registro grava a **configuração** usada e o **hash do prompt** (`hash_prompt`), para
  comprovar que a coleta final usou exatamente a configuração congelada.

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
  - [Ollama](https://ollama.com) (recomendado): URL `http://localhost:11434/v1`. Use uma versão recente,
    pois a criação automática da cópia com contexto ajustado exige a API de criação de modelos de
    dezembro de 2024 em diante;
  - [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`): URL `http://localhost:8080/v1`;
  - LM Studio (`http://localhost:1234/v1`) ou vLLM.
- **Hardware:** o Qwen3-14B Q4_K_M com 16.384 tokens de contexto ocupa cerca de 12 GB entre modelo e
  cache. Se a GPU tiver menos memória, o servidor passa camadas para a CPU e a execução fica mais lenta.
- **Internet na primeira execução**, para baixar o modelo de embeddings (cerca de 1,1 GB) e o
  tokenizador do Qwen3.

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
ollama pull qwen3:14b
```

O modelo e a URL são passados na linha de comando, com `--modelo` e `--url`. Como alternativa,
defina valores padrão num arquivo `.env`, que não é versionado:

```
LOCAL_LLM_URL=http://localhost:11434/v1
LOCAL_LLM_MODEL=qwen3:14b
```

**Contexto:** no Ollama não é preciso configurar nada, o runner ajusta (veja
[Contexto do servidor](#contexto-do-servidor-verificação-automática)). No llama.cpp, suba o servidor
com contexto suficiente:

```powershell
llama-server -m Qwen3-14B-Q4_K_M.gguf -ngl 99 -c 16384 -np 1 -fa on --port 8080
```

O `-c` é dividido entre os slots (`-np`): com `-np 1`, o slot recebe os 16.384 tokens inteiros.

### 5. Rodar os testes (sem LLM e sem dataset)

```powershell
python -m unittest discover -s tests -t . -v
```

Cada arquivo também roda sozinho, por exemplo `python tests\test_prompts.py`. Os testes cobrem:

| Arquivo | O que verifica |
| :--- | :--- |
| `test_split_parser.py` | Leitura do CSV, filtros, split temporal e fase de desenvolvimento (o teste nunca vira alvo nem exemplo) |
| `test_embeddings.py` | RAG: busca semântica, anti-vazamento, quase-duplicatas (0,95), conteúdo idêntico, cache |
| `test_limpeza_texto.py` | Limpeza determinística e truncamento por tokens |
| `test_orcamento_contexto.py` | Orçamento de tokens do prompt e `prep_meta` |
| `test_prompts.py` | Prompt de sistema, marcadores, prompt do item e lembrete de formato |
| `test_agents.py` | Chamada ao LLM com modelo falso: escala, retry, truncamento e privacidade do prompt |
| `test_servidor.py` | Verificação do servidor (Ollama, llama.cpp) e detecção de truncamento |
| `test_regras_parada.py`, `test_fluxo_consenso.py`, `test_isolamento_protocolos.py` | Regra de parada, nós do grafo e o que cada agente enxerga em cada protocolo |
| `test_runner.py` | Runner com grafo falso: separação dev/teste, registro, retomada, falhas |
| `test_dev_resumo.py` | Resumo da fase de desenvolvimento (só agregados) |
| `test_analysis.py` | Estatísticas e tabelas LaTeX |

`TestEmbeddingsRAG` usa o modelo de embeddings **real** e baixa os pesos na primeira vez, por isso é
o mais lento. O teste de integração com o CSV real é pulado se o arquivo não existir.

### 6. Executar as sessões

```powershell
# Coleta final: 4 condições, bloco de teste
python -m src.runner --rag --condicao todas --modelo qwen3:14b --url http://localhost:11434/v1

# Teste rápido de funcionamento: 1 condição, 3 issues da fase de desenvolvimento
python -m src.runner --fase dev --rag -n 3 --condicao simultaneo_especializado --modelo qwen3:14b --url http://localhost:11434/v1

# Uma condição específica
python -m src.runner --rag --condicao sequencial_especializado --modelo qwen3:14b --url http://localhost:11434/v1
```

| Opção | Descrição |
| :--- | :--- |
| `--modelo` | Nome do modelo no servidor. Aceita os prefixos `ollama:`, `lmstudio:`, `vllm:` e `local:` |
| `--url` | URL do endpoint compatível com a OpenAI |
| `--rag` / `--k-rag N` | Ativa o RAG semântico e define quantas issues recuperar (padrão: 3). Sem `--rag`, usa few-shot estratificado |
| `--limiar` | Cosseno a partir do qual uma issue é considerada quase-duplicata (padrão: 0,95) |
| `--fase` | `teste` (padrão) ou `dev` |
| `--condicao` | Uma das 4 condições, ou `todas` |
| `-n N` / `--limite N` | Processa no máximo N issues (as mesmas N em todas as condições) |
| `--temperatura`, `--seed` | Padrão: 0.0 e 42 |
| `--arquivo` | Outro CSV, em vez do definido em `config.py` |

A execução é **retomável**: as issues que já estão no `.jsonl` da condição são puladas.
Para refazer do zero, ou ao mudar o prompt ou a configuração, apague o arquivo correspondente em
`results/raw/` (ou `results/dev/`). Não misture configurações no mesmo arquivo.

**Variáveis de ambiente** (opcionais; servem para experimentos de desenvolvimento, e a coleta final
usa os padrões do código):

| Variável | Padrão | Efeito |
| :--- | :--- | :--- |
| `LIMPAR_TEXTO` | `0` | `1` liga a limpeza determinística das descrições |
| `ORCAMENTO_TOKENS_EXEMPLO` | `2500` | Teto de tokens da descrição de cada exemplo |
| `ORCAMENTO_TOKENS_ALVO` | `0` (sem teto) | Teto para a issue estimada |
| `CONTEXTO_SERVIDOR_POR_SLOT` | `16384` | Contexto que o pipeline assume e exige do servidor |
| `RESERVA_PROMPT_TOKENS` | `5000` | Reserva para instruções, debate e resposta |
| `MAX_CONCORRENCIA` | `1` | Chamadas simultâneas ao servidor |
| `LLM_JSON_MODE` | `0` | `1` pede saída JSON ao servidor (útil em modelos pequenos) |
| `DEBUG_PROMPT` | `0` | `1` imprime os prompts completos. **Confidencial**: não use em logs compartilhados |
| `TOKENIZADOR` | `qwen3` | `none` usa a estimativa conservadora, sem baixar o tokenizador (os testes usam) |

### 7. Verificar e resumir (sem rodar o modelo)

```powershell
# Os prompts cabem no contexto? (só tokens agregados; não chama o LLM)
python -m scripts.verificar_contexto --rag --contexto 16384

# Resumo agregado da fase de desenvolvimento (MAE, acerto exato, ±1 degrau, baselines, encerramento)
python -m scripts.dev_resumo --contexto 16384

# Comparar duas execuções do desenvolvimento nas issues em comum
python -m scripts.dev_resumo --comparar results\dev\<A>.jsonl results\dev\<B>.jsonl
```

`--contexto` em `dev_resumo` avisa se alguma chamada chegou a 95% do contexto, sinal de truncamento.
Os dois scripts imprimem só números, nunca texto de issue.

### 8. Análise estatística

```powershell
# Relatório completo (JSON + tabelas LaTeX de RQ1 a RQ4 + CSV de erros vs Time Spent)
python -m src.analysis --modelo qwen3:14b --contexto rag

# Exportar a amostra de 10 sessões para a codificação do H6
# (o Kappa de Cohen é calculado manualmente; concordancia_h6 está comentada em src/analysis.py)
python -m src.analysis --modelo qwen3:14b --contexto rag --amostra-h6 results/h6_amostra.csv

# H3: comparação com o baseline de Rossi (CSV com issue_id e predito)
python -m src.analysis --modelo qwen3:14b --contexto rag --baseline-rossi baseline.csv
```

O `--modelo` da análise precisa ser **exatamente** o mesmo texto usado no runner.

---

## Saídas

| Caminho | Conteúdo |
| :--- | :--- |
| `results/raw/SIS-ASTROS__<condicao>__<modelo>__<contexto>.jsonl` | Um registro por issue (fase de teste): texto da issue, issues de contexto com estimativa e horas reais, lances, justificativas, contexto recebido por agente, rodadas, motivo de parada, latência e tokens, além de `fase`, `hash_prompt`, `configuracao` (limpeza, orçamentos, modelo de embeddings, limiar, servidor verificado), `prep_meta` e `rag_meta` |
| `results/dev/...__dev.jsonl` | Mesmo formato, para a fase de desenvolvimento (com o sufixo `__limpo` quando `LIMPAR_TEXTO=1`) |
| `results/.../...__falhas.jsonl` | Issues em que a chamada falhou (JSON inválido, contexto truncado, erro do servidor), com a mensagem de erro |
| `results/transcripts/SIS-ASTROS/<condicao>/<issue>.md` | Transcrição legível da sessão. Não leva o nome do modelo: copie a pasta antes de trocar de modelo. A fase de desenvolvimento usa `results/transcripts_dev/` |
| `results/split_sis_astros.json` | Manifesto do split temporal |
| `results/analysis/relatorio__<modelo>__<contexto>.json` | Resultados de H1 a H5 |
| `results/latex/` | Tabelas para o TCC |

`results/raw/`, `results/dev/`, `results/transcripts/`, `results/transcripts_dev/`, `results/cache/` e o
CSV devem estar no `.gitignore`, porque contêm texto do dataset confidencial.

---

## Solução de problemas

| Sintoma | Causa provável e solução |
| :--- | :--- |
| A resposta do modelo comenta a issue em vez de estimar, ou aparece `Não foi possível extrair estimativa válida` | Prompt cortado por contexto pequeno. Confira com `ollama ps` (coluna CONTEXT deve mostrar 16384) e rode o runner de novo |
| `ContextoTruncado` em `__falhas.jsonl` | O servidor processou menos tokens do que o prompt tem. Rode o runner de novo (ele ajusta o Ollama) ou suba o llama-server com `-c 16384 -np 1` |
| `[ERRO DE SERVIDOR]` no começo | Modelo não baixado (`ollama pull <modelo>`) ou llama-server com contexto menor que 16.384 |
| `ollama ps` mostra divisão CPU/GPU | O modelo não cabe inteiro na GPU. Funciona, mas fica mais lento |
| A análise não acha resultados | `--modelo` da análise diferente do usado no runner |
| `coroutine 'arun_with_retry' was never awaited` | Aviso do LangGraph quando um agente falha e os outros são cancelados. Some quando a falha some |

---

## Estrutura do código

| Arquivo | Responsabilidade |
| :--- | :--- |
| [src/config.py](src/config.py) | Caminho do CSV, agentes, 4 condições e as decisões de recuperação (modelo de embeddings, k, limiar) |
| [src/limpar.py](src/limpar.py) | Limpeza do CSV exportado do GitLab |
| [src/sis_astros.py](src/sis_astros.py) | Leitura do CSV, conversão de horas, split temporal, orçamento de tokens e montagem do dataset (fases dev e teste) |
| [src/limpeza_texto.py](src/limpeza_texto.py) | Contagem de tokens, limpeza determinística e truncamento por orçamento |
| [src/retrieval.py](src/retrieval.py) | Embeddings, cache e busca semântica (RAG) com diagnóstico de vazamento |
| [src/prompts.py](src/prompts.py) | System prompt com personas e prompt da issue, do histórico e do debate |
| [src/agents.py](src/agents.py) | Chamada ao LLM, extração tolerante do JSON (`<think>`, vírgulas sobrando), ajuste à escala, retry e detecção de truncamento |
| [src/servidor.py](src/servidor.py) | Verificação e ajuste automático do contexto do servidor (Ollama, llama.cpp) |
| [src/consensus.py](src/consensus.py) | Mediana, σ, dispersão em degraus e regra de parada |
| [src/graph.py](src/graph.py) | `StateGraph`: fan-out (simultâneo), cadeia (sequencial) e laço de até 3 rodadas |
| [src/state.py](src/state.py) | Estado tipado do grafo |
| [src/runner.py](src/runner.py) | CLI de execução, gravação dos JSONL e das transcrições |
| [src/analysis.py](src/analysis.py) | Estatística de H1 a H6 e exportação LaTeX |
| [scripts/verificar_contexto.py](scripts/verificar_contexto.py) | Confere, sem o LLM, se os prompts cabem no contexto |
| [scripts/dev_resumo.py](scripts/dev_resumo.py) | Resumo agregado da fase de desenvolvimento |
| [tests/](tests/) | Testes `unittest` |
