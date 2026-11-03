import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# >>> DATASET SIS-ASTROS — coloque aqui o caminho do CSV (já limpo pelo src/limpar.py) <<<
# Padrão: multi-llm-agent-planning-poker/data/sis_astros/dataset_sis_astros.csv
# Pode ser sobrescrito na linha de comando com --arquivo <caminho>.
# ==============================================================================
RAIZ_PROJETO = Path(__file__).resolve().parent.parent
CAMINHO_SIS_ASTROS = RAIZ_PROJETO / "data" / "sis_astros" / "dataset_sis_astros.csv"

# Cabeçalhos esperados no CSV exportado do GitLab
COLUNAS_SIS_ASTROS = [
    "IID", "Title", "State", "Author", "Assignee", "Time Estimate",
    "Time Spent", "Created At", "Closed At", "Milestone", "Labels",
    "Description", "URL",
]
# Sem estas o experimento não roda (ID, texto, ground truth e data para o split temporal)
COLUNAS_OBRIGATORIAS_SIS_ASTROS = ["IID", "Title", "Time Estimate", "Created At"]

# Limite de caracteres da descrição no PROMPT — o mesmo para a issue estimada e para as
# issues históricas (RAG/few-shot). None = descrição completa. Não afeta os embeddings.
LIMITE_DESCRICAO_PROMPT = 2000

# Modelo Local Base (DeepSeek R1 Distill Llama 70B replicado nos 3 agentes)
# Pode ser customizado via variáveis de ambiente no .env (ex: Ollama, LM Studio, vLLM)
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL")
MODELO_BASE = f"local:{LOCAL_LLM_MODEL}"

# Tupla de agentes com papéis especializados — papéis FIXOS (sem rotação entre itens).
# A ordem da lista é a ordem de fala no protocolo sequencial: Frontend -> Backend -> QA
# (o lance do Frontend, posição 1, é a âncora do H4). No simultâneo a ordem não importa.
# Rigor DSR (H2): Mesmo modelo base constante para isolar o efeito puro da persona.
AGENTES_ESPECIALIZADOS = [
    ("frontend", "frontend", MODELO_BASE),
    ("backend", "backend", MODELO_BASE),
    ("qa", "qa", MODELO_BASE),
]

# Tupla de agentes de controle sem especialização (Papel Neutro)
# Utiliza o MESMO modelo, mesma arquitetura, mas sem persona.
AGENTES_NEUTROS = [
    ("agente_1", "neutro", MODELO_BASE),
    ("agente_2", "neutro", MODELO_BASE),
    ("agente_3", "neutro", MODELO_BASE),
]

# Escalas de Fibonacci por Dataset
ESCALAS_FIBONACCI = {
    "SIS-ASTROS": "1, 2, 3, 5, 8, 13, 21, 34, 55"
}

@dataclass
class Condicao:
    nome: str
    agentes: list[tuple]                 # (id, papel_id, modelo, [api_key_env_var])
    protocolo: str = "simultaneo"        # "simultaneo" (Tratamento A) | "sequencial" (Tratamento B)
    especializado: bool = True           # True: Especializado | False: Neutro (Controle H2)
    deliberacao: bool = True             # Habilita debate Chain-of-Thought (CoT)
    max_rodadas: int = 3                 # Limite de rodadas R_max = 3
    temperatura: float = 0.0             # Determinístico
    seed: int = 42                       # Seed fixa para reprodutibilidade das inferências
    # Modo de contexto (RAG ou few-shot) é escolhido no runner (--rag) e gravado em
    # "modo_contexto" no JSONL. Regra de parada (unanimidade / estagnação R2 / mediana R3)
    # está em src/consensus.py.

# 4 Condições experimentais do MASPE
CONDICOES = {
    # 1. Tratamento Principal (Simultâneo Especializado)
    "simultaneo_especializado": Condicao(
        nome="simultaneo_especializado",
        agentes=AGENTES_ESPECIALIZADOS,
        protocolo="simultaneo",
        especializado=True
    ),

    # 2. Avaliação de Ancoragem (Sequencial Especializado)
    "sequencial_especializado": Condicao(
        nome="sequencial_especializado",
        agentes=AGENTES_ESPECIALIZADOS,
        protocolo="sequencial",
        especializado=True
    ),

    # 3. Controle de Especialização (Simultâneo Neutro)
    "simultaneo_neutro": Condicao(
        nome="simultaneo_neutro",
        agentes=AGENTES_NEUTROS,
        protocolo="simultaneo",
        especializado=False
    ),

    # 4. Controle Cruzado (Sequencial Neutro)
    "sequencial_neutro": Condicao(
        nome="sequencial_neutro",
        agentes=AGENTES_NEUTROS,
        protocolo="sequencial",
        especializado=False
    ),
}

if __name__ == "__main__":
    for cond_id, cond in CONDICOES.items():
        print(f"Condição '{cond_id}': Protocolo={cond.protocolo}, Especializado={cond.especializado}, Modelo={cond.agentes[0][2]}")
