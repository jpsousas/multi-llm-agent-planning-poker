import operator
from typing import Annotated, TypedDict, Optional

class Estimativa(TypedDict):
    agente: str            # Identificador do agente
    papel: str             # "backend" | "frontend" | "qa" | "neutro"
    modelo: str            # Identificador do modelo LLM
    story_points: float    # Valor estimado
    justificativa: str     # Raciocínio Chain-of-Thought (CoT)
    rodada: int            # Rodada em que o lance foi emitido (1, 2 ou 3)
    tokens_entrada: int
    tokens_saida: int
    latencia_ms: float
    data_hora: Optional[str]

class ConsensoState(TypedDict):
    issue_id: str
    titulo: str
    descricao: str
    historicos: Optional[list[dict]]                     # Exemplos Scale-Aware (Few-Shot)
    escala_fibonacci: str                               # Escala usada dependendo do dataset
    protocolo: str                                      # "simultaneo" | "sequencial"
    especializado: bool                                 # True se usa persona, False se neutro
    papeis_agentes: Optional[dict[str, str]]             # Mapeamento dinâmico agente_id -> papel efetivo (rotação de papéis)
    estimativas: Annotated[list[Estimativa], operator.add]
    rodada: int                                         # Rodada corrente (inicia em 1)
    convergiu: bool                                     # True se atingiu consenso na rodada
    motivo_convergencia: Optional[str]                  # Ex: "unanime_r1", "adjacente_r2", "desempate_mediana_r3"
    dispersao_degraus: Optional[int]                    # Distância em degraus Fibonacci entre min e max da rodada
    rodadas_consecutivas_convergidas: Optional[int]     # Campo de retrocompatibilidade
    consenso: float                                     # Valor final acordado
    contexto_rodada_atual: Optional[str]                # Contexto de debate CoT ou lances acumulados
    historico_rodadas: Annotated[list[dict], operator.add]
