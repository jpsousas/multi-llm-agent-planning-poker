import operator
from typing import Annotated, TypedDict, Literal

class Estimativa(TypedDict):
    agente: str     # "backend" | "frontend" | "qa" 
    papel: str      # papel textual atribuido
    modelo: str     # identificador exato do modelo
    story_points: float
    justificativa: str
    rodada: int
    tokens_entrada: int
    tokens_saida: int
    latencia_ms: float

class ConsensoState(TypedDict):
    issue_id: str
    titulo: str
    descricao: str
    historicos: list[dict]
    estimativas: Annotated[list[Estimativa], operator.add]
    rodada: int
    convergiu: bool
    consenso: float
    historico_rodadas: Annotated[list[dict], operator.add]