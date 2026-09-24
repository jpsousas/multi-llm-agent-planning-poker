from functools import partial
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from src.state import ConsensoState
from src.agents import estimar
from src.consensus import agregar, verificar_convergencia, dispersao
from src.config import Condicao

def _no_estimador(state: ConsensoState, *, agente_id, papel_id, modelo_id, temperatura, deliberar):
    """Um no por agente. Executados em paralelo (fan-out), nenhum
    enxerga a estimativa do outro NESTA rodada: revelacao simultanea."""
    contexto = None
    if deliberar and state["rodada"] > 1:
        ultima = [e for e in state["estimativas"]
        if e["rodada"] == state["rodada"] - 1] 
        contexto = "\n".join(f'{e["papel"]} estimou {e["story_points"]} SP: {e["justificativa"]}' for e in ultima)
    est = estimar(agente_id, papel_id, modelo_id, temperatura, state["titulo"], state["descricao"], state.get("historicos"), contexto, state["rodada"])
    return {"estimativas": [est]}

def _no_agregador(state: ConsensoState, *, metodo, tolerancia, max_rodadas):
    atuais = [e for e in state["estimativas"] if e["rodada"] == state["rodada"]]
    valores = [e["story_points"] for e in atuais]
    consenso = agregar(valores, metodo)
    convergiu = verificar_convergencia(valores, tolerancia)
    registro = {
        "rodada": state["rodada"],
        "valores": valores,
        "consenso": consenso,
        "convergiu": convergiu,
        "dispersao": dispersao(valores),
    }
    return {
        "consenso": consenso,
        "convergiu": convergiu,
        "rodada": state["rodada"] + 1,
        "historico_rodadas": [registro],
    }

def _rotear(state: ConsensoState, *, max_rodadas, deliberacao) -> Literal["estimar", "__end__"]:
    """Aresta condicional: fecha o laco de rodadas.
    Resposta direta ao modo de falha 'Unaware of Termination
    Conditions' catalogado por Cemri et al. (2025)."""
    if not deliberacao:
        return END
    if state["convergiu"]:
        return END
    if state["rodada"] > max_rodadas:
        return END
    return "estimar"

def construir_grafo(cond: Condicao, metodo="mediana", tolerancia=1):
    builder = StateGraph(ConsensoState, start=START, end=END)

    nomes = []
    for agente_id, papel_id, modelo_id in cond.agentes:
        nome = f"est_{agente_id}"
        nomes.append(nome)
        builder.add_node(
            nome,
            partial(_no_estimador, agente_id=agente_id, papel_id=papel_id, modelo_id=modelo_id, temperatura=cond.temperatura, deliberar=cond.deliberacao),
            retry_policy=RetryPolicy(max_attempts=3),
        )
    builder.add_node("agregar", partial(_no_agregador, metodo=metodo, tolerancia=tolerancia, max_rodadas=cond.max_rodadas))

    if cond.protocolo == "simultaneo":
        # todas as arestas saem do start ao mesmo tempo.
        for nome in nomes:
            builder.add_edge(START, nome)
            builder.add_edge(nome, "agregar")
    else:
        # sequencial: encadeado. cada agente ve o que ja foi acumulado
        builder.add_edge(START, nomes[0])
        for anterior, seguinte in zip(nomes, nomes[1:]):
            builder.add_edge(anterior, seguinte)
        builder.add_edge(nomes[-1], "agregar")

    builder.add_conditional_edges("agregar", partial(_rotear, max_rodadas=cond.max_rodadas, deliberacao=cond.deliberacao), {"estimar": nomes[0] if cond.protocolo == "sequencial" else nomes[0], END: END},)
    return builder.compile()


if __name__ == "__main__":
    #meramente testes
    for cond_id, cond in Condicao.__dict__.items():
        if not isinstance(cond, Condicao):
            continue
        print(f"Construindo grafo para {cond_id}...")
        g = construir_grafo(cond)
        print(g)
        