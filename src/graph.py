from functools import partial
from typing import Literal, Optional
from langgraph.graph import StateGraph, START, END
from src.state import ConsensoState
from src.agents import estimar, obter_modelo
from src.consensus import (
    agregar,
    desvio_padrao_sp,
    avaliar_convergencia_rodada,
    identificar_extremos,
    formatar_resumo_debate,
    parse_escala
)
from src.config import Condicao

def _no_preparar_rodada(state: ConsensoState, *, deliberacao: bool) -> dict:
    """Prepara o contexto textual para a rodada atual (sintetiza divergências e justificativas CoT)."""
    rodada = state.get("rodada", 1)
    contexto = None

    if deliberacao and rodada > 1:
        anteriores = [e for e in state.get("estimativas", []) if e["rodada"] == rodada - 1]
        if anteriores:
            contexto = formatar_resumo_debate(anteriores, rodada - 1)

    return {"contexto_rodada_atual": contexto}

def _no_estimador(
    state: ConsensoState,
    *,
    agente_id: str,
    papel_id: str,
    modelo_id: str,
    temperatura: float,
    protocolo: str,
    api_key_env_var: Optional[str] = None,
    seed: int = 42
) -> dict:
    """Nó individual do estimador com suporte a revelação cega (simultânea) e sequencial encadeada."""
    rodada = state.get("rodada", 1)
    contexto = state.get("contexto_rodada_atual")

    # Papel fixo definido em config.py (sem rotação); o estado pode apenas confirmá-lo
    papeis_fixos = state.get("papeis_agentes") or {}
    papel_efetivo = papeis_fixos.get(agente_id, papel_id)

    # No protocolo sequencial, o agente recebe também os lances emitidos pelos colegas nesta mesma rodada
    ordem_revelacao = None
    if protocolo == "sequencial":
        atuais_nesta_rodada = [e for e in state.get("estimativas", []) if e["rodada"] == rodada]
        ordem_revelacao = len(atuais_nesta_rodada) + 1  # 1 = primeiro lance revelado (âncora do H4)
        if atuais_nesta_rodada:
            linhas_seq = "\n".join(
                f'- {e["papel"].capitalize()} ({e["modelo"]}): {int(e["story_points"])} SP\n'
                f'  *Justificativa CoT:* {e["justificativa"]}'
                for e in atuais_nesta_rodada
            )
            if contexto:
                contexto = f"{contexto}\n\n### Lances já revelados nesta rodada:\n{linhas_seq}"
            else:
                contexto = f"### Lances já revelados nesta rodada:\n{linhas_seq}"

    est = estimar(
        agente_id=agente_id,
        papel_id=papel_efetivo,
        modelo_id=modelo_id,
        temperatura=temperatura,
        titulo=state["titulo"],
        descricao=state["descricao"],
        escala_fibonacci=state["escala_fibonacci"],
        historicos=state.get("historicos"),
        contexto_rodada=contexto,
        rodada=rodada,
        protocolo=protocolo,
        api_key_env_var=api_key_env_var,
        seed=seed
    )
    est["ordem_revelacao"] = ordem_revelacao  # None no simultâneo (revelação cega)
    # Auditoria: o texto de debate/lances anteriores efetivamente entregue a este agente
    est["contexto_recebido"] = contexto
    return {"estimativas": [est]}

def _no_avaliar_consenso(
    state: ConsensoState,
    *,
    metodo: str,
    max_rodadas: int = 3
) -> dict:
    """
    Avalia o consenso pós-votação da rodada com base nas regras:
    - Unanimidade em qualquer rodada (R1, R2, R3): Encerra imediatamente.
    - Estagnação na Rodada 2: Se os mesmos agentes mantiverem exatamente os mesmos lances da R1,
      encerra imediatamente e adota a mediana dos lances da R2.
    - Rodada 3 (teto máximo): Se não atingir unanimidade, encerra formalizando a mediana das 3 estimativas finais.
    """
    rodada = state.get("rodada", 1)
    atuais = [e for e in state["estimativas"] if e["rodada"] == rodada]
    valores = [e["story_points"] for e in atuais]

    lances_atuais = {e["agente"]: float(e["story_points"]) for e in atuais}
    lances_anteriores = None
    if rodada > 1:
        anteriores = [e for e in state["estimativas"] if e["rodada"] == rodada - 1]
        lances_anteriores = {e["agente"]: float(e["story_points"]) for e in anteriores}

    escala = parse_escala(state["escala_fibonacci"])
    consenso = agregar(valores, escala, metodo)
    sigma = desvio_padrao_sp(valores)

    convergiu, motivo, dispersao = avaliar_convergencia_rodada(
        valores=valores,
        rodada=rodada,
        escala=escala,
        max_rodadas=max_rodadas,
        lances_atuais=lances_atuais,
        lances_anteriores=lances_anteriores
    )

    registro = {
        "rodada": rodada,
        "valores": valores,
        "lances_por_agente": lances_atuais,
        "consenso": consenso,
        "sigma_sp": sigma,
        "dispersao_degraus": dispersao,
        "motivo_convergencia": motivo,
        "convergiu": convergiu,
        "extremos": identificar_extremos(atuais)
    }

    return {
        "consenso": consenso,
        "convergiu": convergiu,
        "motivo_convergencia": motivo,
        "dispersao_degraus": dispersao,
        "rodada": rodada + 1,  # Prepara o contador para a próxima rodada
        "historico_rodadas": [registro],
    }

def _rotear_rodada(state: ConsensoState, *, max_rodadas: int, deliberacao: bool) -> Literal["preparar", "__end__"]:
    """Decide se encerra a sessão ou realiza uma nova rodada de debate CoT."""
    if not deliberacao:
        return END
    # Encerra se atingiu o critério de convergência (unanimidade na R1, unanimidade/adjacência na R2 ou consenso na R3)
    if state.get("convergiu", False):
        return END
    # Encerra se atingiu o limite máximo de rodadas (R_max = 3)
    if state["rodada"] > max_rodadas:
        return END
    return "preparar"

def construir_grafo(cond: Condicao, metodo: str = "mediana"):
    """
    Constrói e compila o StateGraph no LangGraph respeitando:
    - Fan-out paralelo para o protocolo simultâneo (Tratamento A)
    - Encadeamento estrito para o protocolo sequencial (Tratamento B)
    - Loop cíclico até convergência ou limite máximo de rodadas (R_max = 3)
    """
    builder = StateGraph(ConsensoState, start=START, end=END)

    # 1. Nó de orquestração da rodada
    builder.add_node(
        "preparar",
        partial(_no_preparar_rodada, deliberacao=cond.deliberacao)
    )

    # 2. Nós estimadores para cada agente especializado (ou neutro)
    nomes_estimadores = []
    for agente_tuple in cond.agentes:
        agente_id, papel_id, modelo_id = agente_tuple[:3]
        api_key_env_var = agente_tuple[3] if len(agente_tuple) > 3 else None

        obter_modelo(modelo_id, cond.temperatura, api_key_env_var=api_key_env_var, seed=getattr(cond, "seed", 42))

        nome_no = f"est_{agente_id}"
        nomes_estimadores.append(nome_no)
        builder.add_node(
            nome_no,
            partial(
                _no_estimador,
                agente_id=agente_id,
                papel_id=papel_id,
                modelo_id=modelo_id,
                temperatura=cond.temperatura,
                protocolo=cond.protocolo,
                api_key_env_var=api_key_env_var,
                seed=getattr(cond, "seed", 42)
            ),
        )

    # 3. Nó avaliador de consenso
    builder.add_node(
        "avaliar_consenso",
        partial(
            _no_avaliar_consenso,
            metodo=metodo,
            max_rodadas=cond.max_rodadas
        )
    )

    # 4. Estruturação das Arestas do Grafo
    builder.add_edge(START, "preparar")

    if cond.protocolo == "simultaneo":
        # Disparo paralelo (fan-out) a partir de preparar
        for nome_no in nomes_estimadores:
            builder.add_edge("preparar", nome_no)
            builder.add_edge(nome_no, "avaliar_consenso")
    else:
        # Encadeamento sequencial: preparar -> agente1 -> agente2 -> agente3 -> avaliar_consenso
        builder.add_edge("preparar", nomes_estimadores[0])
        for anterior, seguinte in zip(nomes_estimadores, nomes_estimadores[1:]):
            builder.add_edge(anterior, seguinte)
        builder.add_edge(nomes_estimadores[-1], "avaliar_consenso")

    # 5. Aresta condicional para fechar o ciclo de deliberação CoT
    builder.add_conditional_edges(
        "avaliar_consenso",
        partial(_rotear_rodada, max_rodadas=cond.max_rodadas, deliberacao=cond.deliberacao),
        {"preparar": "preparar", END: END}
    )

    return builder.compile()

if __name__ == "__main__":
    from src.config import CONDICOES
    for cond_id, cond in CONDICOES.items():
        g = construir_grafo(cond)
        print(f"Grafo '{cond_id}' compilado com sucesso!")
