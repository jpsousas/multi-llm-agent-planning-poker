import json, time, re
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
from src.prompts import montar_system_prompt, montar_prompt_item

_cache = {}

def obter_modelo(identificador: str, temperatura: float):
    chave = (identificador, temperatura)
    if chave not in _cache:
        _cache[chave] = init_chat_model(identificador, temperature=temperatura)
    return _cache[chave]

def extrair_json(texto: str) -> dict:
    """LLMs as vezes devolvem JSON dentro de blocos markdown ou com
    texto ao redor. Estrategia em camadas, como fazem Shetty et al."""
    texto = re.sub(r"```(?:json)?|```", "", texto).strip()
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{.*\}', texto, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    m = re.search(r'(\d+(?:\.\d+)?)', texto) # ultimo recurso
    if m:
        return {"story_points": float(m.group(1)), "justificativa": texto[:200]}
    raise ValueError(f"Nao foi possivel extrair estimativa de: {texto[:200]}")

def estimar(agente_id, papel_id, modelo_id, temperatura,
    titulo, descricao, historicos=None, contexto_rodada=None, rodada=1):
    modelo = obter_modelo(modelo_id, temperatura)
    mensagens = [
        SystemMessage(montar_system_prompt(papel_id)),
        HumanMessage(montar_prompt_item(titulo, descricao,
        historicos, contexto_rodada)),
    ]
    t0 = time.perf_counter()
    resposta = modelo.invoke(mensagens)
    latencia = (time.perf_counter() - t0) * 1000
    dados = extrair_json(resposta.content)
    uso = getattr(resposta, "usage_metadata", {}) or {}
    return {
        "agente": agente_id,
        "papel": papel_id,
        "modelo": modelo_id,
        "story_points": float(dados["story_points"]),
        "justificativa": dados.get("justificativa", ""),
        "rodada": rodada,
        "tokens_entrada": uso.get("input_tokens", 0),
        "tokens_saida": uso.get("output_tokens", 0),
        "latencia_ms": latencia,
    }

if __name__ == "__main__":
    # testes
    print(estimar("agente1", "backend", "gpt-4o-mini", 0.5, "Titulo do item", "Descricao do item", historicos=[{"titulo": "Item 1", "story_points": 5}, {"titulo": "Item 2", "story_points": 8}], contexto_rodada="Item 1 => 5 SP\nItem 2 => 8 SP", rodada=1))