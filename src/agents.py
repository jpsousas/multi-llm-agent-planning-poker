import json, time, re, os
from typing import Optional
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage
from src.prompts import montar_system_prompt, montar_prompt_item
from src.consensus import arredondar_fibonacci, parse_escala

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_cache = {}

def obter_modelo(identificador: str, temperatura: float, api_key_env_var: Optional[str] = None, seed: int = 42):
    """
    Instancia (ou recupera do cache) o cliente LLM.
    Suporta:
    - Modelos locais (DeepSeek 70B via Ollama, LM Studio, vLLM) através do protocolo local OpenAI-compatible.
    - Modelos via OpenRouter (mantendo compatibilidade).
    """
    chave_cache = (identificador, temperatura, api_key_env_var, seed)
    if chave_cache in _cache:
        return _cache[chave_cache]

    local_url = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1")

    if ":" in identificador:
        provedor, modelo = identificador.split(":", 1)

        if provedor in ("local", "ollama", "lmstudio", "vllm"):
            # DeepSeek-R1-Distill-Llama-70B servido localmente
            _cache[chave_cache] = ChatOpenAI(
                model=modelo,
                temperature=temperatura,
                seed=seed,
                api_key="local-not-needed",
                base_url=local_url,
                max_retries=1,
                timeout=180.0
            )
        elif provedor == "openrouter":
            api_key = None
            if api_key_env_var:
                possiveis_vars = [
                    api_key_env_var,
                    api_key_env_var.replace("API_", "API_KEY_"),
                    api_key_env_var.replace("API_KEY_", "API_"),
                    "OPENROUTER_API_KEY",
                ]
                for v in possiveis_vars:
                    val = os.environ.get(v)
                    if val and val.strip():
                        api_key = val.strip()
                        break

            _cache[chave_cache] = ChatOpenAI(
                model=modelo,
                temperature=temperatura,
                seed=seed,
                api_key=api_key or os.environ.get("OPENROUTER_API_KEY", "dummy"),
                base_url=OPENROUTER_BASE_URL,
                max_retries=0,
            )
        else:
            try:
                _cache[chave_cache] = init_chat_model(
                    modelo,
                    model_provider=provedor,
                    temperature=temperatura,
                    seed=seed
                )
            except Exception:
                _cache[chave_cache] = init_chat_model(
                    modelo,
                    model_provider=provedor,
                    temperature=temperatura
                )
    else:
        # Fallback para local se não houver prefixo de provedor
        _cache[chave_cache] = ChatOpenAI(
            model=identificador,
            temperature=temperatura,
            seed=seed,
            api_key="local-not-needed",
            base_url=local_url,
            max_retries=1,
            timeout=180.0
        )

    return _cache[chave_cache]

def vale_repetir(exc: Exception) -> bool:
    """Repete só o que tem chance de funcionar na segunda tentativa (decide pelo código HTTP ou timeout)."""
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status in (408, 409, 429) or status >= 500
    nome = type(exc).__name__
    return isinstance(exc, (TimeoutError, ConnectionError)) or "Timeout" in nome or "Connection" in nome

def extrair_texto_resposta(conteudo) -> str:
    """Extrai texto contínuo de respostas do LangChain (suporta str, list de blocos ou dicts)."""
    if isinstance(conteudo, str):
        return conteudo
    if isinstance(conteudo, list):
        partes = []
        for bloco in conteudo:
            if isinstance(bloco, dict) and "text" in bloco:
                partes.append(bloco["text"])
            elif isinstance(bloco, str):
                partes.append(bloco)
            else:
                partes.append(str(bloco))
        return "\n".join(partes)
    return str(conteudo)

def extrair_json(texto: str) -> dict:
    """
    Extrai story_points e justificativa com suporte especializado a modelos de raciocínio
    como o DeepSeek-R1 (que emite tags <think>...</think> com o Chain-of-Thought explícito).
    """
    cot_pensamento = ""
    # 1. Captura o raciocínio profundo dentro das tags <think> do DeepSeek R1
    m_think = re.search(r"<think>(.*?)</think>", texto, re.DOTALL)
    if m_think:
        cot_pensamento = m_think.group(1).strip()
        # Remove o bloco <think> para não confundir o parser JSON
        texto_sem_think = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()
    else:
        texto_sem_think = texto.strip()

    texto_limpo = re.sub(r"```(?:json)?|```", "", texto_sem_think).strip()

    def _completar(dados: dict) -> dict:
        if cot_pensamento and not dados.get("justificativa"):
            dados["justificativa"] = cot_pensamento
        elif cot_pensamento:
            dados["justificativa"] = f"[Raciocínio R1]: {cot_pensamento}\n\n[Síntese]: {dados.get('justificativa', '')}"
        return dados

    def _tentar_json(trecho: str) -> Optional[dict]:
        # Tolera vírgula sobrando antes de '}' ou ']' (erro comum de modelos menores)
        trecho = re.sub(r",\s*([}\]])", r"\1", trecho)
        try:
            dados = json.loads(trecho)
        except json.JSONDecodeError:
            return None
        if isinstance(dados, dict) and "story_points" in dados:
            return dados
        return None

    # Tentativa 1: o texto inteiro é o JSON
    dados = _tentar_json(texto_limpo)
    if dados is not None:
        return _completar(dados)

    # Tentativa 2: um objeto {...} no meio do texto
    for m in re.finditer(r"\{[^{}]*\}", texto_limpo, re.DOTALL):
        dados = _tentar_json(m.group())
        if dados is not None:
            return _completar(dados)

    # Tentativa 3: JSON malformado, mas com a chave story_points explícita.
    # Nunca pega um número solto do texto (ex.: "2 endpoints" na justificativa).
    m_num = re.search(r'"?story_points"?\s*[:=]\s*"?(\d+(?:\.\d+)?)', texto_limpo)
    if m_num:
        m_just = re.search(r'"justificativa"\s*:\s*"(.*?)"\s*[,}]', texto_limpo, re.DOTALL)
        justif = m_just.group(1) if m_just else (cot_pensamento or texto_limpo[:400])
        return {"story_points": float(m_num.group(1)), "justificativa": justif}

    # Armazena trecho detalhado para análise posterior do erro
    trecho = texto[:500] if len(texto) > 500 else texto
    raise ValueError(f"Não foi possível extrair estimativa válida do texto ({len(texto)} chars): {trecho}")

def estimar(
    agente_id: str,
    papel_id: str,
    modelo_id: str,
    temperatura: float,
    titulo: str,
    descricao: str,
    escala_fibonacci: str,
    historicos: Optional[list[dict]] = None,
    contexto_rodada: Optional[str] = None,
    rodada: int = 1,
    protocolo: str = "simultaneo",
    api_key_env_var: Optional[str] = None,
    seed: int = 42
) -> dict:
    """
    Executa a invocação do modelo LLM com retry inteligente e backoff exponencial.
    Preserva integralmente o Chain-of-Thought (CoT), latência, tokens e timestamp.
    """
    data_hora = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    modelo = obter_modelo(modelo_id, temperatura, api_key_env_var=api_key_env_var, seed=seed)
    mensagens = [
        SystemMessage(montar_system_prompt(papel_id, escala_fibonacci)),
        HumanMessage(montar_prompt_item(
            titulo=titulo,
            descricao=descricao,
            historicos=historicos,
            contexto_rodada=contexto_rodada,
            rodada=rodada,
            protocolo=protocolo
        )),
    ]
    print(mensagens[1].content)  # DEBUG: imprime o prompt completo enviado ao modelo
    max_tentativas = 3
    intervalo_base = 2.0

    for tentativa in range(max_tentativas):
        try:
            t0 = time.perf_counter()
            resposta = modelo.invoke(mensagens)
            latencia_ms = (time.perf_counter() - t0) * 1000

            texto_resposta = extrair_texto_resposta(resposta.content)
            dados = extrair_json(texto_resposta)
            uso = getattr(resposta, "usage_metadata", {}) or {}

            # Ajusta o lance ao degrau mais próximo da escala e registra se o modelo saiu dela
            sp_bruto = float(dados["story_points"])
            sp_escala = arredondar_fibonacci(sp_bruto, parse_escala(escala_fibonacci))

            return {
                "agente": agente_id,
                "papel": papel_id,
                "modelo": modelo_id,
                "story_points": sp_escala,
                "story_points_bruto": sp_bruto,
                "fora_da_escala": sp_escala != sp_bruto,
                "justificativa": dados.get("justificativa", "").strip(),
                "rodada": rodada,
                "tokens_entrada": uso.get("input_tokens", 0),
                "tokens_saida": uso.get("output_tokens", 0),
                "data_hora": data_hora,
                "latencia_ms": latencia_ms,
            }

        except Exception as e:
            if vale_repetir(e) and tentativa < max_tentativas - 1:
                espera = intervalo_base * (1.5 ** tentativa)
                print(f"\n[Aviso {agente_id.upper()}] {type(e).__name__}: {e}. Aguardando {espera:.1f}s (tentativa {tentativa+1}/{max_tentativas})...")
                time.sleep(espera)
            else:
                raise

if __name__ == "__main__":
    teste_texto = '<think>Esta tarefa exige criação de tabelas e endpoints complexos.</think>```json\n{"story_points": 5, "justificativa": "CRUD e migração de banco."}\n```'
    print("Teste extração DeepSeek R1:", extrair_json(teste_texto))
