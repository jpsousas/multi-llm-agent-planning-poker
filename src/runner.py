import json, time, asyncio, traceback, os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from src.graph import construir_grafo
from src.config import CONDICOES, ESCALAS_FIBONACCI, Condicao, MODELO_BASE, CAMINHO_SIS_ASTROS
from src.sis_astros import montar_dataset_sis_astros
from dotenv import load_dotenv

load_dotenv()

ESCALA_SIS_ASTROS = ESCALAS_FIBONACCI["SIS-ASTROS"]


def _condicao_com_modelo(
    nome_cond: str,
    modelo_override: Optional[str],
    url_override: Optional[str],
    temperatura: Optional[float] = None,
    seed: Optional[int] = None,
) -> Condicao:
    """
    Retorna a condição experimental com modelo, URL, temperatura e seed sobrescritos via CLI.
    O mesmo modelo é usado pelos 3 agentes (controle de H2).
    """
    cond_base = CONDICOES[nome_cond]

    # A URL é lida por agents.obter_modelo; a chave do cache de clientes inclui o modelo
    if url_override:
        os.environ["LOCAL_LLM_URL"] = url_override

    # Aceita "qwen2.5:3b" ou já prefixado ("ollama:qwen2.5:3b"); não duplica o prefixo local
    if modelo_override:
        prefixo = modelo_override.split(":", 1)[0]
        ja_prefixado = prefixo in ("local", "ollama", "lmstudio", "vllm")
        modelo_efetivo = modelo_override if ja_prefixado else f"local:{modelo_override}"
    else:
        modelo_efetivo = MODELO_BASE

    return replace(
        cond_base,
        agentes=[(a[0], a[1], modelo_efetivo) for a in cond_base.agentes],
        temperatura=cond_base.temperatura if temperatura is None else temperatura,
        seed=cond_base.seed if seed is None else seed,
    )

def _fmt_horas(horas) -> str:
    return f"{horas:g} h" if horas is not None else "não registrado"


def _celula_md(texto) -> str:
    """Texto seguro para célula de tabela Markdown (sem quebras de linha nem '|')."""
    return " ".join(str(texto or "").split()).replace("|", "\\|")


def salvar_transcricao_xai(registro: dict, diretorio_base: str = "results/transcripts"):
    """
    Exporta uma transcrição estruturada em Markdown para apoiar estudos
    de Inteligência Artificial Explicável (XAI) e análise qualitativa do debate.
    """
    caminho_dir = Path(diretorio_base) / registro["projeto"] / registro["condicao"]
    caminho_dir.mkdir(parents=True, exist_ok=True)
    caminho_arq = caminho_dir / f"{registro['issue_id']}.md"

    linhas = [
        f"# Transcrição da Sessão de Planning Poker Multiagente (MASPE)",
        f"- **Issue ID:** {registro['issue_id']}",
        f"- **Projeto:** {registro['projeto']}",
        f"- **Condição:** {registro['condicao']}",
        f"- **Data/Hora da Sessão:** {registro.get('timestamp_inicio', 'N/A')}",
        f"- **Seed de Reprodutibilidade:** {registro.get('seed', 42)} | **Temperatura:** {registro.get('temperatura', 0.0)}",
        f"- **Ground Truth (Humano Real):** {registro['ground_truth']} SP",
        f"- **Consenso Sintético Predito:** {registro['predito']} SP",
        f"- **Status de Convergência:** {'Convergiu' if registro['convergiu'] else 'Não convergiu (desempate por mediana no teto de rodadas)'}",
        f"- **Motivo / Critério de Parada:** {registro.get('motivo_convergencia', 'N/A')}",
        f"- **Dispersão Final (Degraus Fibonacci):** {registro.get('dispersao_degraus', 0)} degraus",
        f"- **Rodadas Utilizadas:** {registro['rodadas_usadas']}",
        f"- **Tempo Total:** {registro['tempo_total_s']:.2f}s | **Tokens Totais:** {registro['tokens_total']}",
        "",
        "## Issue Estimada",
        f"- **ID:** {registro['issue_id']}",
        f"- **Título:** {registro.get('titulo', '')}",
        f"- **Esforço Real Gasto (Time Spent):** {_fmt_horas(registro.get('esforco_real_horas'))}",
        "",
        "**Descrição:**",
        "",
        registro.get("descricao") or "_Sem descrição._",
        "",
        f"## Issues Históricas Recebidas como Contexto ({registro.get('modo_contexto', 'N/A')}, "
        f"k={len(registro.get('contexto_historico', []))})",
        "_Título e descrição exatamente como entregues aos agentes no prompt._",
        "",
    ]

    contexto = registro.get("contexto_historico", [])
    if contexto:
        linhas.append("| # | ID | Similaridade | Estimativa (SP) | Esforço Real (Time Spent) | Título | Descrição |")
        linhas.append("|---|----|--------------|-----------------|---------------------------|--------|-----------|")
        for i, h in enumerate(contexto, 1):
            sim = f"{h['similaridade']:.3f}" if h.get("similaridade") is not None else "—"
            sp = int(h["story_points"]) if h.get("story_points") is not None else "—"
            linhas.append(
                f"| {i} | {h.get('id', '')} | {sim} | {sp} | {_fmt_horas(h.get('esforco_real_horas'))} "
                f"| {_celula_md(h.get('titulo'))} | {_celula_md(h.get('descricao'))} |"
            )
    else:
        linhas.append("_Nenhuma issue histórica entregue._")

    linhas += ["", "## Atribuição de Papéis (fixa, sem rotação)"]
    if registro.get("ordem_fala"):
        linhas.append(f"- **Ordem de fala (sequencial):** {' -> '.join(p.upper() for p in registro['ordem_fala'])} "
                      f"(âncora do H4 = lance da posição 1)")
    papeis = registro.get("papeis_agentes") or registro.get("rotacao_papeis", {})
    versoes = registro.get("versoes_modelos", {})
    for agente_id, papel in papeis.items():
        linhas.append(f"- **Agente `{agente_id}`:** Papel `{papel.upper()}` | Modelo: `{versoes.get(agente_id, 'N/A')}`")

    linhas += [
        "",
        "---",
        "## Histórico Detalhado do Debate por Rodada",
        ""
    ]

    for h in registro.get("historico_rodadas", []):
        r = h["rodada"]
        sigma = h.get("sigma_sp", 0.0)
        disp = h.get("dispersao_degraus", 0)
        motivo = h.get("motivo_convergencia", "")
        linhas.append(f"### Rodada {r}")
        linhas.append(f"- **Desvio Padrão (&sigma;<sub>t</sub>):** {sigma:.2f} | **Dispersão Fibonacci (&Delta;):** {disp} degraus")
        linhas.append(f"- **Consenso da Rodada:** {h.get('consenso')} SP | **Decisão:** {motivo}")

        extremos = h.get("extremos", {})
        if extremos.get("diferenca", 0) > 0:
            linhas.append(
                f"- **Extremos:** Mínimo = {extremos['min']['papel']} ({extremos['min']['story_points']} SP) | "
                f"Máximo = {extremos['max']['papel']} ({extremos['max']['story_points']} SP)"
            )

        linhas.append("\n**Lances e Justificativas Chain-of-Thought (CoT):**")
        estimativas_r = [e for e in registro.get("estimativas", []) if e["rodada"] == r]
        for e in estimativas_r:
            ordem = f" (posição {e['ordem_revelacao']})" if e.get("ordem_revelacao") else ""
            recebeu = "sim" if e.get("contexto_recebido") else "não"
            linhas.append(
                f"- **[{e['papel'].upper()}]**{ordem} ({e['modelo']}): **{int(e['story_points'])} SP** "
                f"| recebeu debate/lances anteriores: {recebeu}\n"
                f"  > *Justificativa CoT:* {e['justificativa']}\n"
                f"  *(Horário: {e.get('data_hora', 'N/A')}, Latência: {e['latencia_ms']:.1f}ms, Tokens: {e['tokens_entrada']} in / {e['tokens_saida']} out)*"
            )
        linhas.append("")

    with open(caminho_arq, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))

async def rodar_condicao(
    nome_cond: str,
    dataset: list[dict],
    projeto: str,
    saida_dir: str = "results/raw",
    limite: Optional[int] = None,
    modelo_override: Optional[str] = None,
    url_override: Optional[str] = None,
    temperatura: Optional[float] = None,
    seed: Optional[int] = None,
):
    """Executa uma condição experimental sobre o conjunto de dados e grava logs em JSONL e Markdown."""
    if nome_cond not in CONDICOES:
        raise ValueError(f"Condição desconhecida: {nome_cond}. Disponíveis: {list(CONDICOES.keys())}")

    cond = _condicao_com_modelo(nome_cond, modelo_override, url_override, temperatura, seed)
    grafo = construir_grafo(cond)
    Path(saida_dir).mkdir(parents=True, exist_ok=True)

    # Nome do arquivo inclui o modelo para não misturar resultados de modelos diferentes
    # ...e o modo de contexto, para não misturar RAG com few-shot
    slug_modelo = (modelo_override or "default").replace("/", "_").replace(":", "-")
    modo = dataset[0].get("modo_contexto", "few_shot") if dataset else "few_shot"
    caminho = Path(saida_dir) / f"{projeto}__{nome_cond}__{slug_modelo}__{modo}.jsonl"
    caminho_falhas = Path(saida_dir) / f"{projeto}__{nome_cond}__{slug_modelo}__{modo}__falhas.jsonl"

    ja_feitos = set()
    if caminho.exists():
        with open(caminho, encoding="utf-8") as f:
            for l in f:
                try:
                    ja_feitos.add(json.loads(l)["issue_id"])
                except Exception:
                    pass

    itens_executar = [item for item in dataset if item["issue_id"] not in ja_feitos]
    if limite:
        itens_executar = itens_executar[:limite]

    modelo_label = modelo_override or "config-default"
    print(f"\nIniciando execução: {projeto} / {nome_cond} / modelo={modelo_label} ({len(itens_executar)} itens pendentes)")

    # SIS-ASTROS usa exclusivamente a escala Fibonacci pura (1,2,3,5,8,13,21,34,55)
    escala_atual = ESCALA_SIS_ASTROS

    versoes_modelos = {a[0]: a[2] for a in cond.agentes}

    with open(caminho, "a", encoding="utf-8") as f:
        for item in tqdm(itens_executar, desc=f"{projeto}/{nome_cond}"):
            # Papéis fixos (sem rotação). No sequencial, a ordem de fala é a ordem de cond.agentes
            # (especializado: Frontend -> Backend -> QA; ver config.py).
            papeis_agentes = {agente_tuple[0]: agente_tuple[1] for agente_tuple in cond.agentes}

            ts_inicio = datetime.now(timezone.utc).isoformat()
            entrada = {
                "issue_id": item["issue_id"],
                "titulo": item["titulo"],
                "descricao": item["descricao"],
                "historicos": item.get("historicos", []),
                "escala_fibonacci": escala_atual,
                "protocolo": cond.protocolo,
                "especializado": cond.especializado,
                "papeis_agentes": papeis_agentes,
                "estimativas": [],
                "rodada": 1,
                "convergiu": False,
                "motivo_convergencia": None,
                "dispersao_degraus": None,
                "consenso": 0.0,
                "contexto_rodada_atual": None,
                "historico_rodadas": [],
            }
            t0 = time.perf_counter()
            try:
                final = await grafo.ainvoke(entrada, {"recursion_limit": 50, "max_concurrent": 5})
            except Exception as e:
                registro_falha = {
                    "issue_id": item["issue_id"],
                    "condicao": nome_cond,
                    "projeto": projeto,
                    "modelo": modelo_label,
                    "titulo": item["titulo"],
                    "descricao": item.get("descricao", ""),
                    "ground_truth": item.get("ground_truth"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "tipo_erro": type(e).__name__,
                    "mensagem_erro": str(e),
                    "traceback": traceback.format_exc(),
                    "papeis_agentes": papeis_agentes,
                    "modelos": versoes_modelos,
                }
                with open(caminho_falhas, "a", encoding="utf-8") as f_falhas:
                    f_falhas.write(json.dumps(registro_falha, ensure_ascii=False) + "\n")
                print(f"\n[FALHA] Item {item['issue_id']} ({type(e).__name__}): {e}")
                time.sleep(1.0)
                continue

            rodadas_executadas = max(1, final["rodada"] - 1)
            ts_fim = datetime.now(timezone.utc).isoformat()
            registro = {
                "issue_id": item["issue_id"],
                "condicao": nome_cond,
                "projeto": projeto,
                "modelo": modelo_label,
                "timestamp_inicio": ts_inicio,
                "timestamp_fim": ts_fim,
                "seed": getattr(cond, "seed", 42),
                "temperatura": cond.temperatura,
                "protocolo": cond.protocolo,
                "papeis_agentes": papeis_agentes,
                "ordem_fala": [a[1] for a in cond.agentes] if cond.protocolo == "sequencial" else None,
                "versoes_modelos": versoes_modelos,
                # Issue estimada (texto completo, como entregue aos agentes)
                "titulo": item["titulo"],
                "descricao": item.get("descricao", ""),
                "ground_truth": item["ground_truth"],
                "esforco_real_horas": item.get("esforco_real_horas"),
                # Issues históricas entregues como contexto (título/descrição exatamente como no prompt;
                # esforco_real_horas = coluna Time Spent em horas, None se ausente)
                "modo_contexto": item.get("modo_contexto"),
                "k_contexto": item.get("k_contexto"),
                "contexto_historico": [
                    {
                        "id": h.get("id"),
                        "titulo": h.get("titulo"),
                        "descricao": h.get("descricao"),
                        "story_points": h.get("story_points"),
                        "esforco_real_horas": h.get("esforco_real_horas"),
                        "similaridade": h.get("similaridade"),
                    }
                    for h in item.get("historicos", [])
                ],
                "ids_contexto": [h.get("id") for h in item.get("historicos", [])],
                "similaridades_contexto": [h.get("similaridade") for h in item.get("historicos", [])],
                "predito": final["consenso"],
                "rodadas_usadas": rodadas_executadas,
                "convergiu": final.get("convergiu", False),
                "motivo_convergencia": final.get("motivo_convergencia"),
                "dispersao_degraus": final.get("dispersao_degraus"),
                "estimativas": final["estimativas"],
                "historico_rodadas": final["historico_rodadas"],
                "tempo_total_s": time.perf_counter() - t0,
                "tokens_total": sum(e.get("tokens_entrada", 0) + e.get("tokens_saida", 0) for e in final["estimativas"]),
                "n_chamadas": len(final["estimativas"]),
            }

            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
            f.flush()
            salvar_transcricao_xai(registro)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Executor do Planning Poker Multiagente (MASPE) — SIS-ASTROS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso (CSV padrão: data/sis_astros/dataset_sis_astros.csv — ver src/config.py):
  # Teste rápido com modelo leve no LM Studio (notebook), com RAG:
  python -m src.runner --modelo qwen2.5-14b --url http://localhost:1234/v1 --rag --limite 3

  # Execução no laboratório com DeepSeek 70B via Ollama:
  python -m src.runner --modelo deepseek-r1-distill-llama-70b --url http://localhost:11434/v1 --rag

  # Todas as condições, outro arquivo:
  python -m src.runner --arquivo caminho/outro.csv --condicao todas --rag
        """
    )
    parser.add_argument("--arquivo", "--arquivo-sisastros", dest="arquivo", type=str, default=None,
                        help=f"CSV/XLSX do SIS-ASTROS (padrão: {CAMINHO_SIS_ASTROS}, definido em src/config.py)")
    parser.add_argument("--modelo", type=str, default=None,
                        help="Nome do modelo LLM (ex: deepseek-r1-distill-llama-70b, qwen2.5:14b). "
                             "Sobrescreve LOCAL_LLM_MODEL do .env.")
    parser.add_argument("--url", type=str, default=None,
                        help="URL da API local (ex: http://localhost:1234/v1 para LM Studio, "
                             "http://localhost:11434/v1 para Ollama). Sobrescreve LOCAL_LLM_URL do .env.")
    parser.add_argument("--condicao", type=str, default="simultaneo_especializado",
                        choices=list(CONDICOES.keys()) + ["todas"],
                        help="Condição experimental a executar")
    parser.add_argument("--limite", type=int, default=None,
                        help="Número máximo de issues a processar (útil para testes)")
    parser.add_argument("--rag", action="store_true",
                        help="Ativa busca semântica de issues similares (RAG) ao invés de Few-Shot estratificado")
    parser.add_argument("--k-rag", type=int, default=3,
                        help="Número de issues similares a recuperar no modo RAG (padrão: 3)")
    parser.add_argument("--temperatura", type=float, default=None,
                        help="Temperatura de geração (padrão da Condicao: 0.0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed de geração (padrão da Condicao: 42)")
    args = parser.parse_args()

    # Carrega o CSV, aplica o split temporal 70/30 (manifesto em results/split_sis_astros.json)
    # e, com --rag, constrói o índice de embeddings só com o histórico.
    dataset = montar_dataset_sis_astros(
        caminho_arquivo=args.arquivo,
        usar_rag=args.rag,
        k_rag=args.k_rag,
    )

    condicoes_executar = list(CONDICOES.keys()) if args.condicao == "todas" else [args.condicao]

    for c in condicoes_executar:
        asyncio.run(rodar_condicao(
            c, dataset,
            projeto="SIS-ASTROS",
            limite=args.limite,
            modelo_override=args.modelo,
            url_override=args.url,
            temperatura=args.temperatura,
            seed=args.seed,
        ))
