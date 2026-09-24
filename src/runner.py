import json, time, asyncio
from pathlib import Path
from tqdm import tqdm
from src.graph import construir_grafo
from src.config import CONDICOES
from src.tawos import montar_dataset
from dotenv import load_dotenv

load_dotenv()

async def rodar_condicao(nome_cond, dataset, projeto, saida_dir="results/raw"):
    cond = CONDICOES[nome_cond]
    grafo = construir_grafo(cond)
    Path(saida_dir).mkdir(parents=True, exist_ok=True)
    caminho = Path(saida_dir) / f"{projeto}__{nome_cond}.jsonl"

    ja_feitos = set()
    if caminho.exists():
        with open(caminho, encoding="utf-8") as f:
            ja_feitos = {json.loads(l)["issue_id"] for l in f}

    with open(caminho, "a", encoding="utf-8") as f:
        for item in tqdm(dataset, desc=f"{projeto}/{nome_cond}"):
            if item["issue_id"] in ja_feitos:
                continue
            entrada = {
                "issue_id": item["issue_id"],
                "titulo": item["titulo"],
                "descricao": item["descricao"],
                "historicos": item["historicos"],
                "estimativas": [],
                "rodada": 1,
                "convergiu": False,
                "consenso": 0.0,
                "historico_rodadas": [],
            }
            t0 = time.perf_counter()
            try:
                final = await grafo.ainvoke(entrada, {"recursion_limit": 50, "max_concurrent": 5})
            except Exception as e:
                print(f"FALHA ao processar {item['issue_id']}: {e}")
                continue

            registro = {
                "issue_id": item["issue_id"],
                "condicao": nome_cond,
                "projeto": projeto,
                "ground_truth": item["ground_truth"],
                "esforco_real_min": item["esforco_real_min"],
                "predito": final["consenso"],
                "rodadas_usadas": final["rodada"] - 1,
                "convergiu": final["convergiu"],
                "estimativas": final["estimativas"],
                "historico_rodadas": final["historico_rodadas"],
                "tempo_total_s": time.perf_counter() - t0,
                "tokens_total": sum(e["tokens_entrada"] + e["tokens_saida"] for e in final["estimativas"]),
                "n_chamadas": len(final["estimativas"]),
            }
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
            f.flush() # grava imediatamente nao perde tudo se cair

if __name__ == "__main__":
    projeto = "Apache Mesos"
    dataset = montar_dataset(projeto)
    for cond in ["C0", "C1", "C2", "C3", "C3_delib_simultaneo", "C3_delib_sequencial"]:
        asyncio.run(rodar_condicao(cond, dataset, projeto))