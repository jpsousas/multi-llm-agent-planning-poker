"""
scripts/dev_resumo.py
=====================
Resumo AGREGADO da fase de desenvolvimento (results/dev/*.jsonl). Não imprime texto de issue.

Por condição/arquivo: n, MAE, acerto exato, ±1 degrau, comparação com a mediana global do histórico e com
a mediana dos k vizinhos (kNN), com Wilcoxon pareado no erro absoluto; motivos de encerramento, rodadas,
tokens, tempo e lances fora da escala. Compara também dois arquivos (ex.: sem limpeza vs com limpeza).

Uso:
    python -m scripts.dev_resumo                         # todos os results/dev/*.jsonl
    python -m scripts.dev_resumo --padrao "results/dev/*simultaneo_especializado*"
    python -m scripts.dev_resumo --comparar A.jsonl B.jsonl   # B vs A nas issues comuns
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from src.consensus import parse_escala, arredondar_fibonacci, distancia_degraus
from src.config import ESCALAS_FIBONACCI

ESCALA = parse_escala(ESCALAS_FIBONACCI["SIS-ASTROS"])


def ler(caminho):
    out = {}
    with open(caminho, encoding="utf-8") as f:
        for l in f:
            try:
                r = json.loads(l)
                out[str(r["issue_id"])] = r
            except Exception:
                pass
    return out


def mediana_global(arquivo_csv, split=0.7, split_dev=0.7):
    """Mediana de Time Estimate da BASE de exemplos da fase dev (mesmo corte de sis_astros.montar_dataset)."""
    from src.sis_astros import carregar_sis_astros, dividir_temporal, resolver_caminho
    itens = carregar_sis_astros(str(resolver_caminho(arquivo_csv)))
    hist, _ = dividir_temporal(itens, split, caminho_manifesto=None)
    base = hist[: int(len(hist) * split_dev)]
    return arredondar_fibonacci(float(np.median([it["Story_Point"] for it in base])), ESCALA)


def knn_mediana(r):
    sps = [h["story_points"] for h in r.get("contexto_historico", []) if h.get("story_points") is not None]
    return arredondar_fibonacci(float(np.median(sps)), ESCALA) if sps else None


def wil(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5 or np.all(a == b):
        return float("nan")
    return float(wilcoxon(a, b, zero_method="pratt").pvalue)


def metricas(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    deg = np.array([distancia_degraus(a, b, ESCALA) for a, b in zip(y, p)])
    return {"MAE": float(np.mean(np.abs(y - p))), "exato": float(np.mean(deg == 0) * 100),
            "ate1": float(np.mean(deg <= 1) * 100), "deg_medio": float(np.mean(deg))}


def linha(nome, m):
    return f"  {nome:<28} MAE={m['MAE']:.2f}  exato={m['exato']:.1f}%  ±1 degrau={m['ate1']:.1f}%  degraus médios={m['deg_medio']:.2f}"


def resumo(caminho, med_global):
    d = ler(caminho)
    if not d:
        print(f"{caminho}: vazio")
        return
    rs = list(d.values())
    y = [r["ground_truth"] for r in rs]
    p = [r["predito"] for r in rs]
    err = np.abs(np.array(y) - np.array(p))
    print(f"\n=== {Path(caminho).name}  (n={len(rs)}) ===")
    print(linha("Consenso MASPE", metricas(y, p)))
    if med_global is not None:
        pg = [med_global] * len(rs)
        print(linha(f"Mediana global ({med_global:g})", metricas(y, pg)))
        print(f"    Wilcoxon MASPE vs mediana global: p={wil(err, np.abs(np.array(y) - med_global)):.3f}")
    ok = [(r["ground_truth"], knn_mediana(r), r["predito"]) for r in rs if knn_mediana(r) is not None]
    if ok:
        yk, pk, pm = zip(*ok)
        print(linha("kNN (mediana dos vizinhos)", metricas(yk, pk)))
        print(f"    Wilcoxon MASPE vs kNN: p={wil(np.abs(np.array(yk) - np.array(pm)), np.abs(np.array(yk) - np.array(pk))):.3f}")
    # lance individual da R1 e mediana da R1 (o consenso agrega valor?)
    r1 = [[e["story_points"] for e in r["estimativas"] if e["rodada"] == 1] for r in rs]
    med1 = [float(np.median(v)) for v in r1]
    print(linha("Mediana dos lances da R1", metricas(y, med1)))
    print(f"    Wilcoxon consenso vs mediana R1: p={wil(err, np.abs(np.array(y) - np.array(med1))):.3f}")

    motivos = {}
    for r in rs:
        m = (r.get("motivo_convergencia") or "").split(" ")[0]
        motivos[m] = motivos.get(m, 0) + 1
    print("  encerramento:", ", ".join(f"{k}={v} ({v / len(rs) * 100:.0f}%)" for k, v in sorted(motivos.items())))
    rod = [r["rodadas_usadas"] for r in rs]
    print(f"  rodadas: média={np.mean(rod):.2f}  R1={rod.count(1)} R2={rod.count(2)} R3={rod.count(3)}")
    est = [e for r in rs for e in r["estimativas"]]
    fora = sum(1 for e in est if e.get("fora_da_escala"))
    print(f"  lances fora da escala (arredondados): {fora}/{len(est)}")
    print(f"  tokens/issue (médio)={np.mean([r['tokens_total'] for r in rs]):.0f}  "
          f"tempo/issue={np.mean([r['tempo_total_s'] for r in rs]):.0f}s  "
          f"tokens de entrada máx numa chamada={max(e.get('tokens_entrada', 0) for e in est)}")
    pm = [r.get("prep_meta") or {} for r in rs]
    if pm and pm[0]:
        print(f"  exemplos truncados={sum(m.get('exemplos_truncados', 0) for m in pm)} | "
              f"tokens do alvo (máx)={max(m.get('tokens_alvo', 0) for m in pm)} | "
              f"orçamento de exemplo (mín)={min(m.get('orcamento_exemplo', 0) for m in pm)}")
    falhas = Path(str(caminho).replace(".jsonl", "__falhas.jsonl"))
    if falhas.exists():
        print(f"  FALHAS registradas: {sum(1 for _ in open(falhas, encoding='utf-8'))}")
    hashes = {r.get("hash_prompt") for r in rs}
    print(f"  hash do prompt: {', '.join(sorted(h or '?' for h in hashes))}")


def comparar(a, b):
    da, db = ler(a), ler(b)
    comuns = sorted(set(da) & set(db))
    print(f"\n=== {Path(b).name}  vs  {Path(a).name}  ({len(comuns)} issues comuns) ===")
    if not comuns:
        return
    y = [da[i]["ground_truth"] for i in comuns]
    pa, pb = [da[i]["predito"] for i in comuns], [db[i]["predito"] for i in comuns]
    print(linha("A", metricas(y, pa)))
    print(linha("B", metricas(y, pb)))
    ea, eb = np.abs(np.array(y) - pa), np.abs(np.array(y) - pb)
    print(f"  diferença média de erro (B−A)={np.mean(eb - ea):+.2f} h; Wilcoxon p={wil(eb, ea):.3f}; "
          f"B melhor em {np.mean(eb < ea) * 100:.0f}% | pior em {np.mean(eb > ea) * 100:.0f}% | igual em {np.mean(eb == ea) * 100:.0f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--padrao", default="results/dev/*.jsonl")
    ap.add_argument("--arquivo", default=None, help="CSV do SIS-ASTROS (para a mediana global)")
    ap.add_argument("--comparar", nargs=2, metavar=("A", "B"))
    a = ap.parse_args()
    if a.comparar:
        comparar(*a.comparar)
        return
    try:
        med = mediana_global(a.arquivo)
    except Exception as e:
        print(f"[aviso] mediana global indisponível ({type(e).__name__}); seguindo sem ela.")
        med = None
    for c in sorted(glob.glob(a.padrao)):
        if "__falhas" not in c:
            resumo(c, med)


if __name__ == "__main__":
    main()