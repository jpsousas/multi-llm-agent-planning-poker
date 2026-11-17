"""
Compara, no bloco de DESENVOLVIMENTO, a recuperação por similaridade (kNN, SEM LLM) para
diferentes modelos de embedding, k e limiares de quase-duplicata. Imprime só números agregados
(nenhum texto, título ou ID). O bloco de teste nunca é lido.

Previsão do kNN para uma issue = mediana das estimativas dos k vizinhos, ajustada ao degrau mais próximo
da escala. Também mostra a mediana global da base (baseline sem recuperação).

Ao final, testes PAREADOS (mesmos alvos) sobre a diferença de erro por issue, d = erro(A) - erro(B):
  d < 0 favorece A. Reporta média de d com IC95% (bootstrap sobre as issues), valor-p de um teste de
  permutação por inversão de sinais, p ajustado por Holm (família = todas as comparações impressas),
  % de issues em que A é melhor/pior e a diferença mínima detectável (~80% de poder, alfa 5%).
  Duas métricas de erro: absoluto em horas e distância em degraus da escala (ordinal).

Uso (na raiz do projeto):
    python -m scripts.comparar_knn --arquivo issues_astros_ufsm_limpo.csv
    python -m scripts.comparar_knn --arquivo issues_astros_ufsm_limpo.csv --ks 3 5 --limiares 0.90 0.95 0.98
"""
import argparse
import statistics

import numpy as np

import src.retrieval as R
from src.sis_astros import (ESCALA_SIS_ASTROS, carregar_sis_astros, dividir_temporal, resolver_caminho)

MODELOS_PADRAO = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",   # 384 dim (seu projeto)
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",   # 768 dim (Rossi)
]
LIMIARES_SIMILARIDADE_TOPO = (0.90, 0.95, 0.98)


def para_degrau(valor: float) -> int:
    return min(ESCALA_SIS_ASTROS, key=lambda d: (abs(d - valor), d))


def metricas(previsto, real, n_boot=1000, semente=0):
    previsto, real = np.array(previsto, float), np.array(real, float)
    erro = np.abs(previsto - real)
    idx = {d: i for i, d in enumerate(ESCALA_SIS_ASTROS)}
    dist = np.array([abs(idx[int(p)] - idx[int(r)]) for p, r in zip(previsto, real)])
    rng = np.random.default_rng(semente)
    boots = [erro[rng.integers(0, len(erro), len(erro))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(mae=erro.mean(), lo=lo, hi=hi, exato=(dist == 0).mean() * 100, adj=(dist <= 1).mean() * 100)


def erros(previsto, real):
    """Erro absoluto (horas) e distância em degraus da escala, por issue."""
    idx = {d: i for i, d in enumerate(ESCALA_SIS_ASTROS)}
    absoluto = np.array([abs(p - r) for p, r in zip(previsto, real)], float)
    ordinal = np.array([abs(idx[int(p)] - idx[int(r)]) for p, r in zip(previsto, real)], float)
    return absoluto, ordinal


def teste_pareado(d, n_boot=10000, semente=0):
    """d = erro(A) - erro(B) por issue. Retorna média, IC95% bootstrap, p (permutação por sinais), MDE."""
    d = np.asarray(d, float)
    n = len(d)
    rng = np.random.default_rng(semente)
    media = d.mean()
    boots = d[rng.integers(0, n, (n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sinais = rng.choice([-1.0, 1.0], size=(n_boot, n))
    perm = np.abs((sinais * d).mean(axis=1))
    p = (1 + np.sum(perm >= abs(media) - 1e-12)) / (1 + n_boot)
    mde = 2.8 * d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return dict(media=media, lo=lo, hi=hi, p=p, mde=mde,
                melhor=(d < 0).mean() * 100, pior=(d > 0).mean() * 100)


def holm(ps):
    """Ajuste de Holm-Bonferroni (controla a taxa de erro familiar)."""
    ps = np.asarray(ps, float)
    ordem = np.argsort(ps)
    m = len(ps)
    ajust = np.empty(m)
    maximo = 0.0
    for rank, i in enumerate(ordem):
        maximo = max(maximo, (m - rank) * ps[i])
        ajust[i] = min(1.0, maximo)
    return ajust


def linha(nome, m, extra=""):
    return (f"  {nome:<44} MAE={m['mae']:6.2f} [IC95% {m['lo']:.2f}-{m['hi']:.2f}]  "
            f"exato={m['exato']:5.1f}%  ±1 degrau={m['adj']:5.1f}%{extra}")


def imprimir_comparacoes(erros_cfg, a, limiares):
    nomes = [m.split("/")[-1] for m in a.modelos]
    comps = []   # (grupo, rótulo, chave_A, chave_B)
    # A) modelo contra modelo (sem limiar)
    for k in a.ks:
        for i in range(len(nomes)):
            for j in range(i + 1, len(nomes)):
                comps.append(("Modelo", f"{nomes[i]} vs {nomes[j]} (k={k}, sem limiar)",
                              (nomes[i], None, k), (nomes[j], None, k)))
    # B) cada kNN contra a mediana (sem limiar)
    for nome in nomes:
        for k in a.ks:
            comps.append(("kNN vs mediana", f"{nome} k={k} vs mediana (sem limiar)", (nome, None, k), "mediana"))
    # C) k contra k (sem limiar)
    for nome in nomes:
        for i in range(len(a.ks)):
            for j in range(i + 1, len(a.ks)):
                comps.append(("k", f"{nome}: k={a.ks[i]} vs k={a.ks[j]} (sem limiar)", (nome, None, a.ks[i]), (nome, None, a.ks[j])))
    # D) efeito de cada limiar (contra sem limiar)
    for nome in nomes:
        for k in a.ks:
            for lim in limiares:
                if lim is not None:
                    comps.append(("Limiar", f"{nome} k={k}: limiar {lim:.2f} vs sem limiar", (nome, lim, k), (nome, None, k)))

    resultados = {0: [], 1: []}   # 0 = horas, 1 = degraus
    for _, _, ka, kb in comps:
        for m in (0, 1):
            resultados[m].append(teste_pareado(erros_cfg[ka][m] - erros_cfg[kb][m], n_boot=a.bootstrap))
    for m in (0, 1):
        for r, ph in zip(resultados[m], holm([r["p"] for r in resultados[m]])):
            r["p_holm"] = ph

    print("\n=== TESTES PAREADOS (d = erro(A) - erro(B); d < 0 favorece A) ===")
    print(f"  {len(comps)} comparações; Holm sobre toda a família; bootstrap/permutações = {a.bootstrap}; n alvos = {len(next(iter(erros_cfg.values()))[0])}")
    atual = None
    for idx, (grupo, rotulo, _, _) in enumerate(comps):
        if grupo != atual:
            print(f"\n[{grupo}]")
            atual = grupo
        print(f"  {rotulo}")
        for m, nome in ((0, "horas  "), (1, "degraus")):
            r = resultados[m][idx]
            print(f"      {nome}: d={r['media']:+.3f} [IC95% {r['lo']:+.3f}, {r['hi']:+.3f}]  p={r['p']:.3f}  p_Holm={r['p_holm']:.3f}  "
                  f"A melhor={r['melhor']:.0f}% pior={r['pior']:.0f}%  MDE~{r['mde']:.3f}")
    print("\nLeitura: diferença 'real' = IC95% sem o zero E p_Holm < 0,05. Se o IC inclui zero e o MDE é maior "
          "que a diferença que importa, a amostra não consegue discriminar (empate).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arquivo", default=None)
    ap.add_argument("--split", type=float, default=0.7, help="proporção do histórico (igual ao pipeline)")
    ap.add_argument("--split-dev", type=float, default=0.7, help="proporção da base de exemplos dentro do histórico")
    ap.add_argument("--modelos", nargs="+", default=MODELOS_PADRAO)
    ap.add_argument("--ks", nargs="+", type=int, default=[3, 5])
    ap.add_argument("--bootstrap", type=int, default=10000, help="repetições do bootstrap e da permutação")
    ap.add_argument("--limiares", nargs="+", type=float, default=[],
                    help="limiares de quase-duplicata a testar (além de 'sem limiar')")
    a = ap.parse_args()

    caminho = resolver_caminho(a.arquivo)
    itens = carregar_sis_astros(str(caminho))
    historico, _teste = dividir_temporal(itens, a.split, caminho_csv=str(caminho), caminho_manifesto=None)
    corte = int(len(historico) * a.split_dev)
    base, alvos = historico[:corte], historico[corte:]
    real = [int(t["Story_Point"]) for t in alvos]
    print(f"\n=== DEV: base de exemplos={len(base)} | alvos={len(alvos)} | bloco de teste NÃO usado ===")

    mediana_global = para_degrau(statistics.median(int(b["Story_Point"]) for b in base))
    print(linha(f"baseline: mediana da base ({mediana_global})", metricas([mediana_global] * len(alvos), real)))
    erros_cfg = {"mediana": erros([mediana_global] * len(alvos), real)}   # chave -> (abs, ord)

    kmax = max(a.ks)
    limiares = [None] + list(a.limiares)
    for modelo in a.modelos:
        R.MODELO_EMBEDDINGS = modelo
        R._modelo = None
        R.resetar_indice()
        indice = R.construir_indice(base, str(caminho))
        dim = indice["embeddings"].shape[1]
        print(f"\n--- {modelo.split('/')[-1]} ({dim} dim) ---")

        resultados = {lim: [] for lim in limiares}
        sim_topo, sem_vizinho = [], 0
        for t in alvos:
            for lim in limiares:
                sims, meta = R.buscar_com_diagnostico(
                    t["Title"], t.get("Description", ""), t["ID"], t.get("Creation_Date"), indice, k=kmax, limiar_duplicata=lim)
                resultados[lim].append(sims)
                if lim is None:
                    sim_topo.append(meta["sim_max"])
        validos = [s for s in sim_topo if s is not None]
        if validos:
            frac = " | ".join(f"≥{l:.2f}: {np.mean([s >= l for s in validos]) * 100:.1f}%" for l in LIMIARES_SIMILARIDADE_TOPO)
            print(f"  similaridade do vizinho mais próximo: mediana={np.median(validos):.3f}; alvos com vizinho {frac}")

        for lim in limiares:
            rotulo = "sem limiar" if lim is None else f"limiar {lim:.2f}"
            for k in a.ks:
                prev, reais = [], []
                for sims, r_ in zip(resultados[lim], real):
                    if not sims:
                        sem_vizinho += 1
                        prev.append(mediana_global)
                    else:
                        prev.append(para_degrau(statistics.median(int(s["Story_Point"]) for s in sims[:k])))
                    reais.append(r_)
                print(linha(f"kNN k={k} ({rotulo})", metricas(prev, reais)))
                erros_cfg[(modelo.split("/")[-1], lim, k)] = erros(prev, reais)
        if sem_vizinho:
            print(f"  (alvos sem vizinho elegível usaram a mediana da base: {sem_vizinho} casos no total)")

    imprimir_comparacoes(erros_cfg, a, limiares)

    print("\nObs.: com poucos alvos o IC é largo; diferenças menores que o IC não são evidência.")


if __name__ == "__main__":
    main()