import json, glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

def carregar(padrao="results/raw/*.jsonl") -> pd.DataFrame:
    linhas = []
    for arq in glob.glob(padrao):
        with open(arq) as f:
            linhas += [json.loads(l) for l in f]
    return pd.DataFrame(linhas)

def mae(y, yhat): return float(np.mean(np.abs(np.array(y) - np.array(yhat))))
def mdae(y, yhat): return float(np.median(np.abs(np.array(y) - np.array(yhat))))

def sa(y, yhat, n_runs=1000, seed=42):
    """Standardized Accuracy: compara contra chute aleatorio.
    Definida por Shepperd e MacDonell, usada por Choetkiertikul et al."""
    rng = np.random.default_rng(seed)
    y = np.array(y)
    maes_aleatorios = [np.mean(np.abs(y - rng.choice(y, size=len(y)))) for _ in range(n_runs)]
    return float((1 - mae(y, yhat) / np.mean(maes_aleatorios)) * 100)

def a12(x, y):
    """Vargha-Delaney A12 para amostras pareadas: probabilidade de
    o metodo X errar MENOS que o metodo Y no mesmo item.
    x, y sao vetores de erro absoluto."""
    x, y = np.array(x), np.array(y)
    menor = np.sum(x < y)
    igual = np.sum(x == y)
    return float(menor + 0.5 * igual) / len(x)

def comparar (df, cond_a, cond_b, projeto=None):
    """Compara duas condicoes NO MESMO conjunto de itens (pareado)."""
    d = df if projeto is None else df[df.projeto == projeto]
    a = d[d.condicao == cond_a].set_index("issue_id")
    b = d[d.condicao == cond_b].set_index("issue_id")
    comuns = a.index.intersection(b.index)
    a, b = a.loc[comuns], b.loc[comuns]

    erro_a = np.abs(a.ground_truth - a.predito)
    erro_b = np.abs(b.ground_truth - b.predito)

    try:
        stat, p = wilcoxon(erro_a, erro_b)
    except ValueError:
        stat, p = np.nan, 1.0 # todos os pares identicos

    return {
        "projeto": projeto or "todos",
        "comparacao": f"{cond_a} vs {cond_b}",
        "n": len(comuns),
        "MAE_a": mae(a.ground_truth, a.predito),
        "MAE_b": mae(b.ground_truth, b.predito),
        "MdAE_a": mdae(a.ground_truth, a.predito),
        "MdAE_b": mdae(b.ground_truth, b.predito),
        "SA_a": sa(a.ground_truth, a.predito),
        "SA_b": sa(b.ground_truth, b.predito),
        "p_valor": p,
        "A12": a12(erro_a, erro_b),
        "chamadas_a": a.n_chamadas.mean(),
        "chamadas_b": b.n_chamadas.mean(),
    }

def bonferroni(p_valores, alfa=0.05):
    return alfa / len(p_valores)

def analise_ancoragem(df, cond_simult, cond_seq):
    """Analisa a dispersao e acuracia de condicoes com deliberacao
    simultanea vs sequencial, para verificar se ha indicio de ancoragem."""
    linhas = []
    for cond in [cond_simult, cond_seq]:
        d = df[df.condicao == cond]
        for _, r in d.iterrows():
            for h in r["historico_rodadas"]:
                linhas.append({
                    "condicao": cond,
                    "issue_id": r.issue_id,
                    "rodada": h["rodada"],
                    "dispersao": h["dispersao"]
                })
        return pd.DataFrame(linhas).groupby(["condicao", "rodada"]).dispersao.agg(["mean", "std", "count"]).reset_index()
                