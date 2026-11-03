"""
src/analysis.py
===============
Análise estatística do MASPE (SIS-ASTROS; o mesmo LLM local nos 3 agentes).

Desenho: 2 x 2 = especialização (especializado | neutro) x protocolo (simultâneo | sequencial).

Regra de parada (src/consensus.py), base do H5:
  unanimidade em qualquer rodada -> encerra com o valor unânime;
  estagnação na R2 (cada agente repete o lance da R1) -> encerra com a mediana da R2;
  teto na R3 sem unanimidade -> encerra com a mediana da R3.

Mapeamento RQ/H -> procedimento:
  RQ1/H1   simultaneo_especializado. Consenso vs (a) lance individual da R1 de cada papel e
           (b) agregação simples dos lances iniciais (mediana dos 3 lances da R1).
           MAE (principal), MdAE, PRED(25), Spearman; Wilcoxon + Â12; Holm nas 4 comparações.
  RQ2.1/H2 simultaneo_especializado vs simultaneo_neutro, mesmas issues. MAE; Wilcoxon + Â12.
           O par sequencial é reportado como análise secundária.
  RQ2.2/H3 consenso especializado vs SBERT+LightGBM (Rossi, 2026): MAE e Spearman. O teste
           pareado exige as predições do baseline por issue (--baseline-rossi).
  RQ3/H4   simultaneo_especializado vs sequencial_especializado, âncora FIXA da R1:
             sequencial -> v1_R1, 1º lance revelado na R1;
             simultâneo -> Mediana_R1, mediana dos lances da R1.
           Deslocamento do agente i na rodada t: |voto_{i,t} − Âncora_R1|.
           No sequencial, o agente que emitiu a âncora fica fora APENAS do deslocamento da R1
           (distância zero por construção); a partir da R2 ele é incluído, pois pode mudar de
           lance após o debate.
           (i)  proporção de lances revisados que se aproximam da âncora (descritivo);
           (ii) magnitude média de alteração dos lances (Σ|ΔSP| por agente) — Wilcoxon + Â12;
           (iii) MAE do consenso — Wilcoxon + Â12;
           + trajetória do deslocamento por rodada.
  RQ4/H5   σ_t (desvio padrão dos lances) por rodada; teste pareado R1->R2 e R2->R3 nos itens
           que chegaram à rodada seguinte; distribuição dos motivos de encerramento e das rodadas.
  RQ4/H6   amostra de 10 sessões (5 unânimes na R1 + 5 com mais rodadas). Unidade de codificação:
           a justificativa CoT individual de cada agente em cada rodada (≥ 30 textos: 10 sessões
           x 3 agentes na R1, mais as rodadas de debate). Concordância pelo Kappa de Cohen.

Convenções estatísticas:
  - Â12 (Vargha & Delaney, 2000) sobre todos os pares (produto cartesiano):
        Â12(A, B) = U_A / (m·n),  U_A = mannwhitneyu(A, B, alternative='less').statistic
                  = P(A > B) + 0,5·P(A = B).
    LEITURA: como a variável comparada é o erro absoluto (menor é melhor), Â12 < 0,50 indica
    que A tende a apresentar erro MENOR que B (A é superior). Ver NOTA_A12.
    Magnitude por |Â12 − 0,5|: < 0,06 desprezível; < 0,14 pequeno; < 0,21 médio; senão grande.
  - Wilcoxon de postos sinalizados pareado, bicaudal, zero_method="pratt". α = 0,05.
  - Com N = 3 agentes votando na escala, a mediana de 3 valores já é um valor da escala
    (não há ajuste/arredondamento).
  - Na escala 1, 2, 3, 5, 8, 13, 21, 34, 55, PRED(25) equivale à taxa de acerto exato.
"""

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from sklearn.metrics import cohen_kappa_score

from src.consensus import desvio_padrao_sp, calcular_dispersao_degraus

ESCALA_SIS_ASTROS = [1, 2, 3, 5, 8, 13, 21, 34, 55]
ALFA = 0.05
ALTERNATIVA = "two-sided"

NOTA_A12 = (
    "Como a variável analisada é o erro absoluto (onde menor é melhor), valores de Â12 < 0,50 "
    "indicam a superioridade estatística de A (probabilidade de A apresentar erro menor que B)."
)

COND_SIM_ESP = "simultaneo_especializado"
COND_SEQ_ESP = "sequencial_especializado"
COND_SIM_NEU = "simultaneo_neutro"
COND_SEQ_NEU = "sequencial_neutro"
CONDICOES = [COND_SIM_ESP, COND_SEQ_ESP, COND_SIM_NEU, COND_SEQ_NEU]


# ==============================================================================
# 1. CARREGAMENTO
# ==============================================================================

def carregar_resultados(
    padrao: str = "results/raw/*.jsonl",
    modelo: Optional[str] = None,
    modo_contexto: Optional[str] = None,
    projeto: Optional[str] = "SIS-ASTROS",
) -> pd.DataFrame:
    """
    Carrega os JSONL do runner. Ignora *__falhas.jsonl e linhas corrompidas; filtra por
    projeto, modelo e modo de contexto ("rag"/"few_shot") para não misturar execuções.
    """
    linhas = []
    for arq in sorted(glob.glob(padrao)):
        if "__falhas" in Path(arq).name:
            continue
        with open(arq, "r", encoding="utf-8", errors="ignore") as f:
            for l in f:
                l = l.strip()
                if not l:
                    continue
                try:
                    linhas.append(json.loads(l))
                except json.JSONDecodeError:
                    pass
    if not linhas:
        return pd.DataFrame()
    df = pd.DataFrame(linhas)
    for col, valor in (("projeto", projeto), ("modelo", modelo), ("modo_contexto", modo_contexto)):
        if valor is not None and col in df:
            df = df[df[col] == valor]
    return df


def por_condicao(df: pd.DataFrame, condicao: str) -> pd.DataFrame:
    """Linhas de uma condição, uma por issue, indexadas por issue_id e ordenadas."""
    if df.empty or "condicao" not in df:
        return pd.DataFrame()
    d = df[df.condicao == condicao].copy()
    d["issue_id"] = d["issue_id"].astype(str)
    return d.drop_duplicates("issue_id", keep="last").set_index("issue_id").sort_index()


def pareados(df: pd.DataFrame, cond_a: str, cond_b: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """As duas condições restritas às issues presentes em ambas, na mesma ordem."""
    a, b = por_condicao(df, cond_a), por_condicao(df, cond_b)
    if a.empty or b.empty:
        return pd.DataFrame(), pd.DataFrame()
    comuns = a.index.intersection(b.index)
    return a.loc[comuns], b.loc[comuns]


def lances(row, rodada: int) -> List[dict]:
    """Lances de uma rodada, na ordem registrada."""
    return [e for e in row["estimativas"] if e["rodada"] == rodada]


def rodadas_da_sessao(row) -> List[int]:
    return sorted({e["rodada"] for e in row["estimativas"]})


def votos_por_rodada(row) -> Dict[int, Dict[str, float]]:
    """{rodada: {agente: voto}}."""
    return {r: {e["agente"]: float(e["story_points"]) for e in lances(row, r)} for r in rodadas_da_sessao(row)}


def chave_individual(e: dict) -> str:
    """Identidade do estimador individual: o papel (especializado) ou o agente (neutro)."""
    return e["agente"] if e.get("papel") == "neutro" else e["papel"]


def mediana_r1(row) -> float:
    """Mediana dos lances da R1. Com 3 lances na escala, o valor central já pertence à escala."""
    return float(np.median([e["story_points"] for e in lances(row, 1)]))


# ==============================================================================
# 2. MÉTRICAS E TESTES
# ==============================================================================

def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def mae(y, yhat) -> float:
    """Erro absoluto médio."""
    y, yhat = _arr(y), _arr(yhat)
    return float(np.mean(np.abs(y - yhat))) if len(y) else float("nan")


def mdae(y, yhat) -> float:
    """Erro absoluto mediano."""
    y, yhat = _arr(y), _arr(yhat)
    return float(np.median(np.abs(y - yhat))) if len(y) else float("nan")


def pred(y, yhat, k: float = 0.25) -> float:
    """PRED(k): % de estimativas com |y − ŷ| / y ≤ k."""
    y, yhat = _arr(y), _arr(yhat)
    m = y > 0
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.abs(y[m] - yhat[m]) / y[m] <= k) * 100.0)


def spearman_corr(y_real, y_pred) -> Tuple[float, float]:
    """Spearman (r_s, p) via scipy.stats.spearmanr. NaN se indefinida (ex.: predição constante)."""
    if len(y_real) < 3:
        return float("nan"), float("nan")
    res = spearmanr(y_real, y_pred)
    return float(res.statistic), float(res.pvalue)


def metricas(y, yhat) -> dict:
    """Métricas de acurácia em relação ao ground truth (Time Estimate)."""
    r, p = spearman_corr(y, yhat)
    return {
        "n": int(len(y)),
        "MAE": mae(y, yhat),
        "MdAE": mdae(y, yhat),
        "PRED(25)": pred(y, yhat, 0.25),
        "Spearman_rs": r,
        "Spearman_p": p,
        "vies_medio": float(np.mean(_arr(yhat) - _arr(y))) if len(y) else float("nan"),  # + = superestima
    }


def a12(a, b) -> float:
    """
    Â12 de Vargha-Delaney sobre todos os pares: U_A / (m·n), com U_A de
    scipy.stats.mannwhitneyu(a, b, alternative='less'). Equivale a P(A > B) + 0,5·P(A = B).
    Para erros: Â12 < 0,50 => A tende a errar menos que B.
    """
    a, b = _arr(a), _arr(b)
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return float("nan")
    u = mannwhitneyu(a, b, alternative="less").statistic
    return float(u / (m * n))


def magnitude_a12(valor: float) -> str:
    if np.isnan(valor):
        return "indefinido"
    d = abs(valor - 0.5)
    if d < 0.06:
        return "desprezivel"
    if d < 0.14:
        return "pequeno"
    if d < 0.21:
        return "medio"
    return "grande"


def teste_wilcoxon(a, b) -> dict:
    """Wilcoxon de postos sinalizados pareado, bicaudal, zero_method='pratt'."""
    a, b = _arr(a), _arr(b)
    out = {"n_pares": int(len(a)), "n_pares_nao_nulos": int(np.sum(a != b)),
           "W": float("nan"), "p": float("nan")}
    if len(a) == 0:
        return out
    if out["n_pares_nao_nulos"] == 0:
        out.update(W=0.0, p=1.0)  # amostras idênticas: nenhuma evidência de diferença
        return out
    res = wilcoxon(a, b, zero_method="pratt", alternative=ALTERNATIVA)
    out.update(W=float(res.statistic), p=float(res.pvalue))
    return out


def holm(pvalores: Dict[str, float]) -> Dict[str, float]:
    """Correção de Holm-Bonferroni (NaN é ignorado)."""
    validos = sorted(((p, k) for k, p in pvalores.items() if not np.isnan(p)), key=lambda t: t[0])
    m = len(validos)
    ajustados, acumulado = {}, 0.0
    for i, (p, k) in enumerate(validos):
        acumulado = max(acumulado, min(1.0, (m - i) * p))
        ajustados[k] = acumulado
    return {k: ajustados.get(k, float("nan")) for k in pvalores}


def comparar(valores_a, valores_b, rotulo_a: str, rotulo_b: str) -> dict:
    """Comparação pareada padrão: Wilcoxon (pratt, bicaudal) + Â12(A, B)."""
    w = teste_wilcoxon(valores_a, valores_b)
    v = a12(valores_a, valores_b)
    return {
        "A": rotulo_a,
        "B": rotulo_b,
        "media_A": float(np.mean(valores_a)) if len(valores_a) else float("nan"),
        "media_B": float(np.mean(valores_b)) if len(valores_b) else float("nan"),
        "mediana_A": float(np.median(valores_a)) if len(valores_a) else float("nan"),
        "mediana_B": float(np.median(valores_b)) if len(valores_b) else float("nan"),
        "n_pares": w["n_pares"],
        "n_pares_nao_nulos": w["n_pares_nao_nulos"],
        "Wilcoxon_W": w["W"],
        "Wilcoxon_p": w["p"],
        "significativo": bool(not np.isnan(w["p"]) and w["p"] < ALFA),
        "A12": v,  # < 0,5: A tende a ter valores menores que B
        "A12_magnitude": magnitude_a12(v),
    }


# ==============================================================================
# 3. RQ1 / H1 — EFICÁCIA DO CONSENSO
# ==============================================================================

def analisar_rq1_h1(df: pd.DataFrame, condicao: str = COND_SIM_ESP) -> dict:
    d = por_condicao(df, condicao)
    if d.empty:
        return {"status": "sem_dados", "condicao": condicao}

    y = d["ground_truth"].astype(float).to_numpy()
    consenso = d["predito"].astype(float).to_numpy()
    erro_consenso = np.abs(y - consenso)
    res = {"condicao": condicao, "n_itens": int(len(d)), "Consenso": metricas(y, consenso),
           "nota_A12": NOTA_A12.replace("de A", "do consenso"), "Comparacoes": {}}

    # (a) lance individual da R1 de cada papel (no simultâneo, votação cega e independente)
    chaves = sorted({chave_individual(e) for _, row in d.iterrows() for e in lances(row, 1)})
    for ch in chaves:
        lance = np.array([
            next((e["story_points"] for e in lances(row, 1) if chave_individual(e) == ch), np.nan)
            for _, row in d.iterrows()
        ], dtype=float)
        ok = ~np.isnan(lance)
        res["Comparacoes"][f"Individual_R1[{ch}]"] = {
            "metricas": metricas(y[ok], lance[ok]),
            "teste_erro_absoluto": comparar(erro_consenso[ok], np.abs(y[ok] - lance[ok]), "consenso", f"individual_{ch}"),
        }

    # (b) agregação simples dos lances iniciais: mediana dos 3 lances da R1
    med_r1 = np.array([mediana_r1(row) for _, row in d.iterrows()], dtype=float)
    res["Comparacoes"]["Mediana_R1"] = {
        "metricas": metricas(y, med_r1),
        "teste_erro_absoluto": comparar(erro_consenso, np.abs(y - med_r1), "consenso", "mediana_R1"),
    }

    # Holm sobre as 4 comparações do H1
    ajustados = holm({k: v["teste_erro_absoluto"]["Wilcoxon_p"] for k, v in res["Comparacoes"].items()})
    for k, p in ajustados.items():
        t = res["Comparacoes"][k]["teste_erro_absoluto"]
        t["Wilcoxon_p_Holm"] = p
        t["significativo_Holm"] = bool(not np.isnan(p) and p < ALFA)
    return res


# ==============================================================================
# 4. RQ2.1 / H2 — ESPECIALIZAÇÃO DE PAPÉIS
# ==============================================================================

def comparar_condicoes(df: pd.DataFrame, cond_a: str, cond_b: str) -> dict:
    a, b = pareados(df, cond_a, cond_b)
    if a.empty:
        return {"status": "sem_pares", "A": cond_a, "B": cond_b}
    y = a["ground_truth"].astype(float).to_numpy()
    if not np.allclose(y, b["ground_truth"].astype(float).to_numpy()):
        return {"status": "ground_truth_divergente", "A": cond_a, "B": cond_b}
    pa, pb = a["predito"].astype(float).to_numpy(), b["predito"].astype(float).to_numpy()
    return {
        "n_itens": int(len(a)),
        cond_a: metricas(y, pa),
        cond_b: metricas(y, pb),
        "teste_erro_absoluto": comparar(np.abs(y - pa), np.abs(y - pb), cond_a, cond_b),
        "nota_A12": NOTA_A12,
    }


def analisar_rq2_1_h2(df: pd.DataFrame) -> dict:
    return {
        "principal_simultaneo": comparar_condicoes(df, COND_SIM_ESP, COND_SIM_NEU),
        "secundaria_sequencial": comparar_condicoes(df, COND_SEQ_ESP, COND_SEQ_NEU),
    }


# ==============================================================================
# 5. RQ2.2 / H3 — BASELINE DE APRENDIZADO DE MÁQUINA
# ==============================================================================

ROSSI_PUBLICADO = {"MAE": 3.83, "PRED(25)": 35.92, "fonte": "Rossi (2026), Tabelas 4/5 — outro conjunto de teste"}


def analisar_rq2_2_h3(df: pd.DataFrame, caminho_baseline: Optional[str] = None,
                      condicao: str = COND_SIM_ESP) -> dict:
    """
    Com `caminho_baseline` (CSV com issue_id ou IID e coluna `predito`), compara item a item
    nas issues comuns. Sem ele, só reporta o valor publicado, marcado como NÃO pareado.
    """
    d = por_condicao(df, condicao)
    if d.empty:
        return {"status": "sem_dados", "condicao": condicao}
    if not caminho_baseline:
        return {
            "status": "nao_pareado",
            "aviso": "Sem predições do baseline por issue: H3 não pode ser testada estatisticamente.",
            "MASPE": metricas(d["ground_truth"].astype(float), d["predito"].astype(float)),
            "Rossi_publicado": ROSSI_PUBLICADO,
        }
    base = pd.read_csv(caminho_baseline, dtype=str)
    col_id = "issue_id" if "issue_id" in base.columns else "IID"
    if col_id not in base.columns or "predito" not in base.columns:
        raise KeyError("O CSV do baseline precisa das colunas issue_id (ou IID) e predito.")
    base = base.rename(columns={col_id: "issue_id"}).drop_duplicates("issue_id").set_index("issue_id")
    comuns = d.index.intersection(base.index)
    if len(comuns) == 0:
        return {"status": "sem_issues_comuns"}
    y = d.loc[comuns, "ground_truth"].astype(float).to_numpy()
    p_maspe = d.loc[comuns, "predito"].astype(float).to_numpy()
    p_base = pd.to_numeric(base.loc[comuns, "predito"], errors="coerce").to_numpy(dtype=float)
    ok = ~np.isnan(p_base)
    return {
        "n_itens": int(ok.sum()),
        "MASPE": metricas(y[ok], p_maspe[ok]),
        "SBERT_LightGBM": metricas(y[ok], p_base[ok]),
        "teste_erro_absoluto": comparar(np.abs(y[ok] - p_maspe[ok]), np.abs(y[ok] - p_base[ok]), "MASPE", "SBERT_LightGBM"),
        "nota_A12": NOTA_A12.replace("de A", "do MASPE"),
    }


# ==============================================================================
# 6. RQ3 / H4 — ANCORAGEM SINTÉTICA E TRAJETÓRIA
# ==============================================================================

def ancora_r1(row, protocolo: str) -> Tuple[float, Optional[str]]:
    """
    Âncora FIXA da Rodada 1 e o agente que a emitiu:
      sequencial -> v1_R1, o primeiro lance revelado na R1;
      simultâneo -> Mediana_R1, sem agente-âncora.
    """
    r1 = lances(row, 1)
    if protocolo == "sequencial":
        com_ordem = [e for e in r1 if e.get("ordem_revelacao")]
        primeiro = min(com_ordem, key=lambda e: e["ordem_revelacao"]) if com_ordem else r1[0]
        return float(primeiro["story_points"]), primeiro["agente"]
    return mediana_r1(row), None


def indicadores_ancoragem(d: pd.DataFrame, protocolo: str) -> dict:
    """
    Para cada sessão, com a âncora fixa da R1:
      deslocamento_{i,t} = |voto_{i,t} − Âncora_R1|. No sequencial, o agente-âncora fica fora
        só na R1 (distância zero por construção) e entra a partir da R2;
      (i)  revisão = voto_{i,t+1} ≠ voto_{i,t}; aproxima-se se o deslocamento diminui.
           Inclui as revisões do agente-âncora (reportadas também à parte);
      (ii) magnitude por sessão = média, entre os agentes, de Σ_t |voto_{i,t+1} − voto_{i,t}|
           (sessões encerradas na R1 têm magnitude 0).
    """
    revisados = aproximaram = revisados_ancora = 0
    magnitude, desloc_r1, desloc_final = {}, {}, {}
    trajetoria: Dict[int, List[float]] = {}

    for issue_id, row in d.iterrows():
        ancora, agente_ancora = ancora_r1(row, protocolo)
        votos = votos_por_rodada(row)
        rodadas = sorted(votos)
        agentes = list(votos[1])

        soma_delta = {ag: 0.0 for ag in agentes}
        for t, t_seg in zip(rodadas, rodadas[1:]):
            for ag in agentes:
                if ag not in votos[t] or ag not in votos[t_seg]:
                    continue
                antes, depois = votos[t][ag], votos[t_seg][ag]
                soma_delta[ag] += abs(depois - antes)
                if depois != antes:
                    revisados += 1
                    revisados_ancora += int(ag == agente_ancora)
                    if abs(depois - ancora) < abs(antes - ancora):
                        aproximaram += 1
        magnitude[issue_id] = sum(soma_delta.values()) / max(len(agentes), 1)

        for t in rodadas:
            participantes = [ag for ag in votos[t] if not (t == 1 and ag == agente_ancora)]
            ds = [abs(votos[t][ag] - ancora) for ag in participantes]
            if not ds:
                continue
            media = float(np.mean(ds))
            trajetoria.setdefault(t, []).append(media)
            if t == 1:
                desloc_r1[issue_id] = media
            if t == rodadas[-1]:
                desloc_final[issue_id] = media

    return {
        "i_revisoes": {
            "n_lances_revisados": revisados,
            "n_aproximaram_ancora": aproximaram,
            "proporcao_aproximacao": aproximaram / revisados if revisados else float("nan"),
            "n_revisoes_do_agente_ancora": revisados_ancora,
        },
        "magnitude_alteracao": pd.Series(magnitude, dtype=float),
        "deslocamento_R1": pd.Series(desloc_r1, dtype=float),
        "deslocamento_final": pd.Series(desloc_final, dtype=float),
        "trajetoria": {
            f"R{t}": {"n_sessoes": len(v), "deslocamento_medio": float(np.mean(v))}
            for t, v in sorted(trajetoria.items())
        },
    }


def comparar_protocolos(df: pd.DataFrame, cond_sim: str, cond_seq: str) -> dict:
    a, b = pareados(df, cond_sim, cond_seq)
    if a.empty:
        return {"status": "sem_pares", "A": cond_seq, "B": cond_sim}
    ind_sim = indicadores_ancoragem(a, "simultaneo")
    ind_seq = indicadores_ancoragem(b, "sequencial")
    y = a["ground_truth"].astype(float).to_numpy()
    p_sim, p_seq = a["predito"].astype(float).to_numpy(), b["predito"].astype(float).to_numpy()

    def _par(chave: str) -> dict:
        s, q = ind_sim[chave], ind_seq[chave]
        comuns = s.index.intersection(q.index)
        return comparar(q.loc[comuns].to_numpy(), s.loc[comuns].to_numpy(), cond_seq, cond_sim)

    return {
        "n_itens": int(len(a)),
        "ancoras": {cond_seq: "v1_R1 (1º lance revelado na R1)", cond_sim: "Mediana_R1"},
        "i_proporcao_revisoes_rumo_a_ancora": {
            cond_seq: ind_seq["i_revisoes"],
            cond_sim: ind_sim["i_revisoes"],
            "nota": "indicador de processo (descritivo)",
        },
        "ii_magnitude_alteracao_lances": _par("magnitude_alteracao"),
        "iii_erro_absoluto_consenso": comparar(np.abs(y - p_seq), np.abs(y - p_sim), cond_seq, cond_sim),
        "metricas": {cond_seq: metricas(y, p_seq), cond_sim: metricas(y, p_sim)},
        "trajetoria_deslocamento": {cond_seq: ind_seq["trajetoria"], cond_sim: ind_sim["trajetoria"]},
        "aux_deslocamento_R1": _par("deslocamento_R1"),
        "aux_deslocamento_final": _par("deslocamento_final"),
        "nota_A12": "A = sequencial, B = simultâneo. " + NOTA_A12.replace("de A", "do sequencial"),
    }


def analisar_rq3_h4(df: pd.DataFrame) -> dict:
    return {
        "principal_especializado": comparar_protocolos(df, COND_SIM_ESP, COND_SEQ_ESP),
        "secundaria_neutro": comparar_protocolos(df, COND_SIM_NEU, COND_SEQ_NEU),
    }


# ==============================================================================
# 7. RQ4 / H5 — DINÂMICA DE CONVERGÊNCIA
# ==============================================================================

def classificar_encerramento(motivo: Optional[str]) -> str:
    """Motivo do consensus.py -> categoria do critério operacional."""
    motivo = motivo or ""
    if motivo.startswith("consenso_unanime_r"):
        return "unanimidade_r" + motivo.removeprefix("consenso_unanime_r")[:1]
    if motivo.startswith("estagnacao_r2"):
        return "estagnacao_r2"
    if motivo.startswith("desempate_mediana_r3"):
        return "teto_r3_mediana"
    return "outro"


def dispersao_por_rodada(row) -> Dict[int, Tuple[float, int]]:
    """{rodada: (σ_t dos lances, amplitude em degraus de Fibonacci)}."""
    out = {}
    for r in rodadas_da_sessao(row):
        vals = [e["story_points"] for e in lances(row, r)]
        out[r] = (desvio_padrao_sp(vals), calcular_dispersao_degraus(vals, ESCALA_SIS_ASTROS))
    return out


def analisar_rq4_h5(df: pd.DataFrame, condicao: str = COND_SIM_ESP) -> dict:
    d = por_condicao(df, condicao)
    if d.empty:
        return {"status": "sem_dados", "condicao": condicao}

    n = len(d)
    encerramento = d["motivo_convergencia"].map(classificar_encerramento)
    disp = {i: dispersao_por_rodada(r) for i, r in d.iterrows()}

    # Curva descritiva. Atenção: viés de sobrevivência — só os itens com divergência chegam à R2/R3.
    curva = {}
    for r in (1, 2, 3):
        s = [v[r][0] for v in disp.values() if r in v]
        g = [v[r][1] for v in disp.values() if r in v]
        curva[f"R{r}"] = {
            "n_itens_que_chegaram": len(s),
            "sigma_medio": float(np.mean(s)) if s else float("nan"),
            "sigma_mediano": float(np.median(s)) if s else float("nan"),
            "amplitude_degraus_media": float(np.mean(g)) if g else float("nan"),
        }

    # Redução de σ_t entre rodadas consecutivas, pareada nos mesmos itens
    transicoes = {}
    for r in (1, 2):
        pares = [(v[r][0], v[r + 1][0]) for v in disp.values() if r in v and r + 1 in v]
        rotulo = f"R{r}->R{r + 1}"
        if not pares:
            transicoes[rotulo] = {"n_itens": 0}
            continue
        antes, depois = (np.array(x) for x in zip(*pares))
        t = comparar(depois, antes, f"sigma_R{r + 1}", f"sigma_R{r}")
        t.update({
            "n_itens": len(pares),
            "pct_diminuiu": float(np.mean(depois < antes) * 100),
            "pct_igual": float(np.mean(depois == antes) * 100),
            "pct_aumentou": float(np.mean(depois > antes) * 100),
        })
        transicoes[rotulo] = t

    # Progressividade nos itens com debate: σ_t nunca aumenta e termina menor que na R1
    com_debate = [v for v in disp.values() if len(v) >= 2]
    progressivo = [
        all(v[r_seg][0] <= v[r][0] for r, r_seg in zip(sorted(v), sorted(v)[1:])) and v[max(v)][0] < v[1][0]
        for v in com_debate
    ]

    return {
        "condicao": condicao,
        "n_itens": n,
        "encerramento": {k: {"n": int(c), "pct": c / n * 100} for k, c in encerramento.value_counts().items()},
        "taxa_unanimidade_pct": float(encerramento.str.startswith("unanimidade").mean() * 100),
        "rodadas_usadas": {f"R{int(k)}": int(c) for k, c in d["rodadas_usadas"].value_counts().sort_index().items()},
        "rodadas_media": float(d["rodadas_usadas"].mean()),
        "curva_sigma": curva,
        "transicoes_sigma": transicoes,
        "n_itens_com_debate": len(com_debate),
        "pct_itens_com_debate_sigma_progressivamente_menor": float(np.mean(progressivo) * 100) if progressivo else float("nan"),
        "nota_A12": "Nas transições, A = σ da rodada seguinte, B = σ da anterior: Â12 < 0,50 indica redução da dispersão.",
    }


# ==============================================================================
# 8. RQ4 / H6 — AMOSTRA QUALITATIVA E KAPPA DE COHEN
# ==============================================================================

def gerar_amostra_h6(
    df: pd.DataFrame,
    condicao: str = COND_SIM_ESP,
    n_por_grupo: int = 5,
    seed: int = 42,
    caminho_saida: Optional[str] = None,
) -> pd.DataFrame:
    """
    5 sessões unânimes na R1 + 5 com o maior nº de rodadas (sorteio com seed fixa em empates).
    Unidade de codificação: UMA LINHA POR JUSTIFICATIVA CoT (agente x rodada), com ΔSP do agente
    em relação à rodada anterior, sua direção e o ΔSP total do agente (lance final − lance da R1),
    mais colunas vazias para a codificação dos dois avaliadores.
    """
    d = por_condicao(df, condicao)
    if d.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    encerramento = d["motivo_convergencia"].map(classificar_encerramento)

    unanimes = sorted(d.index[encerramento == "unanimidade_r1"])
    grupo_r1 = list(rng.choice(unanimes, size=min(n_por_grupo, len(unanimes)), replace=False)) if unanimes else []

    grupo_debate: List[str] = []
    for r in sorted(d["rodadas_usadas"].unique(), reverse=True):
        if r <= 1 or len(grupo_debate) >= n_por_grupo:
            break
        candidatos = sorted(d.index[d["rodadas_usadas"] == r])
        faltam = n_por_grupo - len(grupo_debate)
        grupo_debate += list(rng.choice(candidatos, size=min(faltam, len(candidatos)), replace=False))

    linhas = []
    for grupo, ids in (("unanime_R1", grupo_r1), ("mais_rodadas", grupo_debate)):
        for issue_id in ids:
            row = d.loc[issue_id]
            votos = votos_por_rodada(row)
            final = {ag: votos[max(r for r in votos if ag in votos[r])][ag] for ag in votos[1]}
            for e in sorted(row["estimativas"], key=lambda e: (e["rodada"], e["agente"])):
                anterior = votos.get(e["rodada"] - 1, {}).get(e["agente"])
                delta = None if anterior is None else e["story_points"] - anterior
                linhas.append({
                    "issue_id": issue_id,
                    "grupo": grupo,
                    "rodadas_usadas": int(row["rodadas_usadas"]),
                    "ground_truth": row["ground_truth"],
                    "consenso": row["predito"],
                    "rodada": e["rodada"],
                    "agente": e["agente"],
                    "papel": e["papel"],
                    "story_points": e["story_points"],
                    "delta_sp": delta,
                    "direcao_delta_sp": None if delta is None else ("+" if delta > 0 else "-" if delta < 0 else "0"),
                    "delta_sp_total_agente": final[e["agente"]] - votos[1][e["agente"]],
                    "justificativa": e["justificativa"],
                    "fator_principal": "",
                    "fatores_secundarios": "",
                })

    amostra = pd.DataFrame(linhas)
    if caminho_saida:
        Path(caminho_saida).parent.mkdir(parents=True, exist_ok=True)
        amostra.to_csv(caminho_saida, index=False, encoding="utf-8-sig")
    return amostra


def concordancia_h6(arquivo_a: str, arquivo_b: str, coluna: str = "fator_principal") -> dict:
    """
    Kappa de Cohen (sklearn) entre as planilhas dos dois avaliadores.
    Unidade: justificativa CoT individual (chave issue_id + rodada + agente).
    """
    chave = ["issue_id", "rodada", "agente"]
    a = pd.read_csv(arquivo_a, dtype=str).set_index(chave)[coluna].fillna("").str.strip().str.lower()
    b = pd.read_csv(arquivo_b, dtype=str).set_index(chave)[coluna].fillna("").str.strip().str.lower()
    comuns = a.index.intersection(b.index)
    if len(comuns) == 0:
        raise ValueError("As planilhas dos avaliadores não têm justificativas em comum.")
    va, vb = a.loc[comuns].tolist(), b.loc[comuns].tolist()
    return {
        "unidade": "justificativa CoT individual (agente x rodada)",
        "n_justificativas": len(comuns),
        "kappa_cohen": float(cohen_kappa_score(va, vb)),
        "concordancia_bruta_pct": float(np.mean([x == y for x, y in zip(va, vb)]) * 100),
        "n_divergencias": int(sum(x != y for x, y in zip(va, vb))),
    }


# ==============================================================================
# 9. RELATÓRIO E TABELA LATEX
# ==============================================================================

def _fmt(v, casas: int = 2) -> str:
    return "---" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{casas}f}"


def gerar_tabela_latex_rq1(res: dict) -> str:
    """Tabela da RQ1/H1 com a nota de leitura do Â12 na legenda."""
    if "Consenso" not in res:
        return "% Sem dados para a RQ1"
    c = res["Consenso"]
    linhas = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\small",
        "  \\caption{RQ1/H1: consenso multiagente vs. lances individuais e agregação inicial. "
        "Nota: como a variável analisada é o erro absoluto (onde menor é melhor), valores de "
        "$\\hat{A}_{12} < 0{,}50$ indicam a superioridade estatística do consenso "
        "(probabilidade de apresentar erro menor que a abordagem comparada).}",
        "  \\label{tab:resultado_rq1}",
        "  \\begin{tabular}{lccccccc}",
        "    \\toprule",
        "    \\textbf{Abordagem} & \\textbf{MAE} & \\textbf{MdAE} & \\textbf{PRED(25)} & \\textbf{$r_s$} & "
        "\\textbf{$p$ (Holm)} & \\textbf{$\\hat{A}_{12}$} & \\textbf{Magnitude} \\\\",
        "    \\midrule",
        f"    \\textbf{{Consenso}} & \\textbf{{{_fmt(c['MAE'])}}} & \\textbf{{{_fmt(c['MdAE'])}}} & "
        f"\\textbf{{{_fmt(c['PRED(25)'], 1)}\\%}} & \\textbf{{{_fmt(c['Spearman_rs'], 3)}}} & --- & --- & --- \\\\",
    ]
    for nome, comp in res["Comparacoes"].items():
        m, t = comp["metricas"], comp["teste_erro_absoluto"]
        rotulo = (nome.replace("Individual_R1[", "Lance individual R1 (").replace("]", ")")
                  .replace("Mediana_R1", "Mediana dos lances R1"))
        linhas.append(
            f"    {rotulo} & {_fmt(m['MAE'])} & {_fmt(m['MdAE'])} & {_fmt(m['PRED(25)'], 1)}\\% & "
            f"{_fmt(m['Spearman_rs'], 3)} & {_fmt(t.get('Wilcoxon_p_Holm'), 3)} & {_fmt(t['A12'])} & {t['A12_magnitude']} \\\\"
        )
    linhas += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    return "\n".join(linhas)


def _serializavel(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Series):
        return o.to_dict()
    return str(o)


def gerar_relatorio_completo(
    padrao_raw: str = "results/raw/*.jsonl",
    projeto: str = "SIS-ASTROS",
    modelo: Optional[str] = None,
    modo_contexto: Optional[str] = None,
    caminho_baseline: Optional[str] = None,
) -> Optional[dict]:
    df = carregar_resultados(padrao_raw, modelo=modelo, modo_contexto=modo_contexto, projeto=projeto)
    if df.empty:
        print("[AVISO] Nenhum resultado em", padrao_raw, "| modelo =", modelo, "| contexto =", modo_contexto)
        return None

    modelos = sorted(df["modelo"].dropna().unique()) if "modelo" in df else []
    modos = sorted(df["modo_contexto"].dropna().unique()) if "modo_contexto" in df else []
    if len(modelos) > 1 or len(modos) > 1:
        print(f"[AVISO] Execuções misturadas (modelos={modelos}, contextos={modos}). Use --modelo e --contexto.")

    relatorio = {
        "filtros": {"projeto": projeto, "modelo": modelo, "modo_contexto": modo_contexto,
                    "wilcoxon": f"pareado, {ALTERNATIVA}, zero_method=pratt", "alfa": ALFA,
                    "leitura_A12": NOTA_A12},
        "itens_por_condicao": {c: int(len(por_condicao(df, c))) for c in CONDICOES},
        "RQ1_H1": analisar_rq1_h1(df),
        "RQ2_1_H2": analisar_rq2_1_h2(df),
        "RQ2_2_H3": analisar_rq2_2_h3(df, caminho_baseline),
        "RQ3_H4": analisar_rq3_h4(df),
        "RQ4_H5": {c: analisar_rq4_h5(df, c) for c in CONDICOES},
    }
    print(json.dumps(relatorio, indent=2, ensure_ascii=False, default=_serializavel))

    sufixo = "".join(f"__{s}" for s in (modelo, modo_contexto) if s).replace("/", "_").replace(":", "-")
    Path("results/analysis").mkdir(parents=True, exist_ok=True)
    Path("results/latex").mkdir(parents=True, exist_ok=True)
    with open(f"results/analysis/relatorio{sufixo}.json", "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False, default=_serializavel)
    with open(f"results/latex/tabela_rq1{sufixo}.tex", "w", encoding="utf-8") as f:
        f.write(gerar_tabela_latex_rq1(relatorio["RQ1_H1"]))
    print(f"\nRelatório: results/analysis/relatorio{sufixo}.json | Tabela: results/latex/tabela_rq1{sufixo}.tex")
    return relatorio


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Relatório estatístico do MASPE (RQ1–RQ4).")
    ap.add_argument("--padrao", default="results/raw/*.jsonl", help="Glob dos JSONL do runner")
    ap.add_argument("--projeto", default="SIS-ASTROS")
    ap.add_argument("--modelo", default=None, help="Filtra por modelo (mesmo nome passado ao runner)")
    ap.add_argument("--contexto", default=None, choices=["rag", "few_shot"], help="Filtra por modo de contexto")
    ap.add_argument("--baseline-rossi", default=None, help="CSV com issue_id/IID e predito do SBERT+LightGBM")
    ap.add_argument("--amostra-h6", default=None, metavar="SAIDA.csv",
                    help="Exporta a amostra qualitativa do H6 (10 sessões) e encerra")
    ap.add_argument("--kappa", nargs=2, default=None, metavar=("AVALIADOR_A.csv", "AVALIADOR_B.csv"),
                    help="Calcula o Kappa de Cohen entre as codificações do H6 e encerra")
    args = ap.parse_args()

    if args.kappa:
        print(json.dumps(concordancia_h6(*args.kappa), indent=2, ensure_ascii=False))
    elif args.amostra_h6:
        dados = carregar_resultados(args.padrao, modelo=args.modelo, modo_contexto=args.contexto, projeto=args.projeto)
        amostra = gerar_amostra_h6(dados, caminho_saida=args.amostra_h6)
        n_sessoes = amostra["issue_id"].nunique() if not amostra.empty else 0
        print(f"Amostra H6: {n_sessoes} sessões, {len(amostra)} justificativas -> {args.amostra_h6}")
    else:
        gerar_relatorio_completo(args.padrao, args.projeto, args.modelo, args.contexto, args.baseline_rossi)
