"""
tests/test_analysis.py
======================
Testes da análise estatística (src/analysis.py) com sessões sintéticas de resultado conhecido.

Execução (a partir da pasta multi-llm-agent-planning-poker):
    python -m unittest tests.test_analysis -v
"""

import itertools
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import analysis as an


def _est(agente, papel, sp, rodada, ordem=None, justificativa="j"):
    return {"agente": agente, "papel": papel, "modelo": "m", "story_points": float(sp),
            "justificativa": justificativa, "rodada": rodada, "ordem_revelacao": ordem}


def _sessao(issue_id, condicao, gt, predito, rodadas_votos, motivo, protocolo="simultaneo", papeis=("backend", "frontend", "qa")):
    """rodadas_votos: lista por rodada de 3 votos (agentes a1, a2, a3, nessa ordem de revelação)."""
    ests = []
    for r, votos in enumerate(rodadas_votos, start=1):
        for k, (sp, papel) in enumerate(zip(votos, papeis)):
            ordem = k + 1 if protocolo == "sequencial" else None
            ests.append(_est(f"a{k + 1}", papel, sp, r, ordem))
    return {"issue_id": str(issue_id), "condicao": condicao, "projeto": "SIS-ASTROS", "modelo": "m",
            "modo_contexto": "rag", "ground_truth": float(gt), "predito": float(predito),
            "rodadas_usadas": len(rodadas_votos), "convergiu": not motivo.startswith("desempate"),
            "motivo_convergencia": motivo, "estimativas": ests}


class TestEstatisticas(unittest.TestCase):

    def test_a12_produto_cartesiano(self):
        rng = np.random.default_rng(0)
        x, y = rng.integers(0, 10, 15), rng.integers(0, 10, 11)
        esperado = np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a, b in itertools.product(x, y)])
        self.assertAlmostEqual(an.a12(x, y), esperado, places=12)

    def test_a12_extremos_e_diferente_do_pareado(self):
        self.assertEqual(an.a12([5, 6, 7], [1, 2, 3]), 1.0)
        self.assertEqual(an.a12([1, 2, 3], [5, 6, 7]), 0.0)
        self.assertEqual(an.a12([2, 2], [2, 2, 2]), 0.5)
        # Versão antiga (item a item): x > y em 1 de 3 pares -> 1/3.
        # Vargha-Delaney (todos os 9 pares): x > y em 5 pares -> 5/9.
        x, y = [1, 2, 9], [3, 4, 0]
        self.assertAlmostEqual(an.a12(x, y), 5 / 9)

    def test_magnitude(self):
        self.assertEqual(an.magnitude_a12(0.52), "desprezivel")
        self.assertEqual(an.magnitude_a12(0.40), "pequeno")
        self.assertEqual(an.magnitude_a12(0.33), "medio")
        self.assertEqual(an.magnitude_a12(0.20), "grande")

    def test_wilcoxon_identicos_e_pratt(self):
        self.assertEqual(an.teste_wilcoxon([1, 2, 3], [1, 2, 3])["p"], 1.0)
        r = an.teste_wilcoxon([0, 0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(r["n_pares_nao_nulos"], 7)
        self.assertLess(r["p"], 0.05)

    def test_holm(self):
        aj = an.holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(aj["a"], 0.03)
        self.assertAlmostEqual(aj["c"], 0.06)
        self.assertAlmostEqual(aj["b"], 0.06)  # monotonicidade

    def test_mdae_e_pred(self):
        self.assertEqual(an.mdae([1, 2, 8], [1, 3, 13]), 1.0)
        # Na escala Fibonacci, PRED(25) = acerto exato
        self.assertAlmostEqual(an.pred([8, 13, 21], [8, 8, 34]), 100 / 3)


class TestAncoragem(unittest.TestCase):

    def test_sequencial_ancora_v1_e_agente_ancora_so_fora_da_r1(self):
        s = _sessao(1, an.COND_SEQ_ESP, 8, 8, [[8, 5, 13], [5, 8, 8], [8, 8, 8]],
                    "consenso_unanime_r3", protocolo="sequencial")
        d = pd.DataFrame([s]).set_index("issue_id")
        ind = an.indicadores_ancoragem(d, "sequencial")
        # R1->R2: a1 8->5 (âncora, afasta), a2 5->8 (aproxima), a3 13->8 (aproxima); R2->R3: a1 5->8 (aproxima)
        self.assertEqual(ind["i_revisoes"]["n_lances_revisados"], 4)
        self.assertEqual(ind["i_revisoes"]["n_aproximaram_ancora"], 3)
        self.assertEqual(ind["i_revisoes"]["n_revisoes_do_agente_ancora"], 2)
        # R1 sem o agente-âncora: (|5-8| + |13-8|)/2 = 4; R2 com todos: (3+0+0)/3 = 1
        self.assertAlmostEqual(ind["deslocamento_R1"]["1"], 4.0)
        self.assertAlmostEqual(ind["trajetoria"]["R2"]["deslocamento_medio"], 1.0)
        self.assertAlmostEqual(ind["deslocamento_final"]["1"], 0.0)
        self.assertAlmostEqual(ind["magnitude_alteracao"]["1"], (3 + 3 + 3 + 5) / 3)

    def test_simultaneo_ancora_mediana(self):
        s = _sessao(1, an.COND_SIM_ESP, 8, 5, [[3, 5, 13], [5, 5, 8]], "estagnacao_r2_mediana")
        d = pd.DataFrame([s]).set_index("issue_id")
        self.assertEqual(an.ancora_r1(d.iloc[0], "simultaneo"), (5.0, None))
        ind = an.indicadores_ancoragem(d, "simultaneo")
        # a1 3->5 aproxima (2->0); a3 13->8 aproxima (8->3)
        self.assertEqual(ind["i_revisoes"]["n_aproximaram_ancora"], 2)
        self.assertAlmostEqual(ind["deslocamento_R1"]["1"], (2 + 0 + 8) / 3)


def _conjunto_sintetico():
    """Mesmas 12 issues nas 4 condições, com perfis distintos."""
    linhas = []
    gts = [1, 2, 3, 5, 8, 13, 21, 34, 5, 8, 13, 3]
    for i, gt in enumerate(gts):
        # simultâneo especializado: 6 unânimes na R1, 6 com 3 rodadas
        if i < 6:
            linhas.append(_sessao(i, an.COND_SIM_ESP, gt, gt, [[gt] * 3], "consenso_unanime_r1"))
        else:
            linhas.append(_sessao(i, an.COND_SIM_ESP, gt, gt, [[2, gt, 34], [gt, gt, 21], [gt, gt, 13]],
                                  "desempate_mediana_r3 (delta=1 degraus)"))
        errado = 55 if gt < 21 else 1
        linhas.append(_sessao(i, an.COND_SEQ_ESP, gt, errado, [[errado, errado, gt], [errado] * 3],
                              "consenso_unanime_r2", protocolo="sequencial"))
        linhas.append(_sessao(i, an.COND_SIM_NEU, gt, errado, [[errado] * 3], "consenso_unanime_r1",
                              papeis=("neutro",) * 3))
        linhas.append(_sessao(i, an.COND_SEQ_NEU, gt, errado, [[errado] * 3], "consenso_unanime_r1",
                              protocolo="sequencial", papeis=("neutro",) * 3))
    return pd.DataFrame(linhas)


class TestRelatorio(unittest.TestCase):

    def setUp(self):
        self.df = _conjunto_sintetico()

    def test_rq1(self):
        r = an.analisar_rq1_h1(self.df)
        self.assertEqual(r["n_itens"], 12)
        self.assertEqual(r["Consenso"]["MAE"], 0.0)
        self.assertIn("Individual_R1[qa]", r["Comparacoes"])
        self.assertIn("Wilcoxon_p_Holm", r["Comparacoes"]["Mediana_R1"]["teste_erro_absoluto"])
        self.assertLess(r["Comparacoes"]["Individual_R1[qa]"]["teste_erro_absoluto"]["A12"], 0.5)

    def test_rq2_e_rq3(self):
        h2 = an.analisar_rq2_1_h2(self.df)["principal_simultaneo"]
        self.assertLess(h2["teste_erro_absoluto"]["A12"], 0.5)  # especializado erra menos
        h4 = an.analisar_rq3_h4(self.df)["principal_especializado"]
        self.assertGreater(h4["iii_erro_absoluto_consenso"]["A12"], 0.5)  # sequencial erra mais
        self.assertIn("trajetoria_deslocamento", h4)

    def test_rq4_h5(self):
        h5 = an.analisar_rq4_h5(self.df)
        self.assertEqual(h5["encerramento"]["unanimidade_r1"]["n"], 6)
        self.assertEqual(h5["encerramento"]["teto_r3_mediana"]["n"], 6)
        self.assertEqual(h5["transicoes_sigma"]["R1->R2"]["n_itens"], 6)
        self.assertEqual(h5["transicoes_sigma"]["R1->R2"]["pct_diminuiu"], 100.0)
        self.assertEqual(h5["curva_sigma"]["R3"]["n_itens_que_chegaram"], 6)

    def test_h6_amostra(self):
        amostra = an.gerar_amostra_h6(self.df)
        self.assertEqual(amostra["issue_id"].nunique(), 10)
        self.assertEqual(len(amostra), 5 * 3 + 5 * 9)  # 60 justificativas
        # Kappa do H6 calculado manualmente: concordancia_h6 está comentada em src/analysis.py.
        # with tempfile.TemporaryDirectory() as pasta:
        #     a, b = Path(pasta) / "a.csv", Path(pasta) / "b.csv"
        #     ca = amostra.assign(fator_principal=["complexidade" if i % 2 else "volume" for i in range(len(amostra))])
        #     cb = ca.copy()
        #     cb.loc[cb.index[:6], "fator_principal"] = "risco"
        #     ca.to_csv(a, index=False)
        #     cb.to_csv(b, index=False)
        #     k = an.concordancia_h6(str(a), str(b))
        # self.assertEqual(k["n_justificativas"], 60)
        # self.assertEqual(k["n_divergencias"], 6)
        # self.assertTrue(0 < k["kappa_cohen"] < 1)

    def test_erros_vs_time_spent(self):
        casos = [  # (ground truth, predito, Time Spent)
            (8, 5, 5.8333),  # exemplo: modelo mais próximo (0,83 vs 2,17)
            (5, 8, 5.0),     # GT mais próximo
            (3, 5, 4.0),     # empate (1 vs 1)
            (8, 13, None),   # erro sem Time Spent
            (8, 13, 0.0),    # Time Spent zero = sem registro
            (8, 8, 7.0),     # acerto: fora da análise de erros
        ]
        df = pd.DataFrame([
            {**_sessao(i, an.COND_SIM_ESP, gt, p, [[p] * 3], "consenso_unanime_r1"), "esforco_real_horas": ts}
            for i, (gt, p, ts) in enumerate(casos)
        ])
        t = an.tabela_erros_vs_esforco(df).set_index("issue_id")
        self.assertEqual(t.loc["0", "mais_proximo"], "modelo")
        self.assertAlmostEqual(t.loc["0", "dist_predito_time_spent"], 0.8333)
        self.assertAlmostEqual(t.loc["0", "dist_gt_time_spent"], 2.1667)
        self.assertEqual(t.loc["1", "mais_proximo"], "ground_truth")
        self.assertEqual(t.loc["2", "mais_proximo"], "empate")
        self.assertEqual(t.loc["3", "mais_proximo"], "sem_time_spent")
        self.assertEqual(t.loc["4", "mais_proximo"], "sem_time_spent")
        self.assertNotIn("5", t.index)

        r = an.analisar_erros_vs_esforco(df)
        self.assertEqual((r["n_itens"], r["n_acertos"], r["n_erros"]), (6, 1, 5))
        self.assertEqual((r["n_erros_sem_time_spent"], r["n_erros_analisados"]), (2, 3))
        self.assertEqual(r["mais_proximo_do_time_spent"]["modelo"]["n"], 1)
        self.assertEqual(r["mais_proximo_do_time_spent"]["ground_truth"]["n"], 1)
        self.assertEqual(r["mais_proximo_do_time_spent"]["empate"]["n"], 1)
        self.assertEqual(r["teste_distancia_time_spent"]["n_pares"], 3)

    def test_erros_vs_time_spent_resultados_sem_a_coluna(self):
        r = an.analisar_erros_vs_esforco(self.df, an.COND_SEQ_ESP)
        self.assertEqual(r["n_erros"], 12)
        self.assertEqual(r["n_erros_analisados"], 0)
        self.assertEqual(r["status"], "sem_erros_com_time_spent")

    def test_relatorio_completo(self):
        with tempfile.TemporaryDirectory() as pasta:
            raw = Path(pasta) / "raw"
            raw.mkdir()
            with open(raw / "SIS-ASTROS__x__m__rag.jsonl", "w", encoding="utf-8") as f:
                for _, row in self.df.iterrows():
                    f.write(json.dumps(row.to_dict()) + "\n")
            (raw / "SIS-ASTROS__x__m__rag__falhas.jsonl").write_text('{"issue_id": "99"}\n', encoding="utf-8")
            cwd = os.getcwd()
            os.chdir(pasta)
            try:
                rel = an.gerar_relatorio_completo(str(raw / "*.jsonl"), modelo="m", modo_contexto="rag")
                self.assertTrue(Path("results/latex/tabela_rq1__m__rag.tex").exists())
                self.assertTrue(Path("results/analysis/erros_vs_time_spent__m__rag.csv").exists())
                for nome in ("rq2_1_h2", "rq2_2_h3", "rq3_h4", "rq3_h4_trajetoria", "rq4_h5_encerramento",
                             "rq4_h5_curva", "rq4_h5_transicoes", "erros_time_spent"):
                    self.assertTrue(Path(f"results/latex/tabela_{nome}__m__rag.tex").exists(), nome)
                todas = Path("results/latex/tabelas_todas__m__rag.tex").read_text(encoding="utf-8")
                self.assertEqual(todas.count("\\begin{table}"), 9)
            finally:
                os.chdir(cwd)
        self.assertEqual(rel["itens_por_condicao"][an.COND_SIM_ESP], 12)

    def _relatorio_de(self, df):
        return {
            "RQ1_H1": an.analisar_rq1_h1(df),
            "RQ2_1_H2": an.analisar_rq2_1_h2(df),
            "RQ2_2_H3": an.analisar_rq2_2_h3(df),
            "RQ3_H4": an.analisar_rq3_h4(df),
            "RQ4_H5": {c: an.analisar_rq4_h5(df, c) for c in an.CONDICOES},
            "Erros_vs_TimeSpent": {c: an.analisar_erros_vs_esforco(df, c) for c in an.CONDICOES},
        }

    def test_tabelas_latex_conteudo(self):
        tabelas = an.gerar_tabelas_latex(self._relatorio_de(self.df))
        for nome, tabela in tabelas.items():
            self.assertIn("\\begin{tabular}", tabela, nome)
            self.assertIsNone(re.search(r"\bnan\b", tabela.lower()), nome)
        # Encerramento do simultâneo especializado: 6 de 12 unânimes na R1, com vírgula decimal
        self.assertIn("Unanimidade na R1 & 6 (50{,}0\\%)", tabelas["rq4_h5_encerramento"])
        # Sem baseline por issue, H3 é descritiva: sem p-valor na linha do MASPE
        self.assertIn("não pode ser testada", tabelas["rq2_2_h3"])
        self.assertIn("\\multicolumn{8}{l}{\\textit{Análise secundária", tabelas["rq2_1_h2"])
        self.assertIn("Revisões do agente-âncora", tabelas["rq3_h4"])

    def test_tabelas_latex_com_condicao_ausente(self):
        parcial = self.df[self.df.condicao == an.COND_SIM_ESP]
        tabelas = an.gerar_tabelas_latex(self._relatorio_de(parcial))
        self.assertTrue(tabelas["rq2_1_h2"].startswith("% Sem dados"))
        self.assertTrue(tabelas["rq3_h4"].startswith("% Sem dados"))
        self.assertIn("Unanimidade na R1 & 6 (50{,}0\\%) & --- & --- & ---", tabelas["rq4_h5_encerramento"])


if __name__ == "__main__":
    unittest.main(verbosity=2)