"""
tests/test_dev_resumo.py
========================
scripts/dev_resumo.py: métricas por degrau, baselines e — principalmente — que NADA de texto de issue
é impresso (só agregados).

    python -m unittest tests.test_dev_resumo -v
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("TOKENIZADOR", "none")
os.environ.setdefault("LOCAL_LLM_MODEL", "qwen3-14b")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scripts import dev_resumo as dr


def _rec(i, gt, predito, lances_r1, vizinhos=(5, 5, 5), motivo="consenso_unanime_r1", rodadas=1):
    return {"issue_id": str(i), "ground_truth": gt, "predito": predito, "titulo": "TITULO-SECRETO", "descricao": "DESCRICAO-SECRETA",
            "motivo_convergencia": motivo, "rodadas_usadas": rodadas, "tokens_total": 3000, "tempo_total_s": 30.0,
            "estimativas": [{"rodada": 1, "story_points": v, "fora_da_escala": False, "tokens_entrada": 1000} for v in lances_r1],
            "contexto_historico": [{"story_points": v, "titulo": "TITULO-SECRETO"} for v in vizinhos],
            "prep_meta": {"exemplos_truncados": 1, "tokens_alvo": 300, "orcamento_exemplo": 2500}, "hash_prompt": "abc123"}


def _escrever(pasta, nome, recs):
    caminho = Path(pasta) / nome
    with open(caminho, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return str(caminho)


class TestMetricas(unittest.TestCase):

    def test_exato_e_um_degrau(self):
        m = dr.metricas([5, 8, 13, 21], [5, 5, 13, 55])
        self.assertAlmostEqual(m["MAE"], (0 + 3 + 0 + 34) / 4)
        self.assertEqual(m["exato"], 50.0)
        self.assertEqual(m["ate1"], 75.0)                 # 8->5 é 1 degrau; 21->55 são 2
        self.assertAlmostEqual(m["deg_medio"], (0 + 1 + 0 + 2) / 4)

    def test_knn_mediana_e_snap_para_a_escala(self):
        self.assertEqual(dr.knn_mediana({"contexto_historico": [{"story_points": 3.0}, {"story_points": 8.0}, {"story_points": 13.0}]}), 8.0)
        self.assertEqual(dr.knn_mediana({"contexto_historico": [{"story_points": 3.0}, {"story_points": 5.0}]}), 3.0)   # mediana 4 equidista de 3 e 5: empate vai para o degrau menor
        self.assertIsNone(dr.knn_mediana({"contexto_historico": []}))

    def test_wilcoxon_com_poucos_pares_ou_identicos_e_nan(self):
        self.assertTrue(np.isnan(dr.wil([1, 2, 3], [2, 3, 4])))
        self.assertTrue(np.isnan(dr.wil([1] * 10, [1] * 10)))
        self.assertLess(dr.wil(list(range(20)), [x + 1 for x in range(20)]), 0.05)


class TestSaida(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.recs_a = [_rec(i, gt, p, [p, p, p]) for i, (gt, p) in enumerate([(5, 5), (8, 5), (13, 13), (3, 5), (8, 8), (5, 8)])]
        self.recs_b = [_rec(i, gt, p, [p, p, 13]) for i, (gt, p) in enumerate([(5, 5), (8, 8), (13, 13), (3, 3), (8, 8), (5, 13)])]
        self.a = _escrever(self.tmp.name, "A.jsonl", self.recs_a)
        self.b = _escrever(self.tmp.name, "B.jsonl", self.recs_b)

    def tearDown(self):
        self.tmp.cleanup()

    def _capturar(self, fn, *a):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn(*a)
        return buf.getvalue()

    def test_resumo_so_imprime_agregados(self):
        out = self._capturar(dr.resumo, self.a, 5.0)
        self.assertNotIn("SECRET", out)
        for trecho in ("Consenso MASPE", "Mediana global (5)", "kNN", "encerramento", "rodadas", "hash do prompt: abc123", "exemplos truncados"):
            self.assertIn(trecho, out)

    def test_comparar_so_imprime_agregados(self):
        out = self._capturar(dr.comparar, self.a, self.b)
        self.assertNotIn("SECRET", out)
        self.assertIn("6 issues comuns", out)
        self.assertIn("B melhor em", out)

    def test_arquivo_vazio(self):
        vazio = _escrever(self.tmp.name, "vazio.jsonl", [])
        self.assertIn("vazio", self._capturar(dr.resumo, vazio, None))

    def test_ler_ignora_linhas_corrompidas_e_duplicadas_ficam_com_a_ultima(self):
        caminho = Path(self.tmp.name) / "c.jsonl"
        caminho.write_text(json.dumps(_rec(1, 5, 5, [5, 5, 5])) + "\n{quebrado\n" + json.dumps(_rec(1, 5, 8, [8, 8, 8])) + "\n", encoding="utf-8")
        d = dr.ler(str(caminho))
        self.assertEqual(len(d), 1)
        self.assertEqual(d["1"]["predito"], 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
