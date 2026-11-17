"""
tests/test_orcamento_contexto.py
================================
Orçamento de tokens do prompt (src/sis_astros.py): alvo completo, exemplos truncados com orçamento dinâmico,
limpeza opcional (LIMPAR_TEXTO) e o `prep_meta` gravado por item. Sem LLM e sem embeddings reais.

    python -m unittest tests.test_orcamento_contexto -v
"""
import math
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["TOKENIZADOR"] = "none"
sys.path.insert(0, str(Path(__file__).parent.parent))

import src.sis_astros as sa
from tests.fakes import csv_sis_astros, linha

MARCADOR = "[... texto truncado ...]"


def _tokens(txt):  # contar_tokens sem tokenizador: ceil(chars / 3)
    return math.ceil(len(txt) / 3.0)


class TestOrcamentoDinamico(unittest.TestCase):

    def setUp(self):
        self.p = [patch.object(sa, "CONTEXTO_SERVIDOR_POR_SLOT", 16384), patch.object(sa, "RESERVA_PROMPT_TOKENS", 5000),
                  patch.object(sa, "ORCAMENTO_TOKENS_EXEMPLO", 2500), patch.object(sa, "PISO_ORCAMENTO_EXEMPLO", 200)]
        for p in self.p:
            p.start()

    def tearDown(self):
        for p in self.p:
            p.stop()

    def test_alvo_pequeno_usa_o_teto(self):
        self.assertEqual(sa.orcamento_exemplo_dinamico(300, 3), 2500)

    def test_alvo_grande_encolhe_o_exemplo(self):
        self.assertEqual(sa.orcamento_exemplo_dinamico(4704, 3), (16384 - 5000 - 4704) // 3)
        self.assertEqual(sa.orcamento_exemplo_dinamico(8962, 3), (16384 - 5000 - 8962) // 3)

    def test_piso(self):
        self.assertEqual(sa.orcamento_exemplo_dinamico(20000, 3), 200)

    def test_prompt_cabe_no_contexto(self):
        """alvo + k*exemplo + reserva <= contexto para qualquer alvo até a maior issue medida (8.962 tokens)."""
        for alvo in (0, 662, 2267, 4704, 8962):
            for k in (1, 3, 5):
                orc = sa.orcamento_exemplo_dinamico(alvo, k)
                self.assertLessEqual(alvo + k * orc + 5000, 16384, (alvo, k))

    def test_k_zero_nao_divide_por_zero(self):
        self.assertGreaterEqual(sa.orcamento_exemplo_dinamico(100, 0), 200)


class TestPrepararTexto(unittest.TestCase):

    def test_sem_teto_texto_completo(self):
        longo = "palavra " * 20000
        with patch.object(sa, "LIMPAR_TEXTO", False):
            t, meta = sa.preparar_texto(longo, None)
        self.assertEqual(t, longo.strip())
        self.assertEqual(meta["truncado"], 0)

    def test_truncamento_respeita_orcamento_e_marca(self):
        longo = "palavra " * 5000
        with patch.object(sa, "LIMPAR_TEXTO", False):
            t, meta = sa.preparar_texto(longo, 500)
        self.assertTrue(t.endswith(MARCADOR))
        self.assertLessEqual(_tokens(t), 500)
        self.assertEqual(meta["truncado"], 1)

    def test_texto_curto_nao_muda(self):
        with patch.object(sa, "LIMPAR_TEXTO", False):
            t, meta = sa.preparar_texto("Ajustar tela.\r\nDois.", 500)
        self.assertEqual(t, "Ajustar tela.\nDois.")
        self.assertEqual(meta["truncado"], 0)

    def test_none_vira_vazio(self):
        with patch.object(sa, "LIMPAR_TEXTO", False):
            self.assertEqual(sa.preparar_texto(None, 100)[0], "")

    def test_limpeza_so_com_flag(self):
        txt = "Ver http://intranet/x/y para detalhes."
        with patch.object(sa, "LIMPAR_TEXTO", False):
            self.assertIn("http://intranet", sa.preparar_texto(txt, 500)[0])
        with patch.object(sa, "LIMPAR_TEXTO", True):
            t, meta = sa.preparar_texto(txt, 500)
        self.assertNotIn("http://", t)
        self.assertIn("[link]", t)
        self.assertEqual(meta["urls"], 1)


class TestFormatarExemplo(unittest.TestCase):

    def test_uma_linha_e_esforco_real(self):
        item = {"ID": 7, "Title": "T", "Description": "linha1\nlinha2\n\nlinha3", "Story_Point": 5.0, "Tempo_Gasto_Horas": 6.04}
        with patch.object(sa, "LIMPAR_TEXTO", False):
            ex = sa.formatar_exemplo(item, 500)
        self.assertEqual(ex["descricao"], "linha1 linha2 linha3")
        self.assertEqual(ex["esforco_real_horas"], 6.0)
        self.assertEqual(ex["id"], "7")

    def test_sem_esforco_real_ou_zero(self):
        for tempo in (None, 0, 0.0):
            ex = sa.formatar_exemplo({"ID": 1, "Title": "T", "Description": "d", "Story_Point": 3.0, "Tempo_Gasto_Horas": tempo}, 500)
            self.assertNotIn("esforco_real_horas", ex)


class TestDatasetComOrcamento(unittest.TestCase):
    """Dataset montado de ponta a ponta (few-shot, sem RAG) com um alvo gigante."""

    def setUp(self):
        sps = [1, 2, 3, 5, 8, 13, 21, 3, 5, 8, 13, 2, 3, 5, 8, 1, 2, 3, 5, 8]
        linhas = []
        for i, sp in enumerate(sps, start=1):
            desc = "exemplo " * 4000 if i <= 3 else f"descrição {i}"        # exemplos antigos gigantes
            if i == len(sps):
                desc = "alvo " * 12000                                        # último = alvo gigante (≈ 20k tokens)
            linhas.append(linha(i, sp, i, desc=desc))
        self.csv = csv_sis_astros(linhas)
        orig = sa.dividir_temporal
        self.patch = patch.object(sa, "dividir_temporal",
                                  lambda itens, p, caminho_csv=None, caminho_manifesto=None: orig(itens, p, caminho_csv=caminho_csv, caminho_manifesto=None))
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        os.unlink(self.csv)

    def test_alvo_completo_e_exemplos_truncados(self):
        with patch.object(sa, "LIMPAR_TEXTO", False):
            ds = sa.montar_dataset_sis_astros(self.csv, n_historico=3)
        alvo = ds[-1]
        self.assertNotIn(MARCADOR, alvo["descricao"])                         # alvo nunca é truncado
        self.assertEqual(len(alvo["descricao"].split()), 12000)
        pm = alvo["prep_meta"]
        self.assertEqual(pm["orcamento_exemplo"], 200)                        # alvo enorme -> orçamento no piso
        self.assertEqual(pm["tokens_alvo"], _tokens(alvo["descricao"]))
        for h in alvo["historicos"]:
            self.assertLessEqual(_tokens(h["descricao"]), pm["orcamento_exemplo"] + 10)
        self.assertEqual(alvo["fase"], "teste")
        self.assertEqual(alvo["modo_contexto"], "few_shot")
        self.assertIsNone(alvo["rag_meta"])

    def test_orcamento_cheio_quando_alvo_pequeno(self):
        with patch.object(sa, "LIMPAR_TEXTO", False):
            ds = sa.montar_dataset_sis_astros(self.csv, n_historico=3)
        primeiro = ds[0]
        self.assertEqual(primeiro["prep_meta"]["orcamento_exemplo"], sa.ORCAMENTO_TOKENS_EXEMPLO)
        for k in ("tokens_alvo", "tokens_exemplos", "exemplos_truncados", "orcamento_exemplo"):
            self.assertIn(k, primeiro["prep_meta"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
