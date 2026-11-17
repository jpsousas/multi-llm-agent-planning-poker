"""
tests/test_split_parser.py
==========================
Testes rápidos (sem LLM e sem embeddings) para:
  - leitura do CSV do SIS-ASTROS (cabeçalhos, filtros, Time Spent);
  - split temporal 70/30 e manifesto;
  - parser das respostas dos agentes (extrair_json);
  - fase de DESENVOLVIMENTO (montar_dataset_sis_astros(fase="dev")): o bloco de teste nunca é alvo nem exemplo.

Execução (a partir da pasta multi-llm-agent-planning-poker):
    python -m unittest tests.test_split_parser -v
"""

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sis_astros import (
    carregar_sis_astros,
    dividir_temporal,
    parse_horas_gitlab,
    _amostrar_few_shot_estratificado,
)
from src.agents import extrair_json
from unittest.mock import patch
import src.sis_astros as sa
from tests.fakes import csv_sis_astros, linha, patch_retrieval

CABECALHO = ["IID", "Title", "State", "Author", "Assignee", "Time Estimate", "Time Spent",
             "Created At", "Closed At", "Milestone", "Labels", "Description", "URL"]


def _csv_temporario(linhas, cabecalho=CABECALHO) -> str:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8-sig", newline="")
    w = csv.DictWriter(tmp, fieldnames=cabecalho)
    w.writeheader()
    for l in linhas:
        w.writerow({c: l.get(c, "") for c in cabecalho})
    tmp.close()
    return tmp.name


def _linha(iid, est, data, spent="", titulo=None):
    return {"IID": iid, "Title": titulo or f"Issue {iid}", "Time Estimate": est,
            "Time Spent": spent, "Created At": data, "Description": f"Descrição {iid}",
            "State": "closed", "Labels": "backend"}


class TestCarregarSisAstros(unittest.TestCase):

    def setUp(self):
        self.arquivos = []

    def tearDown(self):
        for a in self.arquivos:
            os.unlink(a)

    def _csv(self, linhas, **kw):
        caminho = _csv_temporario(linhas, **kw)
        self.arquivos.append(caminho)
        return caminho

    def test_filtros_e_ordem_temporal(self):
        caminho = self._csv([
            _linha("3", "8", "2024-03-01 10:00:00 UTC"),
            _linha("1", "5h", "2024-01-01 10:00:00 UTC"),
            _linha("2", "", "2024-02-01 10:00:00 UTC"),       # vazio
            _linha("4", "30m", "2024-04-01 10:00:00 UTC"),    # minutos
            _linha("5", "4", "2024-05-01 10:00:00 UTC"),      # fora da escala
            _linha("6", "13", ""),                             # sem data
            _linha("7", "2", "2024-02-15T08:00:00.000Z"),
        ])
        itens = carregar_sis_astros(caminho)
        self.assertEqual([it["ID"] for it in itens], ["1", "7", "3"])
        self.assertEqual([it["Story_Point"] for it in itens], [5.0, 2.0, 8.0])

    def test_time_spent_ausente_fica_none(self):
        caminho = self._csv([
            _linha("1", "8", "2024-01-01", spent=""),
            _linha("2", "8", "2024-01-02", spent="1h 30m"),
        ])
        itens = carregar_sis_astros(caminho)
        self.assertIsNone(itens[0]["Tempo_Gasto_Horas"])
        self.assertAlmostEqual(itens[1]["Tempo_Gasto_Horas"], 1.5)

    def test_coluna_obrigatoria_ausente(self):
        cab = [c for c in CABECALHO if c != "Created At"]
        caminho = self._csv([{"IID": "1", "Title": "x", "Time Estimate": "8"}], cabecalho=cab)
        with self.assertRaises(KeyError):
            carregar_sis_astros(caminho)

    def test_arquivo_inexistente_aponta_leia_me(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            carregar_sis_astros("nao_existe/sis_astros.csv")
        self.assertIn("LEIA-ME", str(ctx.exception))

    def test_parse_horas_gitlab(self):
        self.assertEqual(parse_horas_gitlab("8"), 8.0)
        self.assertEqual(parse_horas_gitlab("8h"), 8.0)
        self.assertEqual(parse_horas_gitlab("45m"), 0.75)
        self.assertEqual(parse_horas_gitlab("1d 2h"), 10.0)
        self.assertIsNone(parse_horas_gitlab(""))
        self.assertIsNone(parse_horas_gitlab("abc"))

    def test_parse_horas_gitlab_com_segundos(self):
        """Formato com segundos: '101h 55m 20s' — deve ser parseado corretamente."""
        resultado = parse_horas_gitlab("101h 55m 20s")
        esperado = 101.0 + 55.0 / 60.0 + 20.0 / 3600.0  # ~101.9222 h
        self.assertIsNotNone(resultado, "101h 55m 20s deve retornar um valor, não None")
        self.assertAlmostEqual(resultado, esperado, places=4)

    def test_parse_horas_gitlab_apenas_segundos(self):
        """Apenas segundos: '90s' = 0.025 h."""
        self.assertAlmostEqual(parse_horas_gitlab("90s"), 90.0 / 3600.0, places=6)

    def test_parse_horas_gitlab_semanas_dias_horas_minutos_segundos(self):
        """Formato completo: '1w 2d 3h 15m 30s'."""
        esperado = 40.0 + 2 * 8.0 + 3.0 + 15.0 / 60.0 + 30.0 / 3600.0
        self.assertAlmostEqual(parse_horas_gitlab("1w 2d 3h 15m 30s"), esperado, places=4)


class TestSplitTemporal(unittest.TestCase):

    def _itens(self, sps):
        return [{"ID": str(i), "Title": f"t{i}", "Description": "", "Story_Point": float(sp),
                 "Tempo_Gasto_Horas": None, "Creation_Date": f"2024-01-{i + 1:02d}"}
                for i, sp in enumerate(sps)]

    def test_proporcao_e_ordem(self):
        itens = self._itens([1, 2, 3, 5, 8, 13, 21, 34, 55, 8])
        treino, teste = dividir_temporal(itens, 0.7, caminho_manifesto=None)
        self.assertEqual(len(treino), 7)
        self.assertEqual(len(teste), 3)
        self.assertEqual(treino + teste, itens)  # não reordena nem mistura

    def test_manifesto(self):
        itens = self._itens([1, 2, 3, 5, 8, 13, 21, 34, 55, 8])
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "split.json"
            dividir_temporal(itens, 0.7, caminho_manifesto=caminho)
            m = json.loads(caminho.read_text(encoding="utf-8"))
        self.assertEqual(m["n_historico"], 7)
        self.assertEqual(m["n_teste"], 3)
        self.assertEqual([i["ID"] for i in m["itens"] if i["lado"] == "teste"], ["7", "8", "9"])
        # 55 SP só aparece no teste -> deve ser apontado como ausente no histórico
        self.assertIn(55, m["degraus_ausentes_historico"])

    def test_few_shot_exclui_alvo(self):
        itens = self._itens([1, 2, 3, 5, 8])
        ex = _amostrar_few_shot_estratificado(itens, n=5, item_alvo=itens[2])
        self.assertNotIn("2", [e["ID"] for e in ex])


class TestFaseDev(unittest.TestCase):
    """20 issues -> histórico 14 (70%) / teste 6. Dev: base = 9 (70% de 14), alvos = 5 (os últimos do histórico)."""

    SPS = [1, 2, 3, 5, 8, 13, 21, 3, 5, 8, 13, 2, 3, 5, 8, 1, 2, 3, 5, 8]

    def setUp(self):
        self.csv = csv_sis_astros([linha(i, sp, i, titulo=f"T{i}", desc=f"descricao {i}") for i, sp in enumerate(self.SPS, start=1)])
        orig = sa.dividir_temporal
        self.p = patch.object(sa, "dividir_temporal",
                              lambda itens, prop, caminho_csv=None, caminho_manifesto=None: orig(itens, prop, caminho_csv=caminho_csv, caminho_manifesto=None))
        self.p.start()
        self.ids_teste = {str(i) for i in range(15, 21)}

    def tearDown(self):
        self.p.stop()
        os.unlink(self.csv)

    def test_dev_alvos_sao_o_fim_do_historico(self):
        ds = sa.montar_dataset_sis_astros(self.csv, fase="dev")
        self.assertEqual([d["issue_id"] for d in ds], ["10", "11", "12", "13", "14"])
        self.assertTrue(all(d["fase"] == "dev" for d in ds))

    def test_dev_nunca_toca_o_bloco_de_teste_few_shot(self):
        ds = sa.montar_dataset_sis_astros(self.csv, fase="dev", n_historico=3)
        for d in ds:
            self.assertNotIn(d["issue_id"], self.ids_teste)
            for h in d["historicos"]:
                self.assertNotIn(h["id"], self.ids_teste)
                self.assertLessEqual(int(h["id"]), 9)                       # exemplos só da base (70% mais antigos do histórico)

    def test_teste_alvos_sao_os_30_por_cento_finais(self):
        ds = sa.montar_dataset_sis_astros(self.csv, fase="teste")
        self.assertEqual({d["issue_id"] for d in ds}, self.ids_teste)
        for d in ds:
            for h in d["historicos"]:
                self.assertLessEqual(int(h["id"]), 14)                      # nunca um item do teste como exemplo

    def test_dev_rag_so_recupera_da_base_e_registra_diagnostico(self):
        with patch_retrieval():
            ds = sa.montar_dataset_sis_astros(self.csv, fase="dev", usar_rag=True, k_rag=3)
        for d in ds:
            self.assertEqual(d["modo_contexto"], "rag")
            self.assertEqual(d["k_contexto"], 3)
            self.assertLessEqual(len(d["historicos"]), 3)
            for h in d["historicos"]:
                self.assertLessEqual(int(h["id"]), 9)
                self.assertNotIn(h["id"], self.ids_teste)
                self.assertIn("similaridade", h)
            self.assertEqual(d["rag_meta"]["limiar"], 0.95)                  # limiar congelado é o padrão
            self.assertIn("sim_max", d["rag_meta"])

    def test_dev_rag_indice_so_com_a_base(self):
        import src.retrieval as r
        with patch_retrieval():
            sa.montar_dataset_sis_astros(self.csv, fase="dev", usar_rag=True)
            indices = list(r._indices_memoria.values())
        self.assertEqual(len(indices), 1)
        self.assertEqual([it["ID"] for it in indices[0]["itens"]], [str(i) for i in range(1, 10)])

    def test_k_padrao_vem_do_config(self):
        from src.config import K_RAG
        self.assertEqual(K_RAG, 3)
        with patch_retrieval():
            ds = sa.montar_dataset_sis_astros(self.csv, fase="dev", usar_rag=True)
        self.assertEqual(ds[0]["k_contexto"], K_RAG)

    def test_fase_invalida(self):
        with self.assertRaises(ValueError):
            sa.montar_dataset_sis_astros(self.csv, fase="producao")

    def test_ground_truth_e_esforco_real_chegam_ao_item(self):
        ds = sa.montar_dataset_sis_astros(self.csv, fase="dev")
        self.assertEqual(ds[0]["ground_truth"], float(self.SPS[9]))
        self.assertIsNone(ds[0]["esforco_real_horas"])


class TestExtrairJson(unittest.TestCase):

    def test_json_valido(self):
        d = extrair_json('{"justificativa": "CRUD simples", "story_points": 3}')
        self.assertEqual(d["story_points"], 3)

    def test_virgula_sobrando(self):
        d = extrair_json('{\n  "justificativa": "precisa de 2 endpoints",\n  "story_points": 8,\n}')
        self.assertEqual(d["story_points"], 8)
        self.assertIn("endpoints", d["justificativa"])

    def test_think_e_bloco_markdown(self):
        d = extrair_json('<think>analisando 5 telas</think>\n```json\n{"justificativa": "ok", "story_points": 13}\n```')
        self.assertEqual(d["story_points"], 13)
        self.assertIn("analisando", d["justificativa"])

    def test_numero_so_na_justificativa_falha(self):
        """Sem a chave story_points, não pode chutar um número do texto."""
        with self.assertRaises(ValueError):
            extrair_json("Acho que são 2 endpoints e umas 3 telas, algo médio.")

    def test_json_quebrado_com_chave(self):
        d = extrair_json('{"justificativa": "texto com "aspas" quebradas", "story_points": 21}')
        self.assertEqual(d["story_points"], 21)


if __name__ == "__main__":
    unittest.main(verbosity=2)
