"""
tests/test_split_parser.py
==========================
Testes rápidos (sem LLM e sem embeddings) para:
  - leitura do CSV do SIS-ASTROS (cabeçalhos, filtros, Time Spent);
  - split temporal 70/30 e manifesto;
  - parser das respostas dos agentes (extrair_json).

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
