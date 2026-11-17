"""
tests/test_runner.py
====================
src/runner.rodar_condicao com um grafo falso (sem LLM): separação dev/teste, registro de configuração e
hash do prompt, retomada e concorrência. Roda numa pasta temporária (não suja results/).

    python -m unittest tests.test_runner -v
"""
import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("TOKENIZADOR", "none")
os.environ.setdefault("LOCAL_LLM_MODEL", "qwen3-14b")
sys.path.insert(0, str(Path(__file__).parent.parent))

import src.runner as rn
from src.prompts import PROMPT_BASE


class _GrafoFalso:
    def __init__(self, falhar_em=None):
        self.configs, self.entradas, self.falhar_em = [], [], falhar_em

    async def ainvoke(self, entrada, config):
        self.configs.append(config)
        self.entradas.append(entrada)
        if entrada["issue_id"] == self.falhar_em:
            raise RuntimeError("falha simulada")
        est = [{"agente": a, "papel": p, "modelo": "m", "story_points": 5.0, "justificativa": "j", "rodada": 1,
                "tokens_entrada": 100, "tokens_saida": 10, "latencia_ms": 5.0, "data_hora": "", "ordem_revelacao": None,
                "contexto_recebido": None} for a, p in entrada["papeis_agentes"].items()]
        return {"rodada": 2, "consenso": 5.0, "convergiu": True, "motivo_convergencia": "consenso_unanime_r1",
                "dispersao_degraus": 0, "estimativas": est,
                "historico_rodadas": [{"rodada": 1, "valores": [5, 5, 5], "sigma_sp": 0.0, "dispersao_degraus": 0,
                                       "motivo_convergencia": "consenso_unanime_r1", "consenso": 5.0, "extremos": {"diferenca": 0}}]}


def _dataset(n=3, modo="rag"):
    return [{"issue_id": str(i), "titulo": f"T{i}", "descricao": f"D{i}", "ground_truth": 5.0, "esforco_real_horas": 6.0,
             "modo_contexto": modo, "k_contexto": 3, "fase": "dev",
             "historicos": [{"id": "9", "titulo": "H", "descricao": "d", "story_points": 3.0, "esforco_real_horas": 4.0, "similaridade": 0.8}],
             "prep_meta": {"tokens_alvo": 10, "orcamento_exemplo": 2500, "exemplos_truncados": 0},
             "rag_meta": {"sim_max": 0.8, "limiar": 0.95, "n_removidas_limiar": 0}} for i in range(n)]


class TestRunner(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def _rodar(self, grafo, dataset=None, **kw):
        args = dict(nome_cond="simultaneo_especializado", dataset=dataset or _dataset(), projeto="SIS-ASTROS", modelo_override="qwen3-14b")
        args.update(kw)
        with patch.object(rn, "construir_grafo", lambda cond: grafo), patch.object(rn.time, "sleep"):
            asyncio.run(rn.rodar_condicao(**args))

    @staticmethod
    def _linhas(caminho):
        with open(caminho, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def test_dev_grava_em_results_dev_e_nao_em_raw(self):
        self._rodar(_GrafoFalso(), fase="dev")
        dev = list(Path("results/dev").glob("*.jsonl"))
        self.assertEqual(len(dev), 1)
        self.assertIn("__dev", dev[0].name)
        self.assertFalse(Path("results/raw").exists() and any(Path("results/raw").glob("*.jsonl")))
        self.assertTrue(any(Path("results/transcripts_dev").rglob("*.md")))
        self.assertFalse(Path("results/transcripts").exists())

    def test_teste_grava_em_raw(self):
        self._rodar(_GrafoFalso(), fase="teste")
        raw = list(Path("results/raw").glob("*.jsonl"))
        self.assertEqual(len(raw), 1)
        self.assertNotIn("__dev", raw[0].name)
        self.assertFalse(Path("results/dev").exists())

    def test_registro_tem_configuracao_hash_e_meta(self):
        self._rodar(_GrafoFalso(), fase="dev")
        r = self._linhas(next(Path("results/dev").glob("*.jsonl")))[0]
        self.assertEqual(r["fase"], "dev")
        self.assertEqual(len(r["hash_prompt"]), 10)
        self.assertEqual(r["hash_prompt"], rn.HASH_PROMPT)
        cfg = r["configuracao"]
        for k in ("limpar_texto", "orcamento_tokens_exemplo", "contexto_por_slot", "reserva_prompt", "modelo_embeddings", "limiar_duplicata"):
            self.assertIn(k, cfg)
        self.assertEqual(cfg["limiar_duplicata"], 0.95)
        self.assertIn("mpnet", cfg["modelo_embeddings"])
        self.assertEqual(r["prep_meta"]["orcamento_exemplo"], 2500)
        self.assertEqual(r["rag_meta"]["sim_max"], 0.8)
        self.assertEqual(r["ids_contexto"], ["9"])
        self.assertEqual(r["rodadas_usadas"], 1)

    def test_few_shot_nao_registra_modelo_de_embeddings(self):
        self._rodar(_GrafoFalso(), dataset=_dataset(modo="few_shot"), fase="dev")
        r = self._linhas(next(Path("results/dev").glob("*.jsonl")))[0]
        self.assertIsNone(r["configuracao"]["modelo_embeddings"])

    def test_hash_do_prompt_muda_com_o_prompt(self):
        from hashlib import md5
        from src.prompts import PAPEIS
        outro = md5(((PROMPT_BASE + " x") + "".join(f"{k}={v}" for k, v in sorted(PAPEIS.items()))).encode()).hexdigest()[:10]
        self.assertNotEqual(outro, rn.HASH_PROMPT)

    def test_concorrencia_um_por_padrao(self):
        g = _GrafoFalso()
        self._rodar(g, fase="dev")
        self.assertTrue(all(c.get("max_concurrency") == rn.MAX_CONCORRENCIA for c in g.configs))
        self.assertTrue(all("max_concurrent" not in c for c in g.configs))
        self.assertEqual(rn.MAX_CONCORRENCIA, 1)

    def test_retomada_pula_issues_feitas(self):
        self._rodar(_GrafoFalso(), fase="dev")
        g2 = _GrafoFalso()
        self._rodar(g2, dataset=_dataset(5), fase="dev")
        self.assertEqual([e["issue_id"] for e in g2.entradas], ["3", "4"])
        self.assertEqual(len(self._linhas(next(Path("results/dev").glob("*.jsonl")))), 5)

    def test_limite_usa_as_primeiras_n(self):
        g = _GrafoFalso()
        self._rodar(g, dataset=_dataset(5), limite=2, fase="dev")
        self.assertEqual([e["issue_id"] for e in g.entradas], ["0", "1"])

    def test_falha_vai_para_arquivo_de_falhas_e_segue(self):
        self._rodar(_GrafoFalso(falhar_em="1"), fase="dev")
        pasta = Path("results/dev")
        ok = [p for p in pasta.glob("*.jsonl") if "__falhas" not in p.name][0]
        falhas = [p for p in pasta.glob("*__falhas.jsonl")][0]
        self.assertEqual([r["issue_id"] for r in self._linhas(ok)], ["0", "2"])
        self.assertEqual(self._linhas(falhas)[0]["tipo_erro"], "RuntimeError")

    def test_condicao_desconhecida(self):
        with self.assertRaises(ValueError):
            self._rodar(_GrafoFalso(), nome_cond="inexistente")

    def test_papeis_fixos_e_ordem_sequencial(self):
        g = _GrafoFalso()
        self._rodar(g, nome_cond="sequencial_especializado", fase="dev")
        self.assertEqual(g.entradas[0]["papeis_agentes"], {"frontend": "frontend", "backend": "backend", "qa": "qa"})
        r = self._linhas(next(Path("results/dev").glob("*.jsonl")))[0]
        self.assertEqual(r["ordem_fala"], ["frontend", "backend", "qa"])


class TestCondicaoComModelo(unittest.TestCase):

    def test_prefixo_local_e_mesmo_modelo_nos_tres_agentes(self):
        c = rn._condicao_com_modelo("simultaneo_neutro", "qwen3-14b", None)
        self.assertEqual({a[2] for a in c.agentes}, {"local:qwen3-14b"})
        c2 = rn._condicao_com_modelo("simultaneo_neutro", "ollama:qwen3:14b", None)
        self.assertEqual({a[2] for a in c2.agentes}, {"ollama:qwen3:14b"})

    def test_temperatura_e_seed(self):
        c = rn._condicao_com_modelo("simultaneo_neutro", "m", None, temperatura=0.3, seed=1)
        self.assertEqual((c.temperatura, c.seed), (0.3, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
