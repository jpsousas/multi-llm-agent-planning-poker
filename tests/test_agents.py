"""
tests/test_agents.py
====================
src/agents.estimar com um modelo falso (sem rede): arredondamento à escala, retry, e que o prompt
(confidencial) só é impresso com DEBUG_PROMPT=1.

    python -m unittest tests.test_agents -v
"""
import contextlib
import io
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("TOKENIZADOR", "none")
os.environ.setdefault("LOCAL_LLM_MODEL", "qwen3-14b")
sys.path.insert(0, str(Path(__file__).parent.parent))

import src.agents as ag

ESCALA = "1, 2, 3, 5, 8, 13, 21, 34, 55"


class _Erro(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


class _ModeloFalso:
    def __init__(self, respostas):
        self.respostas = list(respostas)
        self.chamadas = 0
        self.ultimas_mensagens = None

    def invoke(self, mensagens):
        self.chamadas += 1
        self.ultimas_mensagens = mensagens
        r = self.respostas.pop(0)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(content=r, usage_metadata={"input_tokens": 1200, "output_tokens": 80})


def _estimar(modelo, **kw):
    args = dict(agente_id="a", papel_id="backend", modelo_id="local:m", temperatura=0.0,
                titulo="Título SEGREDO", descricao="Descrição SEGREDO", escala_fibonacci=ESCALA)
    args.update(kw)
    with patch.object(ag, "obter_modelo", lambda *a, **k: modelo):
        return ag.estimar(**args)


class TestEstimar(unittest.TestCase):

    def test_lance_na_escala(self):
        m = _ModeloFalso(['{"justificativa": "ok", "story_points": 8}'])
        r = _estimar(m)
        self.assertEqual(r["story_points"], 8.0)
        self.assertFalse(r["fora_da_escala"])
        self.assertEqual((r["tokens_entrada"], r["tokens_saida"]), (1200, 80))
        self.assertEqual(r["justificativa"], "ok")

    def test_lance_fora_da_escala_e_arredondado_e_sinalizado(self):
        r = _estimar(_ModeloFalso(['{"justificativa": "j", "story_points": 6}']))
        self.assertEqual(r["story_points"], 5.0)           # |6-5| < |6-8|
        self.assertEqual(r["story_points_bruto"], 6.0)
        self.assertTrue(r["fora_da_escala"])

    def test_prompt_enviado_tem_sistema_e_item(self):
        m = _ModeloFalso(['{"justificativa": "j", "story_points": 3}'])
        _estimar(m, historicos=[{"titulo": "H", "story_points": 5.0, "similaridade": 0.7}])
        sistema, humano = m.ultimas_mensagens
        self.assertIn(ESCALA, sistema.content)
        self.assertIn("Backend", sistema.content)
        self.assertIn("Descrição SEGREDO", humano.content)
        self.assertIn("Issues Históricas", humano.content)

    def test_retry_em_erro_transitorio(self):
        m = _ModeloFalso([_Erro(503), '{"justificativa": "j", "story_points": 2}'])
        with patch.object(ag.time, "sleep") as dorme:
            r = _estimar(m)
        self.assertEqual(r["story_points"], 2.0)
        self.assertEqual(m.chamadas, 2)
        dorme.assert_called_once()

    def test_sem_retry_em_erro_permanente(self):
        m = _ModeloFalso([_Erro(400), '{"justificativa": "j", "story_points": 2}'])
        with self.assertRaises(_Erro):
            _estimar(m)
        self.assertEqual(m.chamadas, 1)

    def test_esgota_tentativas(self):
        m = _ModeloFalso([_Erro(500), _Erro(500), _Erro(500)])
        with patch.object(ag.time, "sleep"), self.assertRaises(_Erro):
            _estimar(m)
        self.assertEqual(m.chamadas, 3)

    def test_resposta_sem_story_points_falha(self):
        with self.assertRaises(ValueError):
            _estimar(_ModeloFalso(["não sei estimar isso"]))

    def test_vale_repetir(self):
        for s in (408, 409, 429, 500, 503):
            self.assertTrue(ag.vale_repetir(_Erro(s)), s)
        for s in (400, 401, 404):
            self.assertFalse(ag.vale_repetir(_Erro(s)), s)
        self.assertTrue(ag.vale_repetir(TimeoutError()))
        self.assertFalse(ag.vale_repetir(ValueError()))


class TestPromptConfidencial(unittest.TestCase):

    def _saida(self, debug):
        env = {"DEBUG_PROMPT": "1"} if debug else {}
        buf = io.StringIO()
        with patch.dict(os.environ, env, clear=False):
            if not debug:
                os.environ.pop("DEBUG_PROMPT", None)
            with contextlib.redirect_stdout(buf):
                _estimar(_ModeloFalso(['{"justificativa": "j", "story_points": 3}']))
        return buf.getvalue()

    def test_nao_imprime_por_padrao(self):
        self.assertNotIn("SEGREDO", self._saida(debug=False))

    def test_imprime_com_debug(self):
        self.assertIn("Descrição SEGREDO", self._saida(debug=True))


class TestModeloLocal(unittest.TestCase):

    def test_modo_json_opcional(self):
        ag._cache.clear()
        with patch.dict(os.environ, {"LLM_JSON_MODE": "1", "LOCAL_LLM_URL": "http://localhost:11434/v1"}):
            c = ag.obter_modelo("local:m", 0.0)
        self.assertEqual(c.model_kwargs.get("response_format"), {"type": "json_object"})
        ag._cache.clear()
        with patch.dict(os.environ, {"LLM_JSON_MODE": "0", "LOCAL_LLM_URL": "http://localhost:11434/v1"}):
            c2 = ag.obter_modelo("local:m", 0.0)
        self.assertNotIn("response_format", c2.model_kwargs)
        ag._cache.clear()

    def test_cliente_local_usa_url_do_ambiente(self):
        ag._cache.clear()
        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://localhost:8080/v1"}):
            c = ag.obter_modelo("local:qwen3-14b", 0.0, seed=7)
        self.assertEqual(str(c.openai_api_base), "http://localhost:8080/v1")
        self.assertEqual(c.model_name, "qwen3-14b")
        ag._cache.clear()


if __name__ == "__main__":
    unittest.main(verbosity=2)