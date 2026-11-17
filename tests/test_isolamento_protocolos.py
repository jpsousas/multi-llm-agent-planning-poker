"""
Verifica o que cada agente RECEBE em cada protocolo, usando um LLM falso que registra
o contexto entregue a cada chamada (sem rede, sem modelo real).

Simultâneo: na R1 ninguém vê lance de ninguém (revelação cega); a partir da R2 os três
recebem o MESMO resumo da rodada anterior.
Sequencial: Frontend (posição 1) fala sem ver ninguém; Backend vê o Frontend; QA vê os dois.
"""
import unittest
from unittest.mock import patch

import src.graph as graph
from src.config import CONDICOES

ESCALA = "1, 2, 3, 5, 8, 13, 21, 34, 55"


class _LLMFalso:
    def __init__(self, lances):
        self.lances = lances          # {papel: [lance_r1, lance_r2, lance_r3]}
        self.chamadas = []            # (rodada, papel, contexto_recebido)

    def __call__(self, **kw):
        self.chamadas.append((kw["rodada"], kw["papel_id"], kw["contexto_rodada"]))
        sp = self.lances[kw["papel_id"]][kw["rodada"] - 1]
        return {"agente": kw["agente_id"], "papel": kw["papel_id"], "modelo": kw["modelo_id"],
                "story_points": float(sp), "justificativa": f"JUSTIF-{kw['papel_id']}-R{kw['rodada']}",
                "rodada": kw["rodada"], "tokens_entrada": 0, "tokens_saida": 0,
                "data_hora": "", "latencia_ms": 0.0}


def _rodar(nome_cond, lances):
    llm = _LLMFalso(lances)
    cond = CONDICOES[nome_cond]
    with patch.object(graph, "estimar", llm), patch.object(graph, "obter_modelo", lambda *a, **k: None):
        grafo = graph.construir_grafo(cond)
        final = grafo.invoke({
            "issue_id": "X", "titulo": "t", "descricao": "d", "historicos": [],
            "escala_fibonacci": ESCALA, "protocolo": cond.protocolo, "especializado": cond.especializado,
            "papeis_agentes": {a[0]: a[1] for a in cond.agentes}, "estimativas": [], "rodada": 1,
            "convergiu": False, "motivo_convergencia": None, "dispersao_degraus": None,
            "consenso": 0.0, "contexto_rodada_atual": None, "historico_rodadas": [],
        })
    return llm.chamadas, final


class TestSimultaneo(unittest.TestCase):
    def setUp(self):
        # Divergem na R1, mudam na R2 (sem estagnação), unânimes na R3
        lances = {"frontend": [3, 5, 8], "backend": [13, 8, 8], "qa": [21, 13, 8]}
        self.chamadas, self.final = _rodar("simultaneo_especializado", lances)

    def test_r1_cega(self):
        r1 = [c for c in self.chamadas if c[0] == 1]
        self.assertEqual(len(r1), 3)
        for _, papel, ctx in r1:
            self.assertIsNone(ctx, f"{papel} recebeu contexto na R1 do simultâneo")

    def test_r2_todos_recebem_o_mesmo_resumo_da_r1(self):
        r2 = [c for c in self.chamadas if c[0] == 2]
        self.assertEqual(len(r2), 3)
        contextos = {ctx for _, _, ctx in r2}
        self.assertEqual(len(contextos), 1, "agentes receberam resumos diferentes na R2")
        ctx = contextos.pop()
        for papel, sp in (("Frontend", 3), ("Backend", 13), ("Qa", 21)):
            self.assertIn(f"**{papel}**", ctx)
            self.assertIn(f"{sp} SP", ctx)
        self.assertNotIn("-R2", ctx, "resumo da R2 vazou lance da própria R2")

    def test_lance_registrado_e_do_proprio_agente(self):
        for e in self.final["estimativas"]:
            self.assertEqual(e["justificativa"], f"JUSTIF-{e['papel']}-R{e['rodada']}")

    def test_tres_rodadas_e_consenso(self):
        self.assertEqual(self.final["consenso"], 8.0)
        self.assertTrue(self.final["motivo_convergencia"].startswith("consenso_unanime_r3"))


class TestSequencial(unittest.TestCase):
    def setUp(self):
        lances = {"frontend": [3, 5], "backend": [8, 5], "qa": [5, 5]}
        self.chamadas, self.final = _rodar("sequencial_especializado", lances)

    def test_ordem_e_visibilidade_r1(self):
        r1 = [c for c in self.chamadas if c[0] == 1]
        self.assertEqual([p for _, p, _ in r1], ["frontend", "backend", "qa"])
        self.assertIsNone(r1[0][2])                                   # Frontend não vê ninguém
        self.assertIn("JUSTIF-frontend-R1", r1[1][2])                 # Backend vê Frontend
        self.assertNotIn("JUSTIF-qa-R1", r1[1][2])
        self.assertIn("JUSTIF-frontend-R1", r1[2][2])                 # QA vê os dois
        self.assertIn("JUSTIF-backend-R1", r1[2][2])

    def test_ancora_posicao_1_e_frontend(self):
        r1 = [e for e in self.final["estimativas"] if e["rodada"] == 1]
        primeiro = min(r1, key=lambda e: e["ordem_revelacao"])
        self.assertEqual(primeiro["papel"], "frontend")
        self.assertEqual(primeiro["ordem_revelacao"], 1)


class TestParadaNoGrafo(unittest.TestCase):
    """A regra de parada (unanimidade / estagnação R2 / teto R3) acontecendo dentro do grafo real."""

    def test_unanimidade_na_r1_encerra_com_uma_chamada_por_agente(self):
        chamadas, final = _rodar("simultaneo_especializado", {"frontend": [5], "backend": [5], "qa": [5]})
        self.assertEqual(len(chamadas), 3)
        self.assertTrue(final["convergiu"])
        self.assertEqual(final["consenso"], 5.0)
        self.assertEqual(final["rodada"] - 1, 1)

    def test_estagnacao_na_r2_encerra_com_a_mediana(self):
        lances = {"frontend": [3, 3], "backend": [13, 13], "qa": [8, 8]}
        chamadas, final = _rodar("simultaneo_especializado", lances)
        self.assertEqual(len(chamadas), 6)                       # 2 rodadas x 3 agentes
        self.assertTrue(final["convergiu"])
        self.assertTrue(final["motivo_convergencia"].startswith("estagnacao_r2_mediana"))
        self.assertEqual(final["consenso"], 8.0)

    def test_teto_na_r3_sem_unanimidade_usa_mediana(self):
        lances = {"frontend": [3, 5, 5], "backend": [13, 8, 8], "qa": [21, 13, 13]}
        chamadas, final = _rodar("simultaneo_especializado", lances)
        self.assertEqual(len(chamadas), 9)
        self.assertFalse(final["convergiu"])
        self.assertTrue(final["motivo_convergencia"].startswith("desempate_mediana_r3"))
        self.assertEqual(final["consenso"], 8.0)
        self.assertEqual(final["rodada"] - 1, 3)

    def test_registro_de_auditoria_do_contexto_recebido(self):
        _, final = _rodar("simultaneo_especializado", {"frontend": [3, 8], "backend": [13, 8], "qa": [8, 8]})
        r1 = [e for e in final["estimativas"] if e["rodada"] == 1]
        r2 = [e for e in final["estimativas"] if e["rodada"] == 2]
        self.assertTrue(all(e["contexto_recebido"] is None for e in r1))
        self.assertTrue(all(e["contexto_recebido"] for e in r2))
        self.assertTrue(all(e["ordem_revelacao"] is None for e in r1 + r2))   # simultâneo: revelação cega

    def test_sequencial_ordem_revelacao_registrada(self):
        _, final = _rodar("sequencial_especializado", {"frontend": [5, 5], "backend": [5, 5], "qa": [5, 5]})
        r1 = sorted((e for e in final["estimativas"] if e["rodada"] == 1), key=lambda e: e["ordem_revelacao"])
        self.assertEqual([e["papel"] for e in r1], ["frontend", "backend", "qa"])
        self.assertEqual([e["ordem_revelacao"] for e in r1], [1, 2, 3])

    def test_sequencial_r2_tambem_encadeia_e_ve_o_debate_anterior(self):
        chamadas, _ = _rodar("sequencial_especializado", {"frontend": [3, 5, 5], "backend": [8, 5, 5], "qa": [13, 8, 8]})
        r2 = [c for c in chamadas if c[0] == 2]
        self.assertEqual([p for _, p, _ in r2], ["frontend", "backend", "qa"])
        self.assertIn("Lances da Rodada 1", r2[0][2])             # debate da R1 chega ao primeiro da R2
        self.assertNotIn("Lances já revelados nesta rodada", r2[0][2])
        self.assertIn("Lances já revelados nesta rodada", r2[2][2])


if __name__ == "__main__":
    unittest.main()
