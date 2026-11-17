"""
Fluxo do grafo de consenso (nós de avaliação e roteamento), escala do SIS-ASTROS.
    python -m unittest tests.test_fluxo_consenso -v
"""
import unittest
from src.state import ConsensoState
from src.graph import _no_avaliar_consenso, _rotear_rodada

class TestFluxoGrafoConsenso(unittest.TestCase):
    def test_fluxo_unanimidade_r1(self):
        state: ConsensoState = {
            "issue_id": "TEST-1",
            "titulo": "Issue Teste",
            "descricao": "Descricao Teste",
            "historicos": [],
            "escala_fibonacci": "1, 2, 3, 5, 8, 13, 21, 34, 55",
            "protocolo": "simultaneo",
            "especializado": True,
            "papeis_agentes": None,
            "estimativas": [
                {"agente": "backend", "papel": "backend", "modelo": "m", "story_points": 5.0, "justificativa": "j", "rodada": 1, "tokens_entrada": 10, "tokens_saida": 10, "latencia_ms": 100, "data_hora": None},
                {"agente": "frontend", "papel": "frontend", "modelo": "m", "story_points": 5.0, "justificativa": "j", "rodada": 1, "tokens_entrada": 10, "tokens_saida": 10, "latencia_ms": 100, "data_hora": None},
                {"agente": "qa", "papel": "qa", "modelo": "m", "story_points": 5.0, "justificativa": "j", "rodada": 1, "tokens_entrada": 10, "tokens_saida": 10, "latencia_ms": 100, "data_hora": None},
            ],
            "rodada": 1,
            "convergiu": False,
            "motivo_convergencia": None,
            "dispersao_degraus": None,
            "rodadas_consecutivas_convergidas": None,
            "consenso": 0.0,
            "contexto_rodada_atual": None,
            "historico_rodadas": [],
        }

        updates = _no_avaliar_consenso(state, metodo="mediana", max_rodadas=3)
        self.assertTrue(updates["convergiu"])
        self.assertEqual(updates["consenso"], 5.0)
        self.assertEqual(updates["motivo_convergencia"], "consenso_unanime_r1")

        state.update(updates)
        decisao = _rotear_rodada(state, max_rodadas=3, deliberacao=True)
        self.assertEqual(decisao, "__end__")

    def test_fluxo_estagnacao_r2(self):
        # Simula R1 divergente + R2 com os mesmos lances
        state: ConsensoState = {
            "issue_id": "TEST-2",
            "titulo": "Issue Teste",
            "descricao": "Descricao Teste",
            "historicos": [],
            "escala_fibonacci": "1, 2, 3, 5, 8, 13, 21, 34, 55",
            "protocolo": "simultaneo",
            "especializado": True,
            "papeis_agentes": None,
            "estimativas": [
                # R1
                {"agente": "backend", "papel": "backend", "modelo": "m", "story_points": 3.0, "justificativa": "j1", "rodada": 1, "tokens_entrada": 10, "tokens_saida": 10, "latencia_ms": 100, "data_hora": None},
                {"agente": "frontend", "papel": "frontend", "modelo": "m", "story_points": 5.0, "justificativa": "j1", "rodada": 1, "tokens_entrada": 10, "tokens_saida": 10, "latencia_ms": 100, "data_hora": None},
                {"agente": "qa", "papel": "qa", "modelo": "m", "story_points": 8.0, "justificativa": "j1", "rodada": 1, "tokens_entrada": 10, "tokens_saida": 10, "latencia_ms": 100, "data_hora": None},
                # R2 (mesmos lances por agente)
                {"agente": "backend", "papel": "backend", "modelo": "m", "story_points": 3.0, "justificativa": "j2", "rodada": 2, "tokens_entrada": 10, "tokens_saida": 10, "latencia_ms": 100, "data_hora": None},
                {"agente": "frontend", "papel": "frontend", "modelo": "m", "story_points": 5.0, "justificativa": "j2", "rodada": 2, "tokens_entrada": 10, "tokens_saida": 10, "latencia_ms": 100, "data_hora": None},
                {"agente": "qa", "papel": "qa", "modelo": "m", "story_points": 8.0, "justificativa": "j2", "rodada": 2, "tokens_entrada": 10, "tokens_saida": 10, "latencia_ms": 100, "data_hora": None},
            ],
            "rodada": 2,
            "convergiu": False,
            "motivo_convergencia": None,
            "dispersao_degraus": None,
            "rodadas_consecutivas_convergidas": None,
            "consenso": 0.0,
            "contexto_rodada_atual": None,
            "historico_rodadas": [],
        }

        updates = _no_avaliar_consenso(state, metodo="mediana", max_rodadas=3)
        self.assertTrue(updates["convergiu"])
        self.assertTrue("estagnacao_r2_mediana" in updates["motivo_convergencia"])
        self.assertEqual(updates["consenso"], 5.0)  # Mediana(3, 5, 8) = 5.0

        state.update(updates)
        decisao = _rotear_rodada(state, max_rodadas=3, deliberacao=True)
        self.assertEqual(decisao, "__end__")

    def test_fluxo_avanco_r2_e_desempate_r3(self):
        # R1 divergente, R2 mudou votos (não estagnou), R3 sem unanimidade -> desempate mediana
        state: ConsensoState = {
            "issue_id": "TEST-3",
            "titulo": "Issue Teste",
            "descricao": "Descricao Teste",
            "historicos": [],
            "escala_fibonacci": "1, 2, 3, 5, 8, 13, 21, 34, 55",
            "protocolo": "simultaneo",
            "especializado": True,
            "papeis_agentes": None,
            "estimativas": [
                # R1
                {"agente": "backend", "papel": "backend", "modelo": "m", "story_points": 3.0, "justificativa": "j1", "rodada": 1, "tokens_entrada": 10, "tokens_saida": 10, "latencia_ms": 100, "data_hora": None},
                {"agente": "frontend", "papel": "frontend", "modelo": "m", "story_points": 5.0, "justificativa": "j1", "rodada": 1, "tokens_entrada": 10, "tokens_saida": 10, "latencia_ms": 100, "data_hora": None},
                {"agente": "qa", "papel": "qa", "modelo": "m", "story_points": 8.0, "justificativa": "j1", "rodada": 1, "tokens_entrada": 10, "tokens_saida": 10, "latencia_ms": 100, "data_hora": None},
                # R2 (backend mudou para 5, qa mudou para 13 -> não estagnou)
                {"agente": "backend", "papel": "backend", "modelo": "m", "story_points": 5.0, "justificativa": "j2", "rodada": 2, "tokens_entrada": 10, "tokens_saida": 10, "latencia_ms": 100, "data_hora": None},
                {"agente": "frontend", "papel": "frontend", "modelo": "m", "story_points": 5.0, "justificativa": "j2", "rodada": 2, "tokens_entrada": 10, "tokens_saida": 10, "latencia_ms": 100, "data_hora": None},
                {"agente": "qa", "papel": "qa", "modelo": "m", "story_points": 13.0, "justificativa": "j2", "rodada": 2, "tokens_entrada": 10, "tokens_saida": 10, "latencia_ms": 100, "data_hora": None},
            ],
            "rodada": 2,
            "convergiu": False,
            "motivo_convergencia": None,
            "dispersao_degraus": None,
            "rodadas_consecutivas_convergidas": None,
            "consenso": 0.0,
            "contexto_rodada_atual": None,
            "historico_rodadas": [],
        }

        updates_r2 = _no_avaliar_consenso(state, metodo="mediana", max_rodadas=3)
        self.assertFalse(updates_r2["convergiu"])
        self.assertTrue("divergencia_r2_avancar_r3" in updates_r2["motivo_convergencia"])

        state.update(updates_r2)
        decisao_r2 = _rotear_rodada(state, max_rodadas=3, deliberacao=True)
        self.assertEqual(decisao_r2, "preparar")  # Avança para R3!

        # Adiciona lances da R3 (com divergência persistente 5, 5, 8)
        state["estimativas"].extend([
            {"agente": "backend", "papel": "backend", "modelo": "m", "story_points": 5.0, "justificativa": "j3", "rodada": 3, "tokens_entrada": 10, "tokens_saida": 10, "latencia_ms": 100, "data_hora": None},
            {"agente": "frontend", "papel": "frontend", "modelo": "m", "story_points": 5.0, "justificativa": "j3", "rodada": 3, "tokens_entrada": 10, "tokens_saida": 10, "latencia_ms": 100, "data_hora": None},
            {"agente": "qa", "papel": "qa", "modelo": "m", "story_points": 8.0, "justificativa": "j3", "rodada": 3, "tokens_entrada": 10, "tokens_saida": 10, "latencia_ms": 100, "data_hora": None},
        ])

        updates_r3 = _no_avaliar_consenso(state, metodo="mediana", max_rodadas=3)
        self.assertFalse(updates_r3["convergiu"])
        self.assertTrue("desempate_mediana_r3" in updates_r3["motivo_convergencia"])
        self.assertEqual(updates_r3["consenso"], 5.0)  # Mediana(5, 5, 8) = 5.0

        state.update(updates_r3)
        decisao_r3 = _rotear_rodada(state, max_rodadas=3, deliberacao=True)
        self.assertEqual(decisao_r3, "__end__")  # Termina pelo limite de rodadas

    def test_unanimidade_na_r3_marca_convergiu(self):
        base = lambda ag, sp, r: {"agente": ag, "papel": ag, "modelo": "m", "story_points": sp, "justificativa": "j", "rodada": r,
                                  "tokens_entrada": 1, "tokens_saida": 1, "latencia_ms": 1, "data_hora": None}
        ests = [base("backend", 3.0, 1), base("frontend", 5.0, 1), base("qa", 8.0, 1),
                base("backend", 5.0, 2), base("frontend", 5.0, 2), base("qa", 8.0, 2),
                base("backend", 5.0, 3), base("frontend", 5.0, 3), base("qa", 5.0, 3)]
        state = {"estimativas": ests, "rodada": 3, "escala_fibonacci": "1, 2, 3, 5, 8, 13, 21, 34, 55", "historico_rodadas": []}
        up = _no_avaliar_consenso(state, metodo="mediana", max_rodadas=3)
        self.assertTrue(up["convergiu"])
        self.assertEqual(up["motivo_convergencia"], "consenso_unanime_r3")
        self.assertEqual(up["consenso"], 5.0)
        self.assertEqual(up["rodada"], 4)                 # contador já avançado: rodadas usadas = rodada - 1 = 3

    def test_registro_da_rodada_tem_campos_para_a_analise(self):
        base = lambda ag, sp: {"agente": ag, "papel": ag, "modelo": "m", "story_points": sp, "justificativa": "j", "rodada": 1,
                               "tokens_entrada": 1, "tokens_saida": 1, "latencia_ms": 1, "data_hora": None}
        state = {"estimativas": [base("backend", 3.0), base("frontend", 5.0), base("qa", 13.0)], "rodada": 1,
                 "escala_fibonacci": "1, 2, 3, 5, 8, 13, 21, 34, 55", "historico_rodadas": []}
        reg = _no_avaliar_consenso(state, metodo="mediana", max_rodadas=3)["historico_rodadas"][0]
        for k in ("rodada", "valores", "lances_por_agente", "consenso", "sigma_sp", "dispersao_degraus", "motivo_convergencia", "convergiu", "extremos"):
            self.assertIn(k, reg)
        self.assertEqual(reg["dispersao_degraus"], 3)
        self.assertEqual(reg["consenso"], 5.0)

    def test_sem_deliberacao_encerra_apos_a_r1(self):
        state = {"rodada": 2, "convergiu": False}
        self.assertEqual(_rotear_rodada(state, max_rodadas=3, deliberacao=False), "__end__")

    def test_roteamento_para_apos_r_max(self):
        self.assertEqual(_rotear_rodada({"rodada": 4, "convergiu": False}, max_rodadas=3, deliberacao=True), "__end__")
        self.assertEqual(_rotear_rodada({"rodada": 3, "convergiu": False}, max_rodadas=3, deliberacao=True), "preparar")


if __name__ == "__main__":
    unittest.main()
