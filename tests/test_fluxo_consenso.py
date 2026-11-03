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
            "escala_fibonacci": "0, 1, 2, 3, 5, 8, 13, 20, 40, 100",
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
            "escala_fibonacci": "0, 1, 2, 3, 5, 8, 13, 20, 40, 100",
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
            "escala_fibonacci": "0, 1, 2, 3, 5, 8, 13, 20, 40, 100",
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

if __name__ == "__main__":
    unittest.main()
