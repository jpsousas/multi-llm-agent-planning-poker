import unittest
from src.consensus import (avaliar_convergencia_rodada, parse_escala, agregar, arredondar_fibonacci,
                           distancia_degraus, formatar_resumo_debate)

class TestNovasRegrasParada(unittest.TestCase):
    def setUp(self):
        self.escala = parse_escala("1, 2, 3, 5, 8, 13, 21, 34, 55")   # SIS-ASTROS (único dataset)

    def test_unanimidade_r1(self):
        valores = [5.0, 5.0, 5.0]
        lances = {"backend": 5.0, "frontend": 5.0, "qa": 5.0}
        convergiu, motivo, disp = avaliar_convergencia_rodada(
            valores=valores,
            rodada=1,
            escala=self.escala,
            lances_atuais=lances,
            lances_anteriores=None
        )
        self.assertTrue(convergiu)
        self.assertEqual(motivo, "consenso_unanime_r1")
        self.assertEqual(disp, 0)
        self.assertEqual(agregar(valores, self.escala, "mediana"), 5.0)

    def test_divergencia_r1(self):
        valores = [3.0, 5.0, 8.0]
        lances = {"backend": 3.0, "frontend": 5.0, "qa": 8.0}
        convergiu, motivo, disp = avaliar_convergencia_rodada(
            valores=valores,
            rodada=1,
            escala=self.escala,
            lances_atuais=lances,
            lances_anteriores=None
        )
        self.assertFalse(convergiu)
        self.assertTrue("divergencia_r1_abrir_debate" in motivo)

    def test_unanimidade_r2(self):
        valores = [8.0, 8.0, 8.0]
        lances_r2 = {"backend": 8.0, "frontend": 8.0, "qa": 8.0}
        lances_r1 = {"backend": 3.0, "frontend": 5.0, "qa": 8.0}
        convergiu, motivo, disp = avaliar_convergencia_rodada(
            valores=valores,
            rodada=2,
            escala=self.escala,
            lances_atuais=lances_r2,
            lances_anteriores=lances_r1
        )
        self.assertTrue(convergiu)
        self.assertEqual(motivo, "consenso_unanime_r2")
        self.assertEqual(agregar(valores, self.escala, "mediana"), 8.0)

    def test_estagnacao_r2(self):
        valores_r2 = [3.0, 5.0, 8.0]
        lances_r1 = {"backend": 3.0, "frontend": 5.0, "qa": 8.0}
        lances_r2 = {"backend": 3.0, "frontend": 5.0, "qa": 8.0}
        convergiu, motivo, disp = avaliar_convergencia_rodada(
            valores=valores_r2,
            rodada=2,
            escala=self.escala,
            lances_atuais=lances_r2,
            lances_anteriores=lances_r1
        )
        self.assertTrue(convergiu)
        self.assertTrue("estagnacao_r2_mediana" in motivo)
        # Mediana entre 3, 5, 8 é 5.0
        self.assertEqual(agregar(valores_r2, self.escala, "mediana"), 5.0)

    def test_mudanca_r2_sem_consenso(self):
        valores_r2 = [5.0, 5.0, 8.0]
        lances_r1 = {"backend": 3.0, "frontend": 5.0, "qa": 8.0}
        lances_r2 = {"backend": 5.0, "frontend": 5.0, "qa": 8.0}
        convergiu, motivo, disp = avaliar_convergencia_rodada(
            valores=valores_r2,
            rodada=2,
            escala=self.escala,
            lances_atuais=lances_r2,
            lances_anteriores=lances_r1
        )
        # Não estagnou (backend mudou de 3 para 5) e não é unânime -> vai para R3
        self.assertFalse(convergiu)
        self.assertTrue("divergencia_r2_avancar_r3" in motivo)

    def test_unanimidade_r3(self):
        valores_r3 = [5.0, 5.0, 5.0]
        lances_r2 = {"backend": 5.0, "frontend": 5.0, "qa": 8.0}
        lances_r3 = {"backend": 5.0, "frontend": 5.0, "qa": 5.0}
        convergiu, motivo, disp = avaliar_convergencia_rodada(
            valores=valores_r3,
            rodada=3,
            escala=self.escala,
            lances_atuais=lances_r3,
            lances_anteriores=lances_r2
        )
        self.assertTrue(convergiu)
        self.assertEqual(motivo, "consenso_unanime_r3")
        self.assertEqual(agregar(valores_r3, self.escala, "mediana"), 5.0)

    def test_r3_sem_consenso_final_mediana(self):
        valores_r3 = [5.0, 5.0, 8.0]
        lances_r2 = {"backend": 5.0, "frontend": 3.0, "qa": 8.0}
        lances_r3 = {"backend": 5.0, "frontend": 5.0, "qa": 8.0}
        convergiu, motivo, disp = avaliar_convergencia_rodada(
            valores=valores_r3,
            rodada=3,
            escala=self.escala,
            lances_atuais=lances_r3,
            lances_anteriores=lances_r2
        )
        self.assertFalse(convergiu)
        self.assertTrue("desempate_mediana_r3" in motivo)
        # Mediana entre 5, 5, 8 é 5.0
        self.assertEqual(agregar(valores_r3, self.escala, "mediana"), 5.0)

    def test_mediana_de_tres_lances_ja_pertence_a_escala(self):
        for trio in ([13.0, 21.0, 34.0], [1.0, 55.0, 8.0], [3.0, 3.0, 55.0]):
            self.assertIn(agregar(trio, self.escala, "mediana"), self.escala)
        self.assertEqual(agregar([13.0, 21.0, 34.0], self.escala, "mediana"), 21.0)

    def test_estagnacao_so_vale_na_r2(self):
        """Na R3, lances iguais aos da R2 não contam como estagnação: encerra pelo teto (mediana)."""
        lances = {"backend": 3.0, "frontend": 5.0, "qa": 8.0}
        convergiu, motivo, _ = avaliar_convergencia_rodada(
            [3.0, 5.0, 8.0], 3, self.escala, lances_atuais=lances, lances_anteriores=lances)
        self.assertFalse(convergiu)
        self.assertIn("desempate_mediana_r3", motivo)

    def test_estagnacao_exige_os_mesmos_agentes(self):
        r1 = {"backend": 3.0, "frontend": 5.0, "qa": 8.0}
        r2 = {"backend": 3.0, "frontend": 5.0, "outro": 8.0}
        convergiu, motivo, _ = avaliar_convergencia_rodada([3.0, 5.0, 8.0], 2, self.escala, lances_atuais=r2, lances_anteriores=r1)
        self.assertFalse(convergiu)
        self.assertIn("divergencia_r2_avancar_r3", motivo)

    def test_sem_lances_anteriores_nao_ha_estagnacao(self):
        convergiu, motivo, _ = avaliar_convergencia_rodada([3.0, 5.0, 8.0], 2, self.escala)
        self.assertFalse(convergiu)

    def test_dispersao_em_degraus_da_escala(self):
        # 3 -> índice 2; 13 -> índice 5: amplitude 3 degraus (e não 10 "horas")
        _, _, disp = avaliar_convergencia_rodada([3.0, 5.0, 13.0], 1, self.escala)
        self.assertEqual(disp, 3)
        _, _, disp = avaliar_convergencia_rodada([21.0, 34.0, 34.0], 1, self.escala)
        self.assertEqual(disp, 1)

    def test_arredondar_fibonacci_e_distancia(self):
        self.assertEqual(arredondar_fibonacci(6.0, self.escala), 5.0)
        self.assertEqual(arredondar_fibonacci(100.0, self.escala), 55.0)
        self.assertEqual(distancia_degraus(1, 55, self.escala), 8)
        self.assertEqual(distancia_degraus(8, 13, self.escala), 1)

    def test_agregacao_media_e_moda(self):
        self.assertEqual(agregar([5.0, 5.0, 13.0], self.escala, "moda"), 5.0)
        self.assertEqual(agregar([3.0, 5.0, 13.0], self.escala, "media"), 8.0)   # média 7 -> degrau 8
        with self.assertRaises(ValueError):
            agregar([1.0], self.escala, "xyz")

    def test_formatar_resumo_do_debate_lista_os_tres_lances(self):
        ests = [{"papel": "frontend", "modelo": "m", "story_points": 3.0, "justificativa": "tela simples", "agente": "f"},
                {"papel": "backend", "modelo": "m", "story_points": 13.0, "justificativa": "API nova", "agente": "b"},
                {"papel": "qa", "modelo": "m", "story_points": 5.0, "justificativa": "regressão", "agente": "q"}]
        txt = formatar_resumo_debate(ests, 1)
        for trecho in ("**Frontend**", "**Backend**", "**Qa**", "3 SP", "13 SP", "tela simples", "API nova", "Divergência detectada"):
            self.assertIn(trecho, txt)
        unanime = [dict(e, story_points=5.0) for e in ests]
        self.assertNotIn("Divergência detectada", formatar_resumo_debate(unanime, 1))


if __name__ == "__main__":
    unittest.main()
