import unittest
from src.consensus import avaliar_convergencia_rodada, parse_escala, agregar

class TestNovasRegrasParada(unittest.TestCase):
    def setUp(self):
        self.escala_tawos = parse_escala("0, 1, 2, 3, 5, 8, 13, 20, 40, 100")
        self.escala_astros = parse_escala("1, 2, 3, 5, 8, 13, 21, 34, 55")

    def test_unanimidade_r1(self):
        valores = [5.0, 5.0, 5.0]
        lances = {"backend": 5.0, "frontend": 5.0, "qa": 5.0}
        convergiu, motivo, disp = avaliar_convergencia_rodada(
            valores=valores,
            rodada=1,
            escala=self.escala_tawos,
            lances_atuais=lances,
            lances_anteriores=None
        )
        self.assertTrue(convergiu)
        self.assertEqual(motivo, "consenso_unanime_r1")
        self.assertEqual(disp, 0)
        self.assertEqual(agregar(valores, self.escala_tawos, "mediana"), 5.0)

    def test_divergencia_r1(self):
        valores = [3.0, 5.0, 8.0]
        lances = {"backend": 3.0, "frontend": 5.0, "qa": 8.0}
        convergiu, motivo, disp = avaliar_convergencia_rodada(
            valores=valores,
            rodada=1,
            escala=self.escala_tawos,
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
            escala=self.escala_tawos,
            lances_atuais=lances_r2,
            lances_anteriores=lances_r1
        )
        self.assertTrue(convergiu)
        self.assertEqual(motivo, "consenso_unanime_r2")
        self.assertEqual(agregar(valores, self.escala_tawos, "mediana"), 8.0)

    def test_estagnacao_r2(self):
        valores_r2 = [3.0, 5.0, 8.0]
        lances_r1 = {"backend": 3.0, "frontend": 5.0, "qa": 8.0}
        lances_r2 = {"backend": 3.0, "frontend": 5.0, "qa": 8.0}
        convergiu, motivo, disp = avaliar_convergencia_rodada(
            valores=valores_r2,
            rodada=2,
            escala=self.escala_tawos,
            lances_atuais=lances_r2,
            lances_anteriores=lances_r1
        )
        self.assertTrue(convergiu)
        self.assertTrue("estagnacao_r2_mediana" in motivo)
        # Mediana entre 3, 5, 8 é 5.0
        self.assertEqual(agregar(valores_r2, self.escala_tawos, "mediana"), 5.0)

    def test_mudanca_r2_sem_consenso(self):
        valores_r2 = [5.0, 5.0, 8.0]
        lances_r1 = {"backend": 3.0, "frontend": 5.0, "qa": 8.0}
        lances_r2 = {"backend": 5.0, "frontend": 5.0, "qa": 8.0}
        convergiu, motivo, disp = avaliar_convergencia_rodada(
            valores=valores_r2,
            rodada=2,
            escala=self.escala_tawos,
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
            escala=self.escala_tawos,
            lances_atuais=lances_r3,
            lances_anteriores=lances_r2
        )
        self.assertTrue(convergiu)
        self.assertEqual(motivo, "consenso_unanime_r3")
        self.assertEqual(agregar(valores_r3, self.escala_tawos, "mediana"), 5.0)

    def test_r3_sem_consenso_final_mediana(self):
        valores_r3 = [5.0, 5.0, 8.0]
        lances_r2 = {"backend": 5.0, "frontend": 3.0, "qa": 8.0}
        lances_r3 = {"backend": 5.0, "frontend": 5.0, "qa": 8.0}
        convergiu, motivo, disp = avaliar_convergencia_rodada(
            valores=valores_r3,
            rodada=3,
            escala=self.escala_tawos,
            lances_atuais=lances_r3,
            lances_anteriores=lances_r2
        )
        self.assertFalse(convergiu)
        self.assertTrue("desempate_mediana_r3" in motivo)
        # Mediana entre 5, 5, 8 é 5.0
        self.assertEqual(agregar(valores_r3, self.escala_tawos, "mediana"), 5.0)

    def test_sis_astros_escala(self):
        # Testando com a escala do SIS-ASTROS [1, 2, 3, 5, 8, 13, 21, 34, 55]
        valores_r3 = [13.0, 21.0, 34.0]
        lances_r3 = {"backend": 13.0, "frontend": 21.0, "qa": 34.0}
        lances_r2 = {"backend": 8.0, "frontend": 21.0, "qa": 34.0}
        convergiu, motivo, disp = avaliar_convergencia_rodada(
            valores=valores_r3,
            rodada=3,
            escala=self.escala_astros,
            lances_atuais=lances_r3,
            lances_anteriores=lances_r2
        )
        self.assertFalse(convergiu)
        self.assertTrue("desempate_mediana_r3" in motivo)
        self.assertEqual(agregar(valores_r3, self.escala_astros, "mediana"), 21.0)

if __name__ == "__main__":
    unittest.main()
