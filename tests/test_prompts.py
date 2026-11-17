"""
tests/test_prompts.py
=====================
Prompt de sistema e prompt do item (src/prompts.py), sem LLM.

    python -m unittest tests.test_prompts -v
"""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("TOKENIZADOR", "none")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompts import PAPEIS, PROMPT_BASE, montar_prompt_item, montar_system_prompt
from src.limpeza_texto import limpar_descricao

ESCALA = "1, 2, 3, 5, 8, 13, 21, 34, 55"


class TestSistema(unittest.TestCase):

    def test_escala_horas_e_formato_json(self):
        p = montar_system_prompt("neutro", ESCALA)
        self.assertIn(ESCALA, p)
        self.assertIn("horas", p)
        self.assertNotIn("{papel}", p)
        self.assertNotIn("{escala_fibonacci}", p)
        # a justificativa (CoT) vem ANTES da nota, e a chave lida pelo agents.py continua story_points
        self.assertLess(p.index('"justificativa"'), p.index('"story_points"'))

    def test_papeis(self):
        self.assertNotIn("perspectiva", montar_system_prompt("neutro", ESCALA))
        for papel, trecho in (("backend", "Backend"), ("frontend", "Frontend"), ("qa", "QA")):
            p = montar_system_prompt(papel, ESCALA)
            self.assertIn("perspectiva", p)
            self.assertIn(trecho, p)
        self.assertEqual(set(PAPEIS), {"neutro", "backend", "frontend", "qa"})

    def test_papel_desconhecido_cai_no_neutro(self):
        self.assertEqual(montar_system_prompt("xyz", ESCALA), montar_system_prompt("neutro", ESCALA))

    def test_prompt_nao_depende_da_condicao(self):
        """O mesmo papel gera exatamente o mesmo texto (protocolo/limpeza não entram no prompt de sistema)."""
        self.assertEqual(montar_system_prompt("qa", ESCALA), montar_system_prompt("qa", ESCALA))

    def test_prompts_so_diferem_na_persona(self):
        base = montar_system_prompt("neutro", ESCALA)
        for papel in ("backend", "frontend", "qa"):
            p = montar_system_prompt(papel, ESCALA)
            self.assertEqual(len(p) - len(base), len(PAPEIS[papel]))

    def test_marcadores_da_limpeza_estao_explicados(self):
        """Todo marcador que a limpeza/truncamento pode produzir é descrito no prompt."""
        bloco = "\n".join(f"l{i}" for i in range(30))
        texto = ("veja http://x.com/a e ![img](a.png) " + "z" * 60 + " "
                 + "/aaa/bbb/" + "c" * 60 + " com.empresa.pacote.Classe" + "d" * 40 + ".metodo\n{code}\n" + bloco + "\n{code}")
        limpo = limpar_descricao(texto)
        for marcador in ("[link]", "[imagem]", "[código/log]", "linhas omitidas", "[token-longo]", "texto truncado"):
            self.assertIn(marcador, PROMPT_BASE, marcador)
        self.assertIn("[link]", limpo)
        self.assertIn("[imagem]", limpo)
        self.assertIn("[código/log]", limpo)
        self.assertIn("…", PROMPT_BASE)       # caminhos/nomes reduzidos ("…/arquivo", "…Classe.metodo")
        self.assertIn("…", limpo)


class TestItem(unittest.TestCase):

    def test_alvo_completo_sem_corte(self):
        longa = "palavra " * 5000
        p = montar_prompt_item("Título", longa)
        self.assertIn(longa, p)
        self.assertIn("**Título:** Título", p)

    def test_lembrete_final_de_formato(self):
        p = montar_prompt_item("T", "Corrigir o menu Medidas que fecha ao reabrir.", contexto_rodada="resumo", rodada=2)
        ultimo = p.strip().splitlines()[-1]
        self.assertIn("JSON", ultimo)
        self.assertIn("story_points", ultimo)
        self.assertIn("NÃO deve resolver", p)
        self.assertGreater(p.index("NÃO deve resolver"), p.index("Item do Backlog a Estimar"))   # depois da descrição

    def test_sem_descricao(self):
        self.assertIn("Sem descrição detalhada", montar_prompt_item("T", ""))

    def test_historicos_rag_com_estimativa_e_esforco(self):
        hist = [{"titulo": "A", "descricao": "d1", "story_points": 5.0, "esforco_real_horas": 6.0, "similaridade": 0.8},
                {"titulo": "B", "descricao": "", "story_points": 8.0, "similaridade": 0.7}]
        p = montar_prompt_item("T", "D", historicos=hist)
        self.assertIn("Issues Históricas Mais Parecidas", p)
        self.assertIn("Estimativa Planejada: 5 h | Esforço Real Gasto: 6 h", p)
        linha_b = next(l for l in p.splitlines() if l.startswith('- "B"'))
        self.assertIn("Estimativa Planejada: 8 h", linha_b)
        self.assertNotIn("Esforço Real", linha_b)               # sem Time Spent, não inventa
        # o item a estimar vem depois dos exemplos
        self.assertLess(p.index("Issues Históricas"), p.index("Item do Backlog a Estimar"))

    def test_few_shot_usa_outro_cabecalho(self):
        p = montar_prompt_item("T", "D", historicos=[{"titulo": "A", "story_points": 3.0}])
        self.assertIn("Scale-Aware", p)

    def test_contexto_por_rodada_e_protocolo(self):
        r1_sim = montar_prompt_item("T", "D")
        self.assertNotIn("Debate", r1_sim)
        seq = montar_prompt_item("T", "D", contexto_rodada="Frontend: 3", rodada=1, protocolo="sequencial")
        self.assertIn("Ordem Sequencial", seq)
        debate = montar_prompt_item("T", "D", contexto_rodada="resumo", rodada=2, protocolo="simultaneo")
        self.assertIn("Contexto do Debate", debate)
        debate_seq = montar_prompt_item("T", "D", contexto_rodada="resumo", rodada=2, protocolo="sequencial")
        self.assertIn("Contexto do Debate", debate_seq)


if __name__ == "__main__":
    unittest.main(verbosity=2)