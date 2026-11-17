"""
tests/test_limpeza_texto.py
===========================
Limpeza determinística e orçamento de tokens (src/limpeza_texto.py). Só dados SINTÉTICOS, sem rede.

    TOKENIZADOR=none python -m unittest tests.test_limpeza_texto -v
"""
import math
import os
import sys
import unittest
from pathlib import Path

os.environ["TOKENIZADOR"] = "none"          # estimativa ceil(chars/3): não depende de internet/transformers
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.limpeza_texto import (LINHA_BLOCO_MAX, LINHAS_BLOCO, TOKEN_LONGO, contar_tokens, limpar_descricao,
                               preparar_descricao, truncar_para_orcamento)

PROSA = "Ajustar a validação do campo senha.\n\nO usuário deve receber mensagem clara quando a senha tiver menos de 8 caracteres."
BLOCO_LONGO = "\n".join(f"linha {i} do log com informação" for i in range(40))
URL_LONGA = "https://exemplo.com/" + "a" * 100


def balanceado(x):
    return x.count("(") == x.count(")") and x.count("[") == x.count("]")


class TestProsaEBlocos(unittest.TestCase):

    def test_prosa_curta_nao_muda(self):
        self.assertEqual(limpar_descricao(PROSA), PROSA)

    def test_bloco_code_longo_mantem_so_o_comeco(self):
        m = {}
        r = limpar_descricao(f"Erro na rotina.\n{{code}}\n{BLOCO_LONGO}\n{{code}}\nFim.", meta=m)
        self.assertIn("linha 5 do log", r)
        self.assertNotIn("linha 6 do log", r)
        self.assertIn("[... 34 linhas omitidas ...]", r)
        self.assertIn("[código/log]", r)
        self.assertEqual((m["blocos_omitidos"], m["linhas_omitidas"]), (1, 34))
        self.assertTrue(r.startswith("Erro na rotina.") and r.endswith("Fim."))

    def test_bloco_curto_fica_intacto(self):
        curto = "Veja:\n{code}\na=1\nb=2\n{code}\nok"
        self.assertEqual(limpar_descricao(curto), curto)

    def test_code_com_linguagem_noformat_e_cerca_markdown(self):
        self.assertIn("[código/log]", limpar_descricao("{code:java}\n" + BLOCO_LONGO + "\n{code}"))
        self.assertIn("[código/log]", limpar_descricao("{noformat}\n" + BLOCO_LONGO + "\n{noformat}"))
        self.assertIn("[... 34 linhas omitidas ...]", limpar_descricao("Antes\n```java\n" + BLOCO_LONGO + "\n```\nDepois"))

    def test_parametro_linhas_bloco(self):
        t = "{code}\n" + "\n".join(f"l{i}" for i in range(20)) + "\n{code}"
        self.assertIn("[... 17 linhas omitidas ...]", limpar_descricao(t, linhas_bloco=3))
        self.assertIn("[... 8 linhas omitidas ...]", limpar_descricao(t, linhas_bloco=12))
        self.assertEqual(LINHAS_BLOCO, 6)

    def test_linha_gigante_dentro_do_bloco_e_cortada(self):
        t = "{code}\n" + ("x" * 1000) + "\n" + "\n".join(f"l{i}" for i in range(10)) + "\n{code}"
        r = limpar_descricao(t)
        self.assertLessEqual(max(len(l) for l in r.splitlines()), LINHA_BLOCO_MAX + 5)

    def test_varios_blocos_somam_no_meta(self):
        t = f"{{code}}\n{BLOCO_LONGO}\n{{code}}\ntexto\n{{code}}\n{BLOCO_LONGO}\n{{code}}"
        m = {}
        limpar_descricao(t, meta=m)
        self.assertEqual((m["blocos_omitidos"], m["linhas_omitidas"]), (2, 68))


class TestHtmlUrlsTokens(unittest.TestCase):

    def test_html_removido_e_entidades_decodificadas(self):
        r = limpar_descricao("<p>Texto &amp; mais</p><br/><b>negrito</b> e List<String> fica")
        self.assertNotIn("<p>", r)
        self.assertNotIn("<b>", r)
        self.assertIn("Texto & mais", r)
        self.assertIn("negrito", r)
        self.assertIn("List<String>", r)                 # genéricos não são tocados

    def test_toda_url_vira_link(self):
        m = {}
        self.assertEqual(limpar_descricao(f"Veja {URL_LONGA} e http://x.io/ok", meta=m), "Veja [link] e [link]")
        self.assertEqual(m["urls"], 2)

    def test_caminho_nome_qualificado_e_hash(self):
        r = limpar_descricao("Arquivo C:\\projetos\\" + "pasta\\" * 10 + "Principal.java e hash " + "f" * 64)
        self.assertIn("…/Principal.java", r)
        self.assertIn("[token-longo]", r)
        r = limpar_descricao("Alterar br.ufsm.sistema.cadastro.ClienteService.salvarCliente para validar")
        self.assertIn("…ClienteService.salvarCliente", r)
        self.assertNotIn("br.ufsm", r)

    def test_limite_de_token_longo(self):
        curto = "a" * (TOKEN_LONGO - 1)
        self.assertEqual(limpar_descricao(f"x {curto} y"), f"x {curto} y")          # abaixo do limite: intacto
        self.assertIn("[token-longo]", limpar_descricao("x " + "a" * TOKEN_LONGO + " y"))

    def test_palavras_normais_do_portugues_nao_sao_tocadas(self):
        t = "Inconstitucionalissimamente, otorrinolaringologista e anticonstitucionalissimamente."
        self.assertEqual(limpar_descricao(t), t)

    def test_contadores_de_imagens_e_tokens_longos(self):
        m = {}
        limpar_descricao(f"![a](/uploads/{'f' * 32}/x.png) e {'h' * 50}", meta=m)
        self.assertEqual(m["imagens"], 1)
        self.assertEqual(m["tokens_longos"], 1)


class TestBalanceamento(unittest.TestCase):

    def test_parenteses_e_colchetes_balanceados(self):
        for entrada in (f"Veja ({URL_LONGA}) depois", f"[manual]({URL_LONGA}) fim",
                        f"Anexo ![image](/uploads/{'f' * 32}/captura_de_tela.png) ok",
                        f"Doc [arquivo](/uploads/{'e' * 32}/relatorio_final.pdf) ok",
                        f"(ver /uploads/{'d' * 40}/x.png) fim", f"Fim da frase {URL_LONGA}.", f"({'c' * 64})",
                        f"[{'c' * 64}]", f"([{'c' * 64}])"):
            r = limpar_descricao(entrada)
            self.assertTrue(balanceado(r), f"desbalanceado: {r!r}")

    def test_resultados_exatos(self):
        self.assertEqual(limpar_descricao(f"Veja ({URL_LONGA}) depois"), "Veja ([link]) depois")
        self.assertEqual(limpar_descricao(f"[manual]({URL_LONGA}) fim"), "manual [link] fim")
        self.assertIn("[imagem]", limpar_descricao(f"![image](/uploads/{'f' * 32}/captura.png)"))
        self.assertEqual(limpar_descricao(f"Fim da frase {URL_LONGA}."), "Fim da frase [link].")
        self.assertEqual(limpar_descricao("[site](http://x.io/ok) fica"), "site [link] fica")

    def test_colchetes_de_codigo_sem_url_nao_sao_tocados(self):
        self.assertEqual(limpar_descricao("valor arr[0](x) fica"), "valor arr[0](x) fica")

    def test_url_com_parenteses_proprios(self):
        r = limpar_descricao("Ver https://pt.wikipedia.org/wiki/Teste_(software) agora")
        self.assertEqual(r, "Ver [link] agora")


class TestRepeticaoEPropriedades(unittest.TestCase):

    def test_linhas_repetidas_consecutivas_colapsam(self):
        r = limpar_descricao("Esta linha se repete várias vezes no texto\n" * 10 + "fim")
        self.assertEqual(r.count("Esta linha se repete"), 1)
        self.assertTrue(r.endswith("fim"))

    def test_linhas_curtas_repetidas_ficam(self):
        t = "---\n---\n---\ntexto"
        self.assertEqual(limpar_descricao(t), t)

    def test_repeticao_nao_consecutiva_fica(self):
        linha = "Esta linha se repete várias vezes no texto"
        t = f"{linha}\noutra coisa\n{linha}"
        self.assertEqual(limpar_descricao(t).count(linha), 2)

    def test_json_minificado_e_reduzido(self):
        r = limpar_descricao('{"a":1,"b":[2,3],' * 400)
        self.assertLess(len(r), 400)

    def test_paragrafo_longo_de_prosa_nao_e_cortado(self):
        p = ("O sistema deve permitir que o usuário cadastre o cliente informando os dados pessoais e de contato. " * 30).strip()
        self.assertEqual(limpar_descricao(p), p)

    def test_normaliza_espacos_quebras_e_crlf(self):
        self.assertEqual(limpar_descricao("a   b\t\tc\r\n\r\n\r\n\r\nd\r\n"), "a b c\n\nd")

    def test_idempotente_e_deterministico(self):
        trace = "\n".join(f"    at com.empresa.Servico.metodo{i}(Servico.java:{i})" for i in range(50))
        ruim = f"A\n{{code}}\n{BLOCO_LONGO}\n{{code}}\n{trace}\n<p>B</p> https://e.com/{'z' * 80} ({'q' * 70})"
        um = limpar_descricao(ruim)
        self.assertEqual(limpar_descricao(um), um)
        self.assertEqual(limpar_descricao(ruim), um)

    def test_entrada_vazia_ou_none(self):
        self.assertEqual(limpar_descricao(None), "")
        self.assertEqual(limpar_descricao(""), "")
        self.assertEqual(limpar_descricao("\x00"), "")

    def test_limpar_nunca_aumenta_o_texto(self):
        for t in (PROSA, f"{{code}}\n{BLOCO_LONGO}\n{{code}}", f"Veja {URL_LONGA}", "x" * 200):
            self.assertLessEqual(len(limpar_descricao(t)), len(t) + 12)     # +12: o marcador "[código/log]" pode superar um bloco curtíssimo


class TestOrcamento(unittest.TestCase):

    def test_estimativa_sem_tokenizador(self):
        self.assertEqual(contar_tokens("abc" * 10), 10)
        self.assertEqual(contar_tokens("abcd"), math.ceil(4 / 3))
        self.assertEqual(contar_tokens(""), 0)

    def test_truncar_respeita_orcamento_e_marca(self):
        r, trunc = truncar_para_orcamento("palavra " * 5000, 500)
        self.assertTrue(trunc)
        self.assertLessEqual(contar_tokens(r), 500)
        self.assertTrue(r.endswith("[... texto truncado ...]"))

    def test_truncar_corta_em_limite_de_palavra(self):
        r, _ = truncar_para_orcamento("palavra " * 5000, 500)
        corpo = r.replace("\n[... texto truncado ...]", "")
        self.assertTrue(all(p == "palavra" for p in corpo.split()))

    def test_dentro_do_orcamento_nao_altera(self):
        r, trunc = truncar_para_orcamento(PROSA, 500)
        self.assertFalse(trunc)
        self.assertEqual(r, PROSA)

    def test_orcamento_minusculo_devolve_so_o_marcador(self):
        r, trunc = truncar_para_orcamento("palavra " * 100, 1)
        self.assertTrue(trunc)
        self.assertEqual(r, "[... texto truncado ...]")

    def test_preparar_descricao(self):
        multilinha = "\n".join(f"item {i} descreve um requisito diferente do sistema" for i in range(2000))
        pronto, meta = preparar_descricao(f"{{code}}\n{BLOCO_LONGO}\n{{code}}\n" + multilinha, 300)
        self.assertEqual(meta["truncado"], 1)
        self.assertLessEqual(meta["tokens_final"], 300)
        self.assertLess(meta["caracteres_final"], meta["caracteres_antes"])
        self.assertLessEqual(meta["caracteres_apos_limpeza"], meta["caracteres_antes"])
        self.assertTrue(all(isinstance(v, int) for v in meta.values()), "meta só tem contadores numéricos (sem texto)")
        self.assertEqual(meta["blocos_omitidos"], 1)

    def test_preparar_sem_truncar(self):
        pronto, meta = preparar_descricao(PROSA, 500)
        self.assertEqual(pronto, PROSA)
        self.assertEqual(meta["truncado"], 0)

    def test_funcao_contadora_injetada(self):
        pronto, meta = preparar_descricao("a b c d e f g h", 3, contar=lambda t: len(t.split()))
        self.assertEqual(meta["truncado"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)