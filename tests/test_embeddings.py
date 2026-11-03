"""
tests/test_embeddings.py
========================
Teste isolado do módulo de recuperação semântica (RAG).

Pode ser executado SEM o CSV real do SIS-ASTROS — usa issues sintéticas em PT-BR
que simulam o domínio de gestão de projetos de software.

Execução (a partir da pasta multi-llm-agent-planning-poker):
    python -m unittest tests.test_embeddings -v

Ou diretamente:
    python tests/test_embeddings.py

O que testa:
  1. Geração de embeddings para uma lista de issues mock.
  2. Recuperação semântica: issues de autenticação devem ser mais similares
     entre si do que issues de relatórios.
  3. Filtro anti-leakage por IID: a própria issue não deve aparecer nos resultados.
  4. Filtro temporal: issues criadas DEPOIS da alvo não devem aparecer.
  5. Formato de saída: campos esperados pelo sistema de prompts (inclui esforço real).
  6. Cache: modelo carregado uma vez; cache distinto quando muda o conjunto indexado.
  7. Diagnóstico de distribuição: impressão legível da distribuição por SP.
  8. (Opcional) Integração com o CSV real em CAMINHO_SIS_ASTROS — pulado se não existir.

O cache dos testes vai para uma pasta temporária (não suja results/cache).
"""

import sys
import os
import tempfile
import unittest
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Garante que src/ seja encontrado
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import (
    construir_indice,
    buscar_issues_similares,
    formatar_historicos_rag,
    resetar_indice,
    obter_modelo_embeddings,
    _texto_para_embedding,
    _parse_data,
)

# ==============================================================================
# Fixtures: Issues sintéticas em PT-BR para simular o SIS-ASTROS
# ==============================================================================

def _data_iso(dias_atras: int) -> str:
    """Gera data ISO no passado relativo ao momento do teste."""
    return (datetime.now(timezone.utc) - timedelta(days=dias_atras)).strftime("%Y-%m-%dT%H:%M:%SZ")

ISSUES_MOCK = [
    # --- Cluster: Autenticação / Segurança ---
    {
        "ID": "101",
        "Title": "Implementar autenticação de usuários com JWT",
        "Description": "Criar endpoint de login com geração de token JWT e refresh token.",
        "Story_Point": 8.0,
        "Tempo_Gasto_Horas": 10.0,
        "Creation_Date": _data_iso(200),
    },
    {
        "ID": "102",
        "Title": "Adicionar autenticação OAuth2 com Google",
        "Description": "Integrar login social via Google usando o protocolo OAuth2.",
        "Story_Point": 13.0,
        "Tempo_Gasto_Horas": 16.0,
        "Creation_Date": _data_iso(190),
    },
    {
        "ID": "103",
        "Title": "Controle de sessão e logout automático",
        "Description": "Implementar expiração de sessão após inatividade e invalidação de tokens.",
        "Story_Point": 5.0,
        "Tempo_Gasto_Horas": 6.0,
        "Creation_Date": _data_iso(180),
    },
    # --- Cluster: Relatórios / Dashboards ---
    {
        "ID": "201",
        "Title": "Gerar relatório de horas trabalhadas por sprint",
        "Description": "Dashboard com horas registradas por desenvolvedor em cada sprint.",
        "Story_Point": 13.0,
        "Tempo_Gasto_Horas": 18.0,
        "Creation_Date": _data_iso(170),
    },
    {
        "ID": "202",
        "Title": "Exportar relatório de progresso em PDF",
        "Description": "Funcionalidade de exportação do relatório de andamento do projeto em formato PDF.",
        "Story_Point": 8.0,
        "Tempo_Gasto_Horas": 8.0,
        "Creation_Date": _data_iso(160),
    },
    {
        "ID": "203",
        "Title": "Gráfico de burndown do sprint atual",
        "Description": "Visualização em gráfico de linha do progresso restante versus o tempo planejado.",
        "Story_Point": 5.0,
        "Tempo_Gasto_Horas": 5.0,
        "Creation_Date": _data_iso(150),
    },
    # --- Cluster: CRUD / Formulários ---
    {
        "ID": "301",
        "Title": "Criar formulário de cadastro de novos usuários",
        "Description": "Tela de registro com campos de nome, e-mail, senha e confirmação de senha.",
        "Story_Point": 3.0,
        "Tempo_Gasto_Horas": 3.0,
        "Creation_Date": _data_iso(140),
    },
    {
        "ID": "302",
        "Title": "Editar perfil do usuário",
        "Description": "Permitir que o usuário atualize nome, foto de perfil e dados de contato.",
        "Story_Point": 2.0,
        "Tempo_Gasto_Horas": 2.0,
        "Creation_Date": _data_iso(130),
    },
    {
        "ID": "303",
        "Title": "Excluir conta de usuário e dados associados",
        "Description": "Funcionalidade de exclusão de conta com remoção de dados conforme LGPD.",
        "Story_Point": 5.0,
        "Tempo_Gasto_Horas": 7.0,
        "Creation_Date": _data_iso(120),
    },
    # --- Cluster: Infraestrutura / DevOps ---
    {
        "ID": "401",
        "Title": "Configurar pipeline de CI/CD no GitLab",
        "Description": "Criar .gitlab-ci.yml com stages de build, test e deploy automático.",
        "Story_Point": 8.0,
        "Tempo_Gasto_Horas": 9.0,
        "Creation_Date": _data_iso(110),
    },
    {
        "ID": "402",
        "Title": "Dockerizar a aplicação backend",
        "Description": "Criar Dockerfile e docker-compose para a API e banco de dados.",
        "Story_Point": 5.0,
        "Tempo_Gasto_Horas": 6.0,
        "Creation_Date": _data_iso(100),
    },
    {
        "ID": "403",
        "Title": "Configurar monitoramento com Prometheus e Grafana",
        "Description": "Setup de coleta de métricas da API e criação de dashboards de monitoramento.",
        "Story_Point": 13.0,
        "Tempo_Gasto_Horas": 15.0,
        "Creation_Date": _data_iso(90),
    },
]

# ==============================================================================
# Criação de CSV temporário para simular o arquivo real
# ==============================================================================

def _criar_csv_mock() -> str:
    """Cria um CSV temporário com o formato do SIS-ASTROS para testes."""
    import csv
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="")
    writer = csv.DictWriter(tmp, fieldnames=[
        "IID", "Title", "State", "Author", "Assignee",
        "Time Estimate", "Time Spent", "Created At", "Closed At",
        "Milestone", "Labels", "Description", "URL"
    ])
    writer.writeheader()
    for issue in ISSUES_MOCK:
        writer.writerow({
            "IID": issue["ID"],
            "Title": issue["Title"],
            "State": "closed",
            "Author": "dev@test.com",
            "Assignee": "dev@test.com",
            "Time Estimate": int(issue["Story_Point"]),
            "Time Spent": issue["Tempo_Gasto_Horas"],
            "Created At": issue["Creation_Date"],
            "Closed At": _data_iso(int(issue["ID"]) % 50),
            "Milestone": "",
            "Labels": "",
            "Description": issue["Description"],
            "URL": f"https://gitlab.example.com/issues/{issue['ID']}",
        })
    tmp.close()
    return tmp.name


# ==============================================================================
# Suite de Testes
# ==============================================================================

class TestEmbeddingsRAG(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Constrói o índice uma única vez para toda a suite (evita download repetido do modelo)."""
        resetar_indice()
        cls.csv_path = _criar_csv_mock()
        cls.pasta_cache = tempfile.TemporaryDirectory()
        print(f"\n[Setup] CSV temporário em: {cls.csv_path}")
        print("[Setup] Construindo índice de embeddings (pode levar ~10-30s na primeira vez para baixar o modelo)...")
        cls.indice = construir_indice(ISSUES_MOCK, cls.csv_path, diretorio_cache=cls.pasta_cache.name)
        print(f"[Setup] Índice pronto: {cls.indice['embeddings'].shape} embeddings")

    @classmethod
    def tearDownClass(cls):
        """Remove arquivos temporários após os testes."""
        os.unlink(cls.csv_path)
        cls.pasta_cache.cleanup()
        resetar_indice()

    # ------------------------------------------------------------------
    # 1. Estrutura do índice
    # ------------------------------------------------------------------
    def test_01_indice_tem_embeddings(self):
        """O índice deve conter um array numpy com shape (N, D)."""
        emb = self.indice["embeddings"]
        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.ndim, 2)
        self.assertEqual(emb.shape[0], len(ISSUES_MOCK))
        print(f"  [OK] Shape dos embeddings: {emb.shape}")

    def test_02_embeddings_normalizados(self):
        """Embeddings devem estar normalizados (norma ~ 1.0) para que o produto escalar seja cosseno."""
        normas = np.linalg.norm(self.indice["embeddings"], axis=1)
        np.testing.assert_allclose(normas, 1.0, atol=1e-5)
        print(f"  [OK] Norma média: {normas.mean():.6f}")

    def test_03_quantidade_de_itens(self):
        """O índice deve ter o mesmo número de itens que as issues de entrada."""
        self.assertEqual(len(self.indice["itens"]), len(ISSUES_MOCK))

    # ------------------------------------------------------------------
    # 2. Relevância semântica
    # ------------------------------------------------------------------
    def test_04_autenticacao_retorna_issues_do_mesmo_cluster(self):
        """Issue de autenticação deve recuperar outras issues de autenticação/segurança."""
        similares = buscar_issues_similares(
            titulo_alvo="Autenticação de usuários com token JWT",
            descricao_alvo="Endpoint de login e geração de access token seguro.",
            iid_alvo="NOVA-999",
            data_alvo=None,
            indice=self.indice,
            k=3,
        )
        self.assertEqual(len(similares), 3)
        ids_retornados = {s["ID"] for s in similares}
        # Espera que pelo menos 2 das 3 issues retornadas sejam do cluster de autenticação
        cluster_auth = {"101", "102", "103"}
        interseccao = ids_retornados & cluster_auth
        print(f"  [OK] Issues retornadas: {ids_retornados} | Cluster auth: {cluster_auth} | Interseção: {interseccao}")
        self.assertGreaterEqual(len(interseccao), 2,
            f"Esperava >=2 issues do cluster de autenticação, mas obteve: {ids_retornados}")

    def test_05_relatorio_retorna_issues_do_mesmo_cluster(self):
        """Issue de relatório deve recuperar outras issues de relatórios/dashboards."""
        similares = buscar_issues_similares(
            titulo_alvo="Exportar relatório mensal de atividades em PDF",
            descricao_alvo="Geração de relatório com métricas do projeto no formato PDF.",
            iid_alvo="NOVA-888",
            data_alvo=None,
            indice=self.indice,
            k=3,
        )
        ids_retornados = {s["ID"] for s in similares}
        cluster_relatorio = {"201", "202", "203"}
        interseccao = ids_retornados & cluster_relatorio
        print(f"  [OK] Issues retornadas: {ids_retornados} | Cluster relatório: {cluster_relatorio} | Interseção: {interseccao}")
        self.assertGreaterEqual(len(interseccao), 1,
            f"Esperava >=1 issue do cluster de relatórios, mas obteve: {ids_retornados}")

    def test_06_similaridade_intra_cluster_maior_que_inter_cluster(self):
        """A similaridade entre issues do mesmo cluster deve ser maior que entre clusters opostos."""
        from sentence_transformers import SentenceTransformer
        modelo = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        texto_auth = _texto_para_embedding("Implementar autenticação JWT", "Endpoint de login com token.")
        texto_relat = _texto_para_embedding("Gerar relatório de burndown", "Dashboard de progresso do sprint.")
        texto_auth2 = _texto_para_embedding("Adicionar login social OAuth2", "Login com Google e Facebook.")

        v_auth = modelo.encode([texto_auth], normalize_embeddings=True)[0]
        v_auth2 = modelo.encode([texto_auth2], normalize_embeddings=True)[0]
        v_relat = modelo.encode([texto_relat], normalize_embeddings=True)[0]

        sim_intra = float(v_auth @ v_auth2)   # auth <-> auth
        sim_inter = float(v_auth @ v_relat)   # auth <-> relatório

        print(f"  [OK] Similaridade auth<->auth: {sim_intra:.4f} | auth<->relatório: {sim_inter:.4f}")
        self.assertGreater(sim_intra, sim_inter,
            f"Esperava sim_intra ({sim_intra:.4f}) > sim_inter ({sim_inter:.4f})")

    # ------------------------------------------------------------------
    # 3. Filtros de integridade científica
    # ------------------------------------------------------------------
    def test_07_anti_leakage_por_iid(self):
        """A própria issue não deve aparecer nos resultados da busca."""
        issue_alvo = ISSUES_MOCK[0]  # ID "101"
        similares = buscar_issues_similares(
            titulo_alvo=issue_alvo["Title"],
            descricao_alvo=issue_alvo["Description"],
            iid_alvo=issue_alvo["ID"],
            data_alvo=None,
            indice=self.indice,
            k=5,
        )
        ids_retornados = {s["ID"] for s in similares}
        self.assertNotIn("101", ids_retornados,
            "A issue alvo (ID 101) não deve aparecer nos resultados (anti-leakage por IID)")
        print(f"  [OK] IID '101' ausente dos resultados: {ids_retornados}")

    def test_08_filtro_temporal_causal(self):
        """Issues criadas DEPOIS da data da issue alvo não devem aparecer nos resultados."""
        # Issue alvo com data de 120 dias atrás
        # Issues com ID 401, 402, 403 foram criadas em 110, 100, 90 dias atrás (DEPOIS)
        # Issues com ID 101..303 foram criadas em 200..130 dias atrás (ANTES)
        data_alvo = _data_iso(120)

        # Calcula quais IDs deveriam estar disponíveis (criados ANTES da data alvo)
        ids_elegíveis = {
            it["ID"] for it in ISSUES_MOCK
            if it["ID"] != "303"  # própria issue alvo (simulada)
            and (
                _parse_data(it["Creation_Date"]) is None
                or _parse_data(it["Creation_Date"]) < _parse_data(data_alvo)
            )
        }

        similares = buscar_issues_similares(
            titulo_alvo="Excluir conta de usuário e dados associados",
            descricao_alvo="Remoção de conta conforme LGPD.",
            iid_alvo="303",
            data_alvo=data_alvo,
            indice=self.indice,
            k=10,  # pede muitos para garantir que não vazou nada
        )

        ids_retornados = {s["ID"] for s in similares}
        # Nenhum dos IDs retornados deve ser de issues criadas depois da data alvo
        ids_futuros = {"401", "402", "403"}  # criadas em 110, 100, 90 dias atrás
        ids_vazados = ids_retornados & ids_futuros
        print(f"  [OK] IDs retornados: {ids_retornados} | IDs futuros (proibidos): {ids_futuros} | Vazados: {ids_vazados}")
        self.assertEqual(len(ids_vazados), 0,
            f"Issues do futuro vazaram para o contexto: {ids_vazados}")

    # ------------------------------------------------------------------
    # 4. Formato de saída
    # ------------------------------------------------------------------
    def test_09_campos_obrigatorios_na_saida(self):
        """Cada resultado deve ter os campos mínimos esperados pelo sistema de prompts."""
        similares = buscar_issues_similares(
            titulo_alvo="Pipeline CI/CD para deploy automático",
            descricao_alvo="Automatização de build e deploy via GitLab CI.",
            iid_alvo="NOVA-777",
            data_alvo=None,
            indice=self.indice,
            k=2,
        )
        self.assertEqual(len(similares), 2)
        for s in similares:
            self.assertIn("ID", s)
            self.assertIn("Title", s)
            self.assertIn("Story_Point", s)
            self.assertIn("similaridade", s)
            self.assertGreaterEqual(s["similaridade"], 0.0)
            self.assertLessEqual(s["similaridade"], 1.0)
            print(f"  [OK] [{s['ID']}] '{s['Title'][:50]}' -> SP={s['Story_Point']} sim={s['similaridade']:.3f}")

    def test_10_formato_historicos_rag_para_prompt(self):
        """formatar_historicos_rag deve retornar lista compatível com montar_prompt_item."""
        similares = buscar_issues_similares(
            titulo_alvo="Autenticação com token",
            descricao_alvo="",
            iid_alvo="NOVA-1",
            data_alvo=None,
            indice=self.indice,
            k=3,
        )
        historicos = formatar_historicos_rag(similares)
        for h in historicos:
            self.assertIn("titulo", h)
            self.assertIn("story_points", h)
            self.assertIn("similaridade", h)
            self.assertIsInstance(h["story_points"], float)
            print(f"  [OK] Histórico RAG: '{h['titulo'][:50]}' -> {h['story_points']} SP sim={h['similaridade']}")

    # ------------------------------------------------------------------
    # 5. Diagnóstico de distribuição (informativo, não falha)
    # ------------------------------------------------------------------
    def test_11_diagnostico_distribuicao_dataset_mock(self):
        """Imprime a distribuição de SPs no dataset mock — verificação visual."""
        from collections import Counter
        contagem = Counter(int(it["Story_Point"]) for it in ISSUES_MOCK)
        print("\n  === Distribuição do Dataset Mock ===")
        for sp in sorted(contagem):
            barra = "#" * contagem[sp]
            print(f"  {sp:>3} SP: {contagem[sp]:>2} issues  {barra}")
        print(f"  Total: {sum(contagem.values())} issues")
        self.assertGreater(len(contagem), 3, "Deve haver pelo menos 4 valores de SP distintos")

    def test_12_k_maior_que_disponiveis(self):
        """Pedir K=100 com apenas N issues disponíveis deve retornar no máximo N-1 (sem a própria)."""
        similares = buscar_issues_similares(
            titulo_alvo="Teste",
            descricao_alvo="",
            iid_alvo="101",  # exclui a de ID 101
            data_alvo=None,
            indice=self.indice,
            k=100,
        )
        self.assertLessEqual(len(similares), len(ISSUES_MOCK) - 1)
        print(f"  [OK] K=100 com {len(ISSUES_MOCK)} issues retornou {len(similares)} resultados (correto)")

    # ------------------------------------------------------------------
    # 6. Esforço real e cache
    # ------------------------------------------------------------------
    def test_13_historico_mostra_estimativa_e_esforco_real(self):
        """Cada exemplo traz estimativa planejada (SP) e esforço real (h) quando registrado."""
        similares = buscar_issues_similares(
            titulo_alvo="Login com token JWT", descricao_alvo="", iid_alvo="NOVA-2",
            data_alvo=None, indice=self.indice, k=3,
        )
        for h in formatar_historicos_rag(similares):
            original = next(it for it in ISSUES_MOCK if it["ID"] == h["id"])
            self.assertEqual(h["story_points"], original["Story_Point"])
            self.assertEqual(h["esforco_real_horas"], original["Tempo_Gasto_Horas"])
            self.assertIn("descricao", h)

    def test_14_sem_esforco_real_nao_inventa_valor(self):
        """Sem Time Spent registrado, o exemplo não exibe esforço real (não copia a estimativa)."""
        item = dict(ISSUES_MOCK[0], Tempo_Gasto_Horas=None, similaridade=0.9)
        h = formatar_historicos_rag([item])[0]
        self.assertNotIn("esforco_real_horas", h)

    def test_15_modelo_de_embeddings_carregado_uma_vez(self):
        """O SentenceTransformer é reutilizado entre buscas (não recarrega por issue)."""
        self.assertIs(obter_modelo_embeddings(), obter_modelo_embeddings())

    def test_16_cache_depende_do_conjunto_indexado(self):
        """Indexar outro subconjunto gera outro cache (não reaproveita embeddings errados)."""
        subconjunto = ISSUES_MOCK[:6]
        indice_sub = construir_indice(subconjunto, self.csv_path, diretorio_cache=self.pasta_cache.name)
        self.assertNotEqual(indice_sub["chave"], self.indice["chave"])
        self.assertEqual(indice_sub["embeddings"].shape[0], 6)
        self.assertEqual([it["ID"] for it in indice_sub["itens"]], [it["ID"] for it in subconjunto])

    def test_17_cache_em_disco_reaproveitado(self):
        """Recarregar o mesmo conjunto lê o .npy e devolve embeddings idênticos."""
        resetar_indice()
        recarregado = construir_indice(ISSUES_MOCK, self.csv_path, diretorio_cache=self.pasta_cache.name)
        np.testing.assert_allclose(recarregado["embeddings"], self.indice["embeddings"], atol=1e-6)
        self.assertTrue((Path(self.pasta_cache.name) / f"embeddings_{recarregado['chave']}.npy").exists())


# ==============================================================================
# Teste de integração com SIS-ASTROS real (opcional — pula se não encontrar o CSV)
# ==============================================================================

class TestEmbeddingsComCSVReal(unittest.TestCase):
    """
    Testes de integração com o CSV real do SIS-ASTROS.
    Usam o caminho definido em src/config.py (CAMINHO_SIS_ASTROS) ou SIS_ASTROS_PATH;
    são pulados enquanto o arquivo não existir.
    """

    def setUp(self):
        from src.sis_astros import resolver_caminho
        self.caminho = resolver_caminho()
        if not self.caminho.exists():
            self.skipTest(f"CSV do SIS-ASTROS ainda não colocado em {self.caminho} — pulando teste de integração.")

    def test_carregar_e_indexar_csv_real(self):
        """Carrega o CSV real, aplica o split temporal, constrói o índice e testa uma busca."""
        from src.sis_astros import carregar_sis_astros, dividir_temporal
        resetar_indice()
        itens = carregar_sis_astros(str(self.caminho))
        self.assertGreater(len(itens), 0, "O CSV real deve ter ao menos 1 issue válida após filtragem")

        treino, teste = dividir_temporal(itens, 0.7, caminho_csv=str(self.caminho), caminho_manifesto=None)
        indice = construir_indice(treino, str(self.caminho))
        self.assertEqual(indice["embeddings"].shape[0], len(treino))
        print(f"\n  [OK] Índice: {indice['embeddings'].shape} embeddings para {len(treino)} issues do histórico")

        alvo = teste[0]
        similares = buscar_issues_similares(
            alvo["Title"], alvo["Description"], alvo["ID"], alvo["Creation_Date"], indice, k=3
        )
        ids_teste = {it["ID"] for it in teste}
        self.assertTrue(all(s["ID"] not in ids_teste for s in similares))
        print(f"  Alvo: [{alvo['ID']}] {alvo['Title'][:60]} ({int(alvo['Story_Point'])} SP)")
        for s in similares:
            print(f"    -> [{s['ID']}] {s['Title'][:60]} ({int(s['Story_Point'])} SP, sim={s['similaridade']:.3f})")
        resetar_indice()


# ==============================================================================
# Runner manual com output detalhado
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MASPE — Teste de Recuperação Semântica (RAG) de Issues Históricas")
    print("Modelo de embeddings: paraphrase-multilingual-MiniLM-L12-v2")
    print("=" * 70)
    print()

    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None  # mantém ordem de definição
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingsRAG))
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingsComCSVReal))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

