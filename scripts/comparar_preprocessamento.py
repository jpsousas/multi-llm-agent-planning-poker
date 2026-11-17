"""
Compara, no bloco de DESENVOLVIMENTO, três pré-processamentos do texto usado para gerar os embeddings
da recuperação (kNN, SEM LLM), com testes pareados. Só números agregados são impressos.

  nenhum : título + 400 primeiros caracteres da descrição, texto original (o que o projeto usa hoje)
  regex  : remove ruído de marcação (imagens, links, URLs, menções, tags HTML/Jira, símbolos markdown)
  limpeza: limpeza determinística do projeto (src/limpeza_texto.py): encurta blocos de código/log, stack traces,
           URLs e tokens longos, remove HTML; mantém o texto em linguagem natural
  rossi  : regex + minúsculas + remoção de stopwords (PT) + stemming RSLP (pipeline descrito pelo Rossi)

Atenção: isto muda SÓ o texto que vira embedding (busca). O texto que o LLM lê no prompt não é alterado.
O cache de embeddings do pipeline NÃO é tocado (cada variante usa uma pasta temporária).

Requer, para a variante 'rossi':  pip install nltk   e, uma vez:
    python -c "import nltk; nltk.download('stopwords'); nltk.download('rslp')"

Uso (na raiz do projeto):
    python -m scripts.comparar_preprocessamento --arquivo issues_astros_ufsm_limpo.csv
"""
import argparse
import html
import re
import statistics
import tempfile

import numpy as np

import src.retrieval as R
from src.limpeza_texto import limpar_descricao
from comparar_knn import erros, holm, linha, metricas, para_degrau, teste_pareado
from src.sis_astros import carregar_sis_astros, dividir_temporal, resolver_caminho

MODELO = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

_RE_IMAGEM_MD = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_RE_LINK_MD = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_RE_IMAGEM_JIRA = re.compile(r"![^!\s]+!")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_MENCAO = re.compile(r"(?<!\w)@[\w.-]+")
_RE_TAG_JIRA = re.compile(r"\{/?(panel|code|noformat|quote|color|info|warning|note)[^}]*\}", re.I)
_RE_TAG_HTML = re.compile(r"</?[a-zA-Z][^>]*>")
_RE_SIMBOLOS = re.compile(r"[`#*>|_~=]+")


def regex_ruido(t: str) -> str:
    t = html.unescape(t or "")
    t = _RE_IMAGEM_MD.sub(" ", t)
    t = _RE_LINK_MD.sub(r"\1", t)
    t = _RE_IMAGEM_JIRA.sub(" ", t)
    t = _RE_TAG_JIRA.sub(" ", t)
    t = _RE_TAG_HTML.sub(" ", t)
    t = _RE_URL.sub(" ", t)
    t = _RE_MENCAO.sub(" ", t)
    t = _RE_SIMBOLOS.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def construir_rossi(stopwords=None, stemmer=None):
    """Devolve f(texto) = regex + minúsculas + sem stopwords + stemming RSLP."""
    if stopwords is None or stemmer is None:
        try:
            from nltk.corpus import stopwords as sw
            from nltk.stem import RSLPStemmer
            stopwords, stemmer = set(sw.words("portuguese")), RSLPStemmer()
        except Exception as e:
            raise SystemExit(f"Variante 'rossi' precisa de nltk com os dados 'stopwords' e 'rslp' ({e}). "
                             "Rode: python -c \"import nltk; nltk.download('stopwords'); nltk.download('rslp')\"")
    def f(t: str) -> str:
        palavras = re.findall(r"\w+", regex_ruido(t).lower())
        return " ".join(stemmer.stem(p) for p in palavras if p not in stopwords)
    return f


def fabricar_texto_embedding(transforma):
    def _texto(titulo, descricao):
        desc = transforma(descricao or "")[:400]
        tit = transforma(titulo or "")
        return f"{tit} - {desc}" if desc else tit
    return _texto


def main(stopwords=None, stemmer=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arquivo", default=None)
    ap.add_argument("--split", type=float, default=0.7)
    ap.add_argument("--split-dev", type=float, default=0.7)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--limiar", type=float, default=None)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--variantes", nargs="+", default=["nenhum", "regex", "limpeza", "rossi"],
                    choices=["nenhum", "regex", "limpeza", "rossi"],
                    help="'nenhum' é sempre incluído (referência). Omita 'rossi' se não tiver o nltk.")
    a = ap.parse_args()

    caminho = resolver_caminho(a.arquivo)
    itens = carregar_sis_astros(str(caminho))
    historico, _ = dividir_temporal(itens, a.split, caminho_csv=str(caminho), caminho_manifesto=None)
    corte = int(len(historico) * a.split_dev)
    base, alvos = historico[:corte], historico[corte:]
    real = [int(t["Story_Point"]) for t in alvos]
    print(f"\n=== DEV: base={len(base)} | alvos={len(alvos)} | k={a.k} | modelo={MODELO.split('/')[-1]} | bloco de teste NÃO usado ===")

    todas = {"nenhum": lambda t: t, "regex": regex_ruido, "limpeza": lambda t: limpar_descricao(t)}
    if "rossi" in a.variantes:
        todas["rossi"] = construir_rossi(stopwords, stemmer)
    variantes = {n: todas[n] for n in ["nenhum"] + [v for v in a.variantes if v != "nenhum"]}
    mediana = para_degrau(statistics.median(int(b["Story_Point"]) for b in base))
    print(linha(f"baseline: mediana da base ({mediana})", metricas([mediana] * len(alvos), real)))

    original = R._texto_para_embedding
    R.MODELO_EMBEDDINGS = MODELO
    R._modelo = None
    err = {}
    try:
        for nome, f in variantes.items():
            R._texto_para_embedding = fabricar_texto_embedding(f)
            R.resetar_indice()
            with tempfile.TemporaryDirectory() as pasta:          # não toca no cache do pipeline
                indice = R.construir_indice(base, str(caminho), forcar_rebuild=True, diretorio_cache=pasta)
            prev = []
            for t in alvos:
                sims, _meta = R.buscar_com_diagnostico(t["Title"], t.get("Description", ""), t["ID"],
                                                       t.get("Creation_Date"), indice, k=a.k, limiar_duplicata=a.limiar)
                prev.append(para_degrau(statistics.median(int(s["Story_Point"]) for s in sims)) if sims else mediana)
            err[nome] = erros(prev, real)
            print(linha(f"kNN k={a.k} | pré-processamento: {nome}", metricas(prev, real)))
    finally:
        R._texto_para_embedding = original

    pares = [(v, "nenhum") for v in variantes if v != "nenhum"]   # uma família: cada variante contra o texto original
    res = {m: [teste_pareado(err[x][m] - err[y][m], n_boot=a.bootstrap) for x, y in pares] for m in (0, 1)}
    for m in (0, 1):
        for r, ph in zip(res[m], holm([r["p"] for r in res[m]])):
            r["p_holm"] = ph
    print("\n=== TESTES PAREADOS (d = erro(A) - erro(B); d < 0 favorece A) | Holm sobre todas as comparações ===")
    for i, (x, y) in enumerate(pares):
        print(f"  {x} vs {y}")
        for m, rot in ((0, "horas  "), (1, "degraus")):
            r = res[m][i]
            print(f"      {rot}: d={r['media']:+.3f} [IC95% {r['lo']:+.3f}, {r['hi']:+.3f}]  p={r['p']:.3f}  "
                  f"p_Holm={r['p_holm']:.3f}  A melhor={r['melhor']:.0f}% pior={r['pior']:.0f}%  MDE~{r['mde']:.3f}")
    print("\nLeitura: diferença 'real' = IC95% sem o zero E p_Holm < 0,05. Sem diferença real, fica o mais simples ('nenhum').")


if __name__ == "__main__":
    main()