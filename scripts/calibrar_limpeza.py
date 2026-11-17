"""
Mede, no CSV, as distribuições que dão base empírica às constantes de src/limpeza_texto.py. Só números agregados
(nenhum texto, título ou ID). Não usa as respostas de nenhum modelo: as constantes são derivadas dos DADOS, antes
de qualquer avaliação de erro, para não haver ajuste ao resultado.

Uso (na raiz do projeto):
    python -m scripts.calibrar_limpeza issues_astros_ufsm_limpo.csv --col-desc Description
"""
import argparse
import re

import numpy as np
import pandas as pd

import src.limpeza_texto as lt
from src.limpeza_texto import _RE_BLOCO, _RE_TRACE, _RE_URL, contar_tokens


def q(valores, ps=(1, 5, 25, 50, 75, 90, 95, 99), fmt="{:.0f}"):
    if len(valores) == 0:
        return "(sem dados)"
    v = np.percentile(valores, ps)
    return " ".join(f"p{p}={fmt.format(x)}" for p, x in zip(ps, v)) + f" máx={fmt.format(max(valores))} (n={len(valores)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--col-desc", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.csv, sep=None, engine="python", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    textos = [t for t in df[a.col_desc].tolist() if t.strip()]
    usa_tok = bool(contar_tokens("teste") is not None and lt._tokenizador)
    print(f"Issues com descrição: {len(textos)} | tokens: {'tokenizador Qwen3' if usa_tok else 'ESTIMATIVA (sem tokenizador: a razão abaixo é circular)'}")

    # 1) caracteres por token (CARACTERES_POR_TOKEN, usado só sem tokenizador)
    razao = [len(t) / max(contar_tokens(t), 1) for t in textos]
    print("\n[1] caracteres por token, por issue (razão menor = mais tokens por caractere = risco maior de estourar o orçamento)")
    print("   ", q(razao, ps=(1, 5, 25, 50, 75, 95), fmt="{:.2f}"))

    # 2) comprimento de palavras alfabéticas (TOKEN_LONGO)
    sem_blocos_urls = [_RE_URL.sub(" ", _RE_BLOCO.sub(" ", t)) for t in textos]
    alfa = [len(w) for t in sem_blocos_urls for w in re.findall(r"[^\W\d_]+", t)]
    tudo = [len(w) for t in sem_blocos_urls for w in t.split()]
    print("\n[2] comprimento de tokens (fora de blocos de código e URLs)")
    print("    só letras (palavras e nomes CamelCase):", q(alfa, ps=(50, 90, 99, 99.9, 99.99), fmt="{:.0f}"))
    print("    qualquer sequência sem espaço       :", q(tudo, ps=(50, 90, 99, 99.9, 99.99), fmt="{:.0f}"))
    for limite in (20, 25, 30, 40, 60):
        print(f"    sequências >= {limite:>2} caracteres: {sum(1 for x in tudo if x >= limite)} ocorrências "
              f"em {sum(1 for t in sem_blocos_urls if any(len(w) >= limite for w in t.split()))} issues")

    # 3) URLs (URL_MAX)
    urls = [len(u) for t in textos for u in _RE_URL.findall(t)]
    print("\n[3] comprimento das URLs http(s)")
    print("   ", q(urls, ps=(10, 25, 50, 75, 90, 99)))
    for limite in (40, 60, 100):
        print(f"    URLs <= {limite:>3} caracteres: {np.mean([u <= limite for u in urls]) * 100 if urls else 0:.0f}%")
    rel = sum(len(re.findall(r"\]\(/uploads/", t)) for t in textos)
    print(f"    anexos relativos do GitLab (](/uploads/...): {rel} ocorrências")

    # 4) blocos de código/log (LINHAS_BLOCO)
    blocos = [next(g for g in m.groups() if g is not None) for t in textos for m in _RE_BLOCO.finditer(t)]
    linhas = [len(b.strip("\n").splitlines()) for b in blocos]
    tok_linha = [contar_tokens(b) / max(len(b.strip("\n").splitlines()), 1) for b in blocos if b.strip()]
    print("\n[4] blocos de código/log")
    print("    linhas por bloco :", q(linhas, ps=(25, 50, 75, 90, 99)))
    print("    tokens por linha :", q(tok_linha, ps=(25, 50, 75, 90), fmt="{:.1f}"))
    for n in (3, 6, 12):
        print(f"    blocos com <= {n:>2} linhas (ficam intactos): {np.mean([x <= n for x in linhas]) * 100 if linhas else 0:.0f}%")

    # 5) stack traces e linhas gigantes (LINHAS_TRACE, LINHA_MAX, LINHA_MAX_ESPACOS)
    corridas, gigantes = [], 0
    for t in textos:
        c = 0
        for l in t.splitlines():
            if _RE_TRACE.match(l):
                c += 1
            else:
                if c:
                    corridas.append(c)
                c = 0
            if len(l) > lt.LINHA_MAX and l.count(" ") / len(l) < lt.LINHA_MAX_ESPACOS:
                gigantes += 1
        if c:
            corridas.append(c)
    print("\n[5] regras possivelmente inativas")
    print(f"    sequências de linhas de stack trace: {len(corridas)} (>{lt.LINHAS_TRACE} linhas: {sum(1 for c in corridas if c > lt.LINHAS_TRACE)})")
    print(f"    linhas gigantes sem cara de prosa: {gigantes}")
    esp = [t.count(" ") / len(t) for t in sem_blocos_urls if len(t) > 100]
    print("    fração de espaços por issue (prosa):", q(esp, ps=(5, 25, 50, 75, 95), fmt="{:.3f}"))


if __name__ == "__main__":
    main()