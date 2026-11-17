"""
Diagnóstico das descrições longas SEM expor conteúdo: imprime só números agregados
(nenhum título, ID, trecho de texto ou exemplo de token). Seguro para dados confidenciais.

Uso:
    python diagnosticar_texto.py issues_astros_ufsm_limpo.csv --col-desc Description
    python diagnosticar_texto.py issues.csv --col-desc Description --limite 4000

Opcional (só para uso local, nunca para compartilhar a saída):
    --mostrar-texto --col-titulo Title --col-id ID --top 3
"""
import argparse
import re

import pandas as pd

CATEGORIAS = {
    "URLs": lambda t: sum(len(m) for m in re.findall(r"https?://\S+", t)),
    "imagens/dados em base64 (data:...;base64,...)":
        lambda t: sum(len(m) for m in re.findall(r"data:[\w/+.-]+;base64,[A-Za-z0-9+/=]+", t)),
    "outros tokens longos (>=40 chars sem espaço: hashes, caminhos)":
        lambda t: sum(len(w) for w in re.sub(r"https?://\S+|data:[\w/+.-]+;base64,[A-Za-z0-9+/=]+", " ", t).split() if len(w) >= 40),
    "tags HTML/XML": lambda t: sum(len(m) for m in re.findall(r"<[^>]{1,200}>", t)),
    "blocos de código/log ({code}, {noformat}, ```)":
        lambda t: sum(len(m) for m in re.findall(r"(?s)\{code[^}]*\}.*?\{code\}|\{noformat\}.*?\{noformat\}|```.*?```", t)),
    "linhas de stack trace/erro":
        lambda t: sum(len(l) for l in t.splitlines() if re.match(r"\s*(at |Caused by|Traceback|File \")", l) or re.search(r"Exception|Error:", l)),
    "linhas de tabela (3+ barras '|')": lambda t: sum(len(l) for l in t.splitlines() if l.count("|") >= 3),
    "linhas repetidas (cópias de linhas já vistas)":
        lambda t: sum(len(l) for l, n in _contar_linhas(t) if n > 1),
}


def _contar_linhas(t):
    vistas, extras = {}, []
    for l in t.splitlines():
        l = l.strip()
        if len(l) < 15:
            continue
        vistas[l] = vistas.get(l, 0) + 1
        if vistas[l] > 1:
            extras.append((l, vistas[l]))
    return extras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--col-desc", required=True)
    ap.add_argument("--limite", type=int, default=4000, help="analisa a composição das issues acima deste nº de caracteres")
    ap.add_argument("--mostrar-texto", action="store_true", help="EXIBE conteúdo (título, início, tokens longos): só local")
    ap.add_argument("--col-titulo")
    ap.add_argument("--col-id")
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()

    df = pd.read_csv(a.csv, sep=None, engine="python", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    desc = df[a.col_desc]
    chars = desc.str.len()
    words = desc.str.split().str.len().clip(lower=1)

    print(f"Issues: {len(df)} | descrições vazias: {(chars == 0).sum()}")
    q = chars.quantile([.5, .75, .9, .95, .99]).round().astype(int)
    print("Caracteres por descrição:", dict(zip(["p50", "p75", "p90", "p95", "p99"], q.tolist())), "| máx:", chars.max())
    print(f"Caracteres por palavra (mediana): {(chars / words).median():.1f}   [prosa em português: ~6]")
    for lim in (2000, 4000, 10000, 20000):
        print(f"  > {lim:>6} caracteres: {(chars > lim).sum():>4} issues ({(chars > lim).mean() * 100:.1f}%)")

    longas = desc[chars > a.limite]
    print(f"\n=== Composição das {len(longas)} issues com mais de {a.limite} caracteres (% dos caracteres de cada issue) ===")
    if len(longas) == 0:
        print("Nenhuma issue acima do limite.")
    else:
        print(f"{'categoria':<66}{'mediana':>8}{'p90':>8}{'issues>=50%':>13}")
        for nome, f in CATEGORIAS.items():
            share = pd.Series([f(t) / max(len(t), 1) * 100 for t in longas])
            print(f"{nome:<66}{share.median():>7.1f}%{share.quantile(.9):>7.1f}%{(share >= 50).sum():>13}")
        print("(As categorias podem se sobrepor; os percentuais não precisam somar 100%.)")

    if a.mostrar_texto:
        print("\n=== CONTEÚDO (não compartilhe esta parte) ===")
        for i in chars.sort_values(ascending=False).head(a.top).index:
            ident = df[a.col_id][i] if a.col_id else i
            titulo = df[a.col_titulo][i][:80] if a.col_titulo else ""
            print(f"\n[{ident}] {chars[i]} caracteres | {titulo}")
            print("  início:", re.sub(r"\s+", " ", desc[i][:250]))


if __name__ == "__main__":
    main()