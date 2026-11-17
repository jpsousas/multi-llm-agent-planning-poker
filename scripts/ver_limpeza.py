"""
Mostra, para algumas issues, o texto ORIGINAL ao lado do texto APÓS A LIMPEZA (src/limpeza_texto.py), num HTML local.
Leve: só lê o CSV; não usa embeddings, LLM nem o split.

!! O ARQUIVO GERADO CONTÉM O TEXTO DAS ISSUES (confidencial). Mantenha-o só na sua máquina: não cole no chat,
   não versione (.gitignore) e anonimize qualquer trecho que for para a tese.

Uso (na raiz do projeto):
    python -m scripts.ver_limpeza issues_astros_ufsm_limpo.csv                      # 6 issues aleatórias
    python -m scripts.ver_limpeza issues_astros_ufsm_limpo.csv -n 10 --semente 7
    python -m scripts.ver_limpeza issues_astros_ufsm_limpo.csv --selecao mais-afetadas -n 8   # onde a limpeza mais corta
    python -m scripts.ver_limpeza issues_astros_ufsm_limpo.csv --ids 1234 5678
    python -m scripts.ver_limpeza issues_astros_ufsm_limpo.csv --ids 1234 --terminal   # imprime no terminal, sem HTML
Colunas: --col-id (padrão IID), --col-titulo (Title), --col-desc (Description).
"""
import argparse
import html
import random
from pathlib import Path

import pandas as pd

from src.limpeza_texto import contar_tokens, limpar_descricao

CSS = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px;max-width:1500px;color:#1b1b1b;background:#fff}
.aviso{background:#fde8e8;border:2px solid #c0392b;padding:10px 14px;font-weight:600;margin-bottom:18px}
h2{font-size:1.05rem;margin-top:30px;border-bottom:2px solid #ddd;padding-bottom:4px}
.grade{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.col{border:1px solid #ccc;border-radius:6px;padding:8px 10px;background:#fafafa;min-width:0}
.meta{font-size:.82rem;color:#444;margin-bottom:4px}
pre{white-space:pre-wrap;word-break:break-word;font-size:.82rem;margin:4px 0;background:#fff;border:1px solid #e3e3e3;padding:6px;max-height:520px;overflow:auto}
@media(prefers-color-scheme:dark){body{background:#161616;color:#e8e8e8}.col{background:#1f1f1f;border-color:#444}pre{background:#111;border-color:#333}.meta{color:#aaa}.aviso{background:#3a1515}}
"""
NOMES = {"blocos_omitidos": "blocos de código/log encurtados", "traces_omitidos": "stack traces encurtados",
         "urls": "URLs longas", "imagens": "imagens markdown", "tokens_longos": "tokens longos", "linhas_cortadas": "linhas gigantes cortadas"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--col-id", default="IID")
    ap.add_argument("--col-titulo", default="Title")
    ap.add_argument("--col-desc", default="Description")
    ap.add_argument("-n", type=int, default=6)
    ap.add_argument("--selecao", choices=["aleatoria", "mais-afetadas"], default="aleatoria")
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--ids", nargs="+", default=None)
    ap.add_argument("--saida", default="ver_limpeza.html")
    ap.add_argument("--terminal", action="store_true", help="imprime original e limpo no terminal em vez de gerar HTML")
    a = ap.parse_args()

    df = pd.read_csv(a.csv, sep=None, engine="python", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for c in (a.col_id, a.col_titulo, a.col_desc):
        if c not in df.columns:
            raise SystemExit(f"Coluna '{c}' não existe no CSV. Colunas: {list(df.columns)}")

    if a.ids:
        idx = [i for i in range(len(df)) if df[a.col_id].iloc[i] in set(a.ids)]
        if not idx:
            raise SystemExit("Nenhum dos IDs informados foi encontrado.")
    elif a.selecao == "aleatoria":
        idx = sorted(random.Random(a.semente).sample(range(len(df)), min(a.n, len(df))))
    else:
        reducao = []
        for i, t in enumerate(df[a.col_desc]):
            reducao.append((contar_tokens(t) - contar_tokens(limpar_descricao(t)), i))
        idx = [i for _, i in sorted(reducao, reverse=True)[: a.n]]

    if a.terminal:
        for n, i in enumerate(idx, 1):
            original = df[a.col_desc].iloc[i]
            meta = {}
            limpo = limpar_descricao(original, meta=meta)
            ops = "; ".join(f"{NOMES[k]}: {meta[k]}" for k in NOMES if meta.get(k)) or "nenhuma operação aplicada"
            print(f"\n{'=' * 78}\n#{n} ID {df[a.col_id].iloc[i]} | {df[a.col_titulo].iloc[i]}")
            print(f"{'=' * 78}\n--- ORIGINAL ({len(original)} caracteres, {contar_tokens(original)} tokens) ---\n{original}")
            print(f"\n--- APÓS LIMPEZA ({len(limpo)} caracteres, {contar_tokens(limpo)} tokens) | {ops} ---\n{limpo}")
        return

    partes = [f"<!doctype html><html lang='pt-br'><meta charset='utf-8'><title>Ver limpeza</title><style>{CSS}</style><body>",
              "<div class='aviso'>CONFIDENCIAL: este arquivo contém o texto das issues. Não compartilhe, não versione, não cole no chat.</div>"]
    for n, i in enumerate(idx, 1):
        original = df[a.col_desc].iloc[i]
        meta = {}
        limpo = limpar_descricao(original, meta=meta)
        t0, t1 = contar_tokens(original), contar_tokens(limpo)
        ops = "; ".join(f"{NOMES[k]}: {meta[k]}" for k in NOMES if meta.get(k)) or "nenhuma operação aplicada"
        partes.append(f"<h2>#{n} — ID {html.escape(df[a.col_id].iloc[i])} — {html.escape(df[a.col_titulo].iloc[i])}</h2>")
        partes.append("<div class='grade'>"
                      f"<div class='col'><div class='meta'>ORIGINAL — {len(original)} caracteres, {t0} tokens</div><pre>{html.escape(original)}</pre></div>"
                      f"<div class='col'><div class='meta'>APÓS LIMPEZA — {len(limpo)} caracteres, {t1} tokens "
                      f"({(1 - t1 / max(t0, 1)) * 100:.0f}% menos) | {html.escape(ops)}</div><pre>{html.escape(limpo)}</pre></div></div>")
    partes.append("</body></html>")
    saida = Path(a.saida)
    saida.write_text("\n".join(partes), encoding="utf-8")
    print(f"Salvo em: {saida.resolve()} ({len(idx)} issues). Contém texto confidencial: mantenha local.")


if __name__ == "__main__":
    main()