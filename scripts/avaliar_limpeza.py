"""
Avalia o efeito da limpeza + orçamento de tokens num CSV, imprimindo SÓ números agregados
(nenhum trecho de texto, título ou ID). Seguro para dados confidenciais.

Uso (na raiz do projeto):
    python -m scripts.avaliar_limpeza issues_astros_ufsm_limpo.csv --col-desc Description
    python -m scripts.avaliar_limpeza issues_astros_ufsm_limpo.csv --col-desc Description --orcamento-exemplo 500 --orcamento-alvo 2000 -k 3
"""
import argparse

import pandas as pd

from src.limpeza_texto import contar_tokens, limpar_descricao, preparar_descricao, truncar_para_orcamento
import src.limpeza_texto as lt


def percentis(s: pd.Series) -> str:
    q = s.quantile([.5, .75, .9, .95, .99])
    return " ".join(f"p{int(p * 100)}={int(round(v))}" for p, v in q.items()) + f" máx={int(s.max())}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--col-desc", required=True)
    ap.add_argument("--orcamento-exemplo", type=int, default=500, help="tokens máximos por exemplo recuperado")
    ap.add_argument("--orcamento-alvo", type=int, default=2000, help="tokens máximos da issue alvo")
    ap.add_argument("-k", type=int, default=3, help="nº de exemplos recuperados")
    ap.add_argument("--sem-limpeza", action="store_true",
                    help="mede o truncamento sobre o texto ORIGINAL (LIMPAR_TEXTO=0), sem a limpeza determinística")
    a = ap.parse_args()

    df = pd.read_csv(a.csv, sep=None, engine="python", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    textos = df[a.col_desc].tolist()
    usa_tok = bool(lt.contar_tokens("teste") is not None and lt._tokenizador)
    print(f"Issues: {len(textos)} | contagem de tokens: "
          f"{'tokenizador Qwen3' if usa_tok else f'estimativa (caracteres/{lt.CARACTERES_POR_TOKEN:g}, conservadora)'}")

    antes = pd.Series([contar_tokens(t) for t in textos])
    metas, apos = [], []
    for t in textos:
        m = {}
        apos.append(contar_tokens(limpar_descricao(t, meta=m)))
        metas.append(m)
    apos = pd.Series(apos)
    mm = pd.DataFrame(metas)

    print("\nTokens por descrição")
    print("  original ......... ", percentis(antes))
    print("  após limpeza ..... ", percentis(apos))
    print(f"  redução total: {(1 - apos.sum() / max(antes.sum(), 1)) * 100:.1f}% dos tokens no total")

    print("\nOperações da limpeza (nº de issues afetadas / total de ocorrências)")
    for col, nome in [("blocos_omitidos", "blocos de código/log encurtados"),
                      ("traces_omitidos", "stack traces encurtados"),
                      ("urls", "URLs longas -> [link]"),
                      ("tokens_longos", "tokens longos encurtados"),
                      ("linhas_cortadas", "linhas gigantes cortadas")]:
        print(f"  {nome:<36}{int((mm[col] > 0).sum()):>5} issues / {int(mm[col].sum()):>6}")

    for nome, orc in (("exemplo", a.orcamento_exemplo), ("alvo", a.orcamento_alvo)):
        if a.sem_limpeza:
            prep = []
            for t in textos:
                pronto, trunc = truncar_para_orcamento((t or "").replace("\r\n", "\n").strip(), orc)
                prep.append({"truncado": int(trunc), "tokens_final": contar_tokens(pronto)})
            rotulo = "truncadas (texto original, sem limpeza)"
        else:
            prep = [preparar_descricao(t, orc)[1] for t in textos]
            rotulo = "truncadas após a limpeza"
        pm = pd.DataFrame(prep)
        print(f"\nOrçamento de {orc} tokens ({nome})")
        print(f"  {rotulo}: {int(pm['truncado'].sum())} issues ({pm['truncado'].mean() * 100:.1f}%)")
        print("  tokens finais:", percentis(pm["tokens_final"]))

    pior = a.orcamento_alvo + a.k * a.orcamento_exemplo
    print(f"\nPrompt de descrições, pior caso com k={a.k}: {pior} tokens "
          f"(alvo {a.orcamento_alvo} + {a.k} x {a.orcamento_exemplo}), sem contar persona, instruções e debate.")


if __name__ == "__main__":
    main()