"""
scripts/verificar_contexto.py
=============================
Confere, SEM chamar o LLM, se os prompts cabem no contexto do servidor. Só imprime números agregados.

Para cada issue-alvo da fase escolhida, monta o prompt exatamente como o pipeline monta (prompt de sistema +
exemplos + alvo, na R1) e conta os tokens. O pior caso soma ainda o debate (até 5 justificativas de colegas na
R3 do sequencial) e a resposta do modelo.

Uso (raiz do projeto):
    python -m scripts.verificar_contexto                    # fase dev, few-shot, contexto 16384
    python -m scripts.verificar_contexto --rag --contexto 16384
    python -m scripts.verificar_contexto --fase teste --rag # só conta tokens; não avalia erro
Respeita as mesmas variáveis de ambiente do pipeline (LIMPAR_TEXTO, ORCAMENTO_TOKENS_EXEMPLO, ...).
"""
import argparse

import numpy as np

from src.config import ESCALAS_FIBONACCI, K_RAG, LIMIAR_DUPLICATA
from src.limpeza_texto import contar_tokens
from src.prompts import montar_prompt_item, montar_system_prompt
from src.sis_astros import LIMPAR_TEXTO, montar_dataset_sis_astros


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fase", choices=["dev", "teste"], default="dev")
    ap.add_argument("--rag", action="store_true")
    ap.add_argument("--contexto", type=int, default=16384, help="contexto por slot do servidor (padrão 16384)")
    ap.add_argument("--saida", type=int, default=1500, help="tokens reservados para a resposta do modelo")
    ap.add_argument("--justificativa", type=int, default=600, help="tokens por justificativa no debate (pior caso)")
    ap.add_argument("--arquivo", default=None)
    a = ap.parse_args()

    ds = montar_dataset_sis_astros(a.arquivo, usar_rag=a.rag, k_rag=K_RAG, fase=a.fase,
                                   limiar_duplicata=LIMIAR_DUPLICATA if a.rag else None)
    escala = ESCALAS_FIBONACCI["SIS-ASTROS"]
    sistema = max(contar_tokens(montar_system_prompt(p, escala)) for p in ("neutro", "backend", "frontend", "qa"))

    r1, pior = [], []
    for item in ds:
        humano = contar_tokens(montar_prompt_item(item["titulo"], item["descricao"], historicos=item["historicos"]))
        r1.append(sistema + humano)
        pior.append(sistema + humano + 5 * a.justificativa + a.saida)
    r1, pior = np.array(r1), np.array(pior)

    metodo = "tokenizador do Qwen3" if contar_tokens("x" * 300) != 100 else "estimativa conservadora (caracteres/3)"
    pm = [d["prep_meta"] for d in ds]
    print(f"\n=== Verificação de contexto: fase={a.fase}, {'RAG' if a.rag else 'few-shot'}, limpeza={'sim' if LIMPAR_TEXTO else 'não'}, "
          f"n={len(ds)}, contexto={a.contexto} ===")
    print(f"contagem de tokens: {metodo}; prompt de sistema = {sistema} tokens")
    print(f"prompt da R1 (sistema + exemplos + alvo): p50={np.percentile(r1, 50):.0f}  p90={np.percentile(r1, 90):.0f}  "
          f"p99={np.percentile(r1, 99):.0f}  máx={r1.max():.0f}")
    print(f"pior caso (R1 + 5 justificativas de {a.justificativa} + saída de {a.saida}): máx={pior.max():.0f}")
    estoura_r1, estoura_pior = int((r1 > a.contexto).sum()), int((pior > a.contexto).sum())
    print(f"issues cujo prompt da R1 NÃO cabe: {estoura_r1}/{len(ds)}")
    print(f"issues cujo pior caso NÃO cabe:    {estoura_pior}/{len(ds)}")
    print(f"alvos truncados: {sum(1 for m in pm if m.get('truncado'))} | "
          f"exemplos truncados (total): {sum(m.get('exemplos_truncados', 0) for m in pm)} | "
          f"orçamento de exemplo mínimo: {min(m['orcamento_exemplo'] for m in pm)} tokens")
    print("RESULTADO:", "OK — tudo cabe." if estoura_pior == 0 else
          ("ATENÇÃO — o pior caso estoura em algumas issues; o orçamento de exemplos já encolhe, mas confira o contexto do servidor."
           if estoura_r1 == 0 else "FALHA — o prompt da R1 já não cabe: aumente o contexto do servidor."))


if __name__ == "__main__":
    main()