"""
Limpa a coluna "Time Estimate" de um CSV (ex.: SIS-ASTROS), nesta ordem:

  1. remove linhas com Time Estimate vazio;
  2. remove linhas cuja célula contém o caractere "m" (minutos, ex.: "30m", "1h 30m");
  3. remove o caractere "h" da célula (ex.: "5h" -> "5");
  4. remove linhas cujo valor restante não está na sequência de Fibonacci
     (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, ...).

Uso:
    python limpar_estimativas.py dados.csv
    python limpar_estimativas.py dados.csv --col "Time Estimate" --out dados_limpo.csv

Saídas (o arquivo original nunca é alterado):
    <nome>_limpo.csv      linhas mantidas, com a coluna já normalizada (ex.: "5")
    <nome>_removidas.csv  linhas removidas, com a coluna "motivo" (auditoria)
    Relatório no terminal com a contagem de cada exclusão (fluxo de exclusão do TCC).

Observações:
  - "m" e "h" são tratados sem diferenciar maiúsculas de minúsculas.
  - Valores como "5.0" ou "5,0" são aceitos como 5. Decimais reais ("2.5") são removidos.
  - O 0 não está na sequência pedida, então "0" ou "0h" são removidos.
  - Todas as colunas são lidas como texto, para não alterar IDs nem datas.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def gerar_fibonacci(limite: int = 10**9) -> set:
    """1, 2, 3, 5, 8, 13, ... (sem repetir o 1 e sem o 0)."""
    fib = [1, 2]
    while fib[-1] + fib[-2] <= limite:
        fib.append(fib[-1] + fib[-2])
    return set(fib)


FIB = gerar_fibonacci()


def ler_csv(caminho: Path) -> tuple[pd.DataFrame, str, str]:
    """Lê o CSV como texto, detectando encoding e separador."""
    ultimo_erro = None
    for enc in ENCODINGS:
        try:
            with open(caminho, "r", encoding=enc, newline="") as f:
                amostra = f.read(8192)
            try:
                sep = csv.Sniffer().sniff(amostra, delimiters=",;\t|").delimiter
            except csv.Error:
                sep = ","
            df = pd.read_csv(
                caminho, sep=sep, dtype=str, keep_default_na=False, encoding=enc
            )
            return df, enc, sep
        except UnicodeDecodeError as e:
            ultimo_erro = e
    raise SystemExit(f"Não consegui decodificar o arquivo: {ultimo_erro}")


def achar_coluna(df: pd.DataFrame, nome: str) -> str:
    """Encontra a coluna ignorando maiúsculas/minúsculas e espaços nas pontas."""
    alvo = nome.strip().lower()
    for c in df.columns:
        if c.strip().lower() == alvo:
            return c
    raise SystemExit(
        f'Coluna "{nome}" não encontrada. Colunas disponíveis: {list(df.columns)}'
    )


def para_inteiro(txt: str):
    """'5' / '5.0' / '5,0' -> 5. Qualquer outra coisa -> None."""
    m = re.fullmatch(r"(\d+)(?:[.,]0+)?", txt)
    return int(m.group(1)) if m else None


def limpar(df: pd.DataFrame, col: str):
    bruto = df[col].astype(str).str.strip()

    # 1. vazio
    mask_vazio = bruto.eq("")

    # 2. contém "m"
    mask_m = ~mask_vazio & bruto.str.contains("m", case=False, regex=False)

    # 3. remove "h" (nas linhas que continuam)
    restante = ~(mask_vazio | mask_m)
    sem_h = bruto.where(~restante, bruto.str.replace(r"[hH]", "", regex=True).str.strip())

    # 4. precisa estar na sequência de Fibonacci
    valores = sem_h.map(para_inteiro)
    na_escala = valores.map(lambda v: v in FIB if v is not None else False)
    mask_fib = restante & ~na_escala

    motivo = pd.Series("", index=df.index)
    motivo[mask_vazio] = "Time Estimate vazio"
    motivo[mask_m] = 'contém "m"'
    motivo[mask_fib] = "fora da sequência de Fibonacci"

    mantidas = df[motivo.eq("")].copy()
    mantidas[col] = valores[motivo.eq("")].astype(int).astype(str)

    removidas = df[motivo.ne("")].copy()
    removidas["motivo"] = motivo[motivo.ne("")]
    # mantém na auditoria o valor original da célula
    return mantidas, removidas, mask_vazio.sum(), mask_m.sum(), mask_fib.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada", type=Path, help="CSV de entrada")
    ap.add_argument("--col", default="Time Estimate", help='Coluna (padrão: "Time Estimate")')
    ap.add_argument("--out", type=Path, default=None, help="CSV de saída (padrão: <entrada>_limpo.csv)")
    args = ap.parse_args()

    if not args.entrada.exists():
        raise SystemExit(f"Arquivo não encontrado: {args.entrada}")

    saida = args.out or args.entrada.with_name(args.entrada.stem + "_limpo.csv")
    saida_rem = saida.with_name(saida.stem.removesuffix("_limpo") + "_removidas.csv")

    df, enc, sep = ler_csv(args.entrada)
    col = achar_coluna(df, args.col)

    mantidas, removidas, n_vazio, n_m, n_fib = limpar(df, col)

    mantidas.to_csv(saida, index=False, sep=sep, encoding="utf-8-sig")
    removidas.to_csv(saida_rem, index=False, sep=sep, encoding="utf-8-sig")

    print(f"Arquivo lido: {args.entrada}  (encoding={enc}, separador={sep!r})")
    print(f'Coluna: "{col}"')
    print("-" * 56)
    print(f"Linhas iniciais .................... {len(df):>6}")
    print(f"  removidas: Time Estimate vazio ... {n_vazio:>6}")
    print(f'  removidas: contém "m" ............ {n_m:>6}')
    print(f"  removidas: fora de Fibonacci ..... {n_fib:>6}")
    print(f"Linhas finais ...................... {len(mantidas):>6}")
    print("-" * 56)
    print("Distribuição final de Time Estimate:")
    dist = mantidas[col].astype(int).value_counts().sort_index()
    for valor, n in dist.items():
        print(f"  {valor:>4}: {n}")
    print("-" * 56)
    print(f"Mantidas  -> {saida}")
    print(f"Removidas -> {saida_rem}")


if __name__ == "__main__":
    sys.exit(main())