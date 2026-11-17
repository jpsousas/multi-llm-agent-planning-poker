"""
src/sis_astros.py
=================
Loader do dataset confidencial SIS-ASTROS (CSV exportado do GitLab).

Onde colocar o arquivo: ver `CAMINHO_SIS_ASTROS` em src/config.py
(padrão: data/sis_astros/dataset_sis_astros.csv — instruções em data/sis_astros/LEIA-ME.md).

Colunas do CSV:
    IID, Title, State, Author, Assignee, Time Estimate, Time Spent,
    Created At, Closed At, Milestone, Labels, Description, URL

Convenção de domínio:
    1 Story Point (SP) = 1 hora de esforço.
    Escala pura de Fibonacci: [1, 2, 3, 5, 8, 13, 21, 34, 55].
    "Time Estimate" é a estimativa da equipe (GROUND TRUTH do experimento).
    "Time Spent"    é o esforço real registrado (só ilustra os exemplos históricos).

Split: temporal, 70% mais antigos (histórico / base do RAG) e 30% mais recentes (teste),
ordenados por "Created At". O manifesto do split é salvo em results/split_sis_astros.json.
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timezone

import pandas as pd

from src.limpeza_texto import preparar_descricao, truncar_para_orcamento, contar_tokens

from src.config import (
    CAMINHO_SIS_ASTROS,
    COLUNAS_SIS_ASTROS,
    COLUNAS_OBRIGATORIAS_SIS_ASTROS,
    RAIZ_PROJETO,
    LIMIAR_DUPLICATA,
    K_RAG,
)


# Orçamentos de tokens da DESCRIÇÃO (após a limpeza determinística). Valem para todos os agentes,
# rodadas e condições. Ajuste por .env só no desenvolvimento; congele antes do teste.
# Prioridade no prompt: instruções > issue alvo > exemplos. Só os EXEMPLOS são truncados.
#   ORCAMENTO_TOKENS_EXEMPLO : teto por exemplo (o orçamento real pode ser menor, veja abaixo)
#   ORCAMENTO_TOKENS_ALVO    : 0 = alvo completo (padrão); >0 = teto para o alvo
#   CONTEXTO_SERVIDOR_POR_SLOT / RESERVA_PROMPT_TOKENS: o orçamento do exemplo encolhe quando o alvo é
#   grande, para que  alvo + k*exemplo + reserva  <=  contexto. A reserva cobre instruções (~500), debate
#   (até 3 justificativas), saída do modelo (max_tokens) e margem.
ORCAMENTO_TOKENS_EXEMPLO = int(os.getenv("ORCAMENTO_TOKENS_EXEMPLO", "2500"))
ORCAMENTO_TOKENS_ALVO = int(os.getenv("ORCAMENTO_TOKENS_ALVO", "0")) or None
CONTEXTO_SERVIDOR_POR_SLOT = int(os.getenv("CONTEXTO_SERVIDOR_POR_SLOT", "16384"))
RESERVA_PROMPT_TOKENS = int(os.getenv("RESERVA_PROMPT_TOKENS", "5000"))
PISO_ORCAMENTO_EXEMPLO = 200


def orcamento_exemplo_dinamico(tokens_alvo: int, k: int) -> int:
    """Teto por exemplo = min(teto configurado, espaço que sobra no contexto dividido por k)."""
    sobra = CONTEXTO_SERVIDOR_POR_SLOT - RESERVA_PROMPT_TOKENS - tokens_alvo
    return max(PISO_ORCAMENTO_EXEMPLO, min(ORCAMENTO_TOKENS_EXEMPLO, sobra // max(k, 1)))
# LIMPAR_TEXTO=0 (padrão): texto original, só truncado pelo orçamento. LIMPAR_TEXTO=1: limpeza determinística antes.
LIMPAR_TEXTO = os.getenv("LIMPAR_TEXTO", "0").lower() in ("1", "true", "sim")


def preparar_texto(texto: Optional[str], max_tokens: Optional[int]) -> Tuple[str, Dict[str, int]]:
    """Descrição pronta para o prompt (alvo e exemplos usam a mesma regra). Devolve (texto, meta numérico)."""
    if max_tokens is None:                       # sem teto: texto completo (só normaliza quebras de linha)
        max_tokens = 10 ** 9
    if LIMPAR_TEXTO:
        return preparar_descricao(texto, max_tokens)
    original = (texto or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    pronto, truncado = truncar_para_orcamento(original, max_tokens)
    return pronto, {"caracteres_antes": len(texto or ""), "caracteres_final": len(pronto),
                    "tokens_final": contar_tokens(pronto), "truncado": int(truncado)}


# Escala pura de Fibonacci do SIS-ASTROS (1 SP = 1h)
ESCALA_SIS_ASTROS = [1, 2, 3, 5, 8, 13, 21, 34, 55]
ESCALA_SIS_ASTROS_SET = set(ESCALA_SIS_ASTROS)

CAMINHO_MANIFESTO_SPLIT = RAIZ_PROJETO / "results" / "split_sis_astros.json"

# Conversões de unidade do GitLab (padrão da plataforma: 1d = 8h, 1w = 5d)
_HORAS_POR_UNIDADE = {"w": 40.0, "d": 8.0, "h": 1.0, "m": 1.0 / 60.0, "s": 1.0 / 3600.0}


# ==============================================================================
# Utilidades de parsing
# ==============================================================================

def resolver_caminho(caminho_arquivo: Optional[str] = None) -> Path:
    """Precedência: argumento (--arquivo) > variável SIS_ASTROS_PATH > CAMINHO_SIS_ASTROS do config."""
    return Path(caminho_arquivo or os.getenv("SIS_ASTROS_PATH") or CAMINHO_SIS_ASTROS)


def parse_data(texto_data) -> Optional[datetime]:
    """Converte datas do GitLab (ISO, 'YYYY-MM-DD HH:MM:SS UTC', etc.) para datetime UTC."""
    if texto_data is None:
        return None
    s = str(texto_data).strip()
    if not s or s.lower() in ("nan", "nat", "none"):
        return None
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    if pd.isna(dt):
        return None
    return dt.to_pydatetime()


def _parse_estimativa(valor) -> Tuple[Optional[float], Optional[str]]:
    """
    Converte "Time Estimate" para horas, com as mesmas regras do src/limpar.py.
    Retorna (horas, motivo_rejeicao).
    """
    if valor is None or (not isinstance(valor, str) and pd.isna(valor)):
        return None, "vazio"
    s = str(valor).strip().lower()
    if not s or s in ("nan", "none"):
        return None, "vazio"
    if "m" in s:
        return None, "minutos"
    s = s.replace("h", "").replace(",", ".").strip()
    try:
        return float(s), None
    except ValueError:
        return None, "nao_numerico"


def parse_horas_gitlab(valor) -> Optional[float]:
    """
    Converte "Time Spent" para horas. Aceita '8', '8.5', '8h', '1h 30m', '45m', '1d 2h',
    '1h 30m 20s' (formato GitLab com segundos).
    Retorna None quando ausente ou ilegível (nunca inventa um valor).
    """
    if valor is None or (not isinstance(valor, str) and pd.isna(valor)):
        return None
    s = str(valor).strip().lower().replace(",", ".")
    if not s or s in ("nan", "none"):
        return None
    try:
        return float(s)
    except ValueError:
        pass
    partes = re.findall(r"(\d+(?:\.\d+)?)\s*([wdhms])", s)
    if not partes or re.sub(r"(\d+(?:\.\d+)?)\s*([wdhms])|\s", "", s):
        return None
    return sum(float(n) * _HORAS_POR_UNIDADE[u] for n, u in partes)


def _texto(row, col) -> str:
    if not col:
        return ""
    v = row[col]
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return ""
    return str(v).strip()


def hash_arquivo(caminho) -> str:
    """Hash MD5 (12 chars) do conteúdo do arquivo — identifica a versão do dataset."""
    h = hashlib.md5()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(65536), b""):
            h.update(bloco)
    return h.hexdigest()[:12]


# ==============================================================================
# Carregamento
# ==============================================================================

def _ler_tabela(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str).fillna("")
    if path.suffix.lower() == ".csv":
        # Mesma leitura do src/limpar.py: detecta encoding (utf-8-sig, cp1252...) e separador
        from src.limpar import ler_csv
        df, _enc, _sep = ler_csv(path)
        return df
    raise ValueError(f"Formato não suportado: {path.suffix}. Use .csv ou .xlsx.")


def _mapear_colunas(df: pd.DataFrame) -> Dict[str, str]:
    """Mapeia o nome canônico -> nome real da coluna (ignora maiúsculas e espaços nas pontas)."""
    reais = {c.strip().lower(): c for c in df.columns}
    mapa = {nome: reais[nome.lower()] for nome in COLUNAS_SIS_ASTROS if nome.lower() in reais}

    faltando_obrig = [c for c in COLUNAS_OBRIGATORIAS_SIS_ASTROS if c not in mapa]
    if faltando_obrig:
        raise KeyError(
            f"Colunas obrigatórias ausentes no CSV: {faltando_obrig}. "
            f"Encontradas: {list(df.columns)}. Esperadas: {COLUNAS_SIS_ASTROS}."
        )
    faltando = [c for c in COLUNAS_SIS_ASTROS if c not in mapa]
    if faltando:
        print(f"[SIS-ASTROS] Aviso: colunas não encontradas (opcionais): {faltando}")
    return mapa


def carregar_sis_astros(caminho_arquivo: Optional[str] = None) -> List[Dict]:
    """
    Carrega e normaliza o SIS-ASTROS.

    Filtros (documentados para o TCC):
      - Time Estimate vazio, em minutos ou não numérico -> removido;
      - Time Estimate fora da escala Fibonacci -> removido;
      - Created At ausente/ilegível -> removido (não dá para posicionar no split temporal).
    Retorna a lista ordenada por Created At (mais antiga primeiro).
    """
    path = resolver_caminho(caminho_arquivo)
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo SIS-ASTROS não encontrado em: {path.resolve()}\n"
            f"Coloque o CSV limpo em {CAMINHO_SIS_ASTROS} (veja data/sis_astros/LEIA-ME.md) "
            f"ou informe --arquivo <caminho>."
        )

    df = _ler_tabela(path)
    col = _mapear_colunas(df)

    linhas = []
    removidas = {"vazio": 0, "minutos": 0, "nao_numerico": 0, "fora_fibonacci": 0, "sem_data": 0}

    for _, row in df.iterrows():
        sp_horas, motivo = _parse_estimativa(row[col["Time Estimate"]])
        if sp_horas is None:
            removidas[motivo] += 1
            continue
        if not float(sp_horas).is_integer() or int(sp_horas) not in ESCALA_SIS_ASTROS_SET:
            removidas["fora_fibonacci"] += 1
            continue

        data_str = _texto(row, col["Created At"])
        dt = parse_data(data_str)
        if dt is None:
            removidas["sem_data"] += 1
            continue

        linhas.append({
            "ID": _texto(row, col["IID"]),
            "Title": _texto(row, col["Title"]),
            "Description": _texto(row, col.get("Description")),
            "Story_Point": float(int(sp_horas)),
            # Esforço real em horas; None quando não registrado (não é inventado)
            "Tempo_Gasto_Horas": parse_horas_gitlab(row[col["Time Spent"]]) if "Time Spent" in col else None,
            "Creation_Date": data_str,
            "_dt": dt,
            "State": _texto(row, col.get("State")),
            "Labels": _texto(row, col.get("Labels")),
            "Milestone": _texto(row, col.get("Milestone")),
            "Closed_At": _texto(row, col.get("Closed At")),
        })

    # Ordenação estável por data de criação (empate: IID numérico quando possível)
    def _chave(it):
        try:
            iid = int(it["ID"])
        except ValueError:
            iid = 0
        return (it["_dt"], iid)

    linhas.sort(key=_chave)
    for it in linhas:
        it.pop("_dt")

    print(
        f"[SIS-ASTROS] {len(df)} linhas lidas -> {len(linhas)} válidas "
        f"(vazios={removidas['vazio']}, minutos={removidas['minutos']}, "
        f"nao_numerico={removidas['nao_numerico']}, fora_fibonacci={removidas['fora_fibonacci']}, "
        f"sem_data={removidas['sem_data']})"
    )
    return linhas


def distribuicao_escala(itens: List[Dict]) -> Dict[int, int]:
    """Contagem por valor de SP (todos os degraus da escala, inclusive os zerados)."""
    contagem = {sp: 0 for sp in ESCALA_SIS_ASTROS}
    for it in itens:
        sp = int(it["Story_Point"])
        contagem[sp] = contagem.get(sp, 0) + 1
    return dict(sorted(contagem.items()))


# ==============================================================================
# Split temporal
# ==============================================================================

def dividir_temporal(
    itens: List[Dict],
    proporcao: float = 0.7,
    caminho_csv: Optional[str] = None,
    caminho_manifesto: Optional[Path] = CAMINHO_MANIFESTO_SPLIT,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Divide os itens (já ordenados por Created At) em histórico (treino) e teste.

    Não altera o corte — o split é temporal por decisão metodológica — mas reporta a
    distribuição de cada lado e avisa quando algum degrau da escala não aparece em um deles
    (ex.: 55 SP tem poucos itens). Se `caminho_manifesto` for informado, grava os IDs de cada
    lado e as distribuições em JSON para reprodutibilidade.
    """
    if not 0 < proporcao < 1:
        raise ValueError("proporcao deve estar entre 0 e 1")
    corte = int(len(itens) * proporcao)
    treino, teste = itens[:corte], itens[corte:]

    dist_tr, dist_te = distribuicao_escala(treino), distribuicao_escala(teste)
    ausentes_tr = [sp for sp, n in dist_tr.items() if n == 0]
    ausentes_te = [sp for sp, n in dist_te.items() if n == 0]

    print(f"[SIS-ASTROS] Split temporal {proporcao:.0%}/{1 - proporcao:.0%}: "
          f"{len(treino)} histórico / {len(teste)} teste")
    if treino and teste:
        print(f"[SIS-ASTROS] Histórico até {treino[-1]['Creation_Date']} | "
              f"teste a partir de {teste[0]['Creation_Date']}")
    print("    SP | histórico | teste")
    for sp in ESCALA_SIS_ASTROS:
        print(f"  {sp:>4} | {dist_tr[sp]:>9} | {dist_te[sp]:>5}")
    if ausentes_tr:
        print(f"[SIS-ASTROS] AVISO: degraus sem exemplo no histórico (RAG nunca os mostrará): {ausentes_tr}")
    if ausentes_te:
        print(f"[SIS-ASTROS] AVISO: degraus sem item no teste (não serão avaliados): {ausentes_te}")

    if caminho_manifesto is not None:
        caminho_manifesto = Path(caminho_manifesto)
        caminho_manifesto.parent.mkdir(parents=True, exist_ok=True)
        manifesto = {
            "gerado_em": datetime.now(timezone.utc).isoformat(),
            "arquivo": str(caminho_csv) if caminho_csv else None,
            "hash_arquivo": hash_arquivo(caminho_csv) if caminho_csv and Path(caminho_csv).exists() else None,
            "proporcao_historico": proporcao,
            "n_total": len(itens),
            "n_historico": len(treino),
            "n_teste": len(teste),
            "ultima_data_historico": treino[-1]["Creation_Date"] if treino else None,
            "primeira_data_teste": teste[0]["Creation_Date"] if teste else None,
            "distribuicao_historico": dist_tr,
            "distribuicao_teste": dist_te,
            "degraus_ausentes_historico": ausentes_tr,
            "degraus_ausentes_teste": ausentes_te,
            "itens": [
                {
                    "ID": it["ID"],
                    "lado": lado,
                    "Story_Point": int(it["Story_Point"]),
                    "Creation_Date": it.get("Creation_Date"),
                    "Closed_At": it.get("Closed_At", ""),
                    "State": it.get("State", ""),
                    "Labels": it.get("Labels", ""),
                    "Milestone": it.get("Milestone", ""),
                }
                for lado, grupo in (("historico", treino), ("teste", teste))
                for it in grupo
            ],
        }
        with open(caminho_manifesto, "w", encoding="utf-8") as f:
            json.dump(manifesto, f, ensure_ascii=False, indent=2)
        print(f"[SIS-ASTROS] Manifesto do split salvo em: {caminho_manifesto}")

    return treino, teste


# ==============================================================================
# Exemplos históricos para o prompt
# ==============================================================================

def formatar_exemplo(item: Dict, max_tokens: int = ORCAMENTO_TOKENS_EXEMPLO) -> Dict:
    """Converte um item histórico para o formato consumido por prompts.montar_prompt_item."""
    # Limpeza determinística + orçamento de tokens (mesma regra do alvo, orçamento menor).
    # Quebras de linha viram espaço: cada exemplo ocupa uma linha da lista no prompt.
    texto, _meta = preparar_texto(item.get("Description"), max_tokens)
    desc = " ".join(texto.split())
    ex = {
        "id": str(item.get("ID", "")),
        "titulo": item.get("Title", ""),
        "descricao": desc,
        "story_points": float(item["Story_Point"]),
    }
    tempo = item.get("Tempo_Gasto_Horas")
    if tempo is not None and tempo > 0:
        ex["esforco_real_horas"] = round(float(tempo), 1)
    return ex


def _amostrar_few_shot_estratificado(itens_treino: List[Dict], n: int = 5, item_alvo: Optional[Dict] = None) -> List[Dict]:
    """
    Few-shot scale-aware (modo sem RAG): um exemplo por degrau da escala, do menor para o
    maior, pegando o item do meio de cada grupo. Exclui o próprio item alvo.
    """
    if not itens_treino:
        return []

    alvo_id = str(item_alvo.get("ID", "")) if item_alvo else ""
    candidatos = [t for t in itens_treino if str(t.get("ID", "")) != alvo_id]
    if not candidatos:
        return []

    grupos: Dict[float, List[Dict]] = {}
    for it in candidatos:
        grupos.setdefault(float(it["Story_Point"]), []).append(it)

    escolhidos, ids = [], set()
    for deg in ESCALA_SIS_ASTROS:
        if len(escolhidos) >= n:
            break
        grupo = grupos.get(float(deg))
        if grupo:
            c = grupo[len(grupo) // 2]
            if str(c["ID"]) not in ids:
                escolhidos.append(c)
                ids.add(str(c["ID"]))

    if len(escolhidos) < n:
        escolhidos.extend([it for it in candidatos if str(it["ID"]) not in ids][: n - len(escolhidos)])
    return escolhidos[:n]


# ==============================================================================
# Dataset para o runner
# ==============================================================================

def montar_dataset_sis_astros(
    caminho_arquivo: Optional[str] = None,
    n_historico: int = 3,
    split: float = 0.7,
    usar_rag: bool = False,
    k_rag: int = K_RAG,
    fase: str = "teste",
    split_dev: float = 0.7,
    limiar_duplicata: Optional[float] = LIMIAR_DUPLICATA,
) -> List[Dict]:
    """
    Carrega o CSV, aplica o split temporal e devolve os itens a estimar prontos para o runner.

    fase="teste" (padrão): alvos = 30% mais recentes; base de exemplos = os 70% mais antigos.
    fase="dev"           : alvos = os últimos (1 - split_dev) da base histórica; base de exemplos = os
                           split_dev mais antigos dela. O bloco de TESTE não é lido nem como exemplo.
                           Use para calibrar limiar_duplicata, orçamentos de tokens, prompts e
                           convergência ANTES de congelar a configuração.

    - usar_rag=False: few-shot scale-aware com `n_historico` exemplos da base.
    - usar_rag=True : RAG semântico — os `k_rag` itens da base mais parecidos com cada issue,
      excluindo a própria issue, itens criados depois dela, conteúdo idêntico e (se informado)
      similaridade >= limiar_duplicata.
    Os mesmos exemplos vão para os 3 agentes e reaparecem em todas as rodadas do debate.
    A descrição (alvo e exemplos) passa por limpeza determinística + orçamento de tokens; os
    contadores agregados ficam em `prep_meta` e `rag_meta` (nunca texto).
    """
    if fase not in ("teste", "dev"):
        raise ValueError("fase deve ser 'teste' ou 'dev'")
    caminho = resolver_caminho(caminho_arquivo)
    itens = carregar_sis_astros(str(caminho))
    historico, teste = dividir_temporal(itens, split, caminho_csv=str(caminho))

    if fase == "dev":
        corte = int(len(historico) * split_dev)
        base, alvos = historico[:corte], historico[corte:]
        print(f"[SIS-ASTROS] FASE DEV: base de exemplos={len(base)} | alvos={len(alvos)} "
              f"(bloco de teste com {len(teste)} itens NÃO utilizado)")
    else:
        base, alvos = historico, teste

    indice = None
    if usar_rag:
        from src.retrieval import construir_indice, buscar_com_diagnostico
        # a chave do cache já inclui os IDs da base, então dev e teste nunca compartilham índice
        indice = construir_indice(base, str(caminho))

    dataset = []
    k_ctx = k_rag if usar_rag else n_historico
    for item in alvos:
        # 1) alvo completo (ou com o teto configurado)
        descricao, prep_meta = preparar_texto(item.get("Description"), ORCAMENTO_TOKENS_ALVO)
        orc_ex = orcamento_exemplo_dinamico(contar_tokens(descricao), k_ctx)

        # 2) exemplos, truncados ao orçamento dinâmico
        rag_meta = None
        if usar_rag:
            similares, rag_meta = buscar_com_diagnostico(
                titulo_alvo=item["Title"],
                descricao_alvo=item.get("Description", ""),
                iid_alvo=item["ID"],
                data_alvo=item.get("Creation_Date"),
                indice=indice,
                k=k_rag,
                limiar_duplicata=limiar_duplicata,
            )
            historicos = []
            for sim in similares:
                ex = formatar_exemplo(sim, orc_ex)
                ex["similaridade"] = round(float(sim.get("similaridade", 0.0)), 3)
                historicos.append(ex)
        else:
            historicos = [formatar_exemplo(h, orc_ex) for h in _amostrar_few_shot_estratificado(base, n=n_historico, item_alvo=item)]

        tokens_ex = [contar_tokens(h["descricao"]) for h in historicos]
        prep_meta.update(
            orcamento_exemplo=orc_ex,
            tokens_alvo=contar_tokens(descricao),
            tokens_exemplos=sum(tokens_ex),
            exemplos_truncados=sum(1 for h in historicos if "[... texto truncado ...]" in h["descricao"]),
        )

        dataset.append({
            "issue_id": str(item["ID"]),
            "titulo": item["Title"],
            "descricao": descricao,
            "ground_truth": float(item["Story_Point"]),
            "esforco_real_horas": item.get("Tempo_Gasto_Horas"),
            "modo_contexto": "rag" if usar_rag else "few_shot",
            "k_contexto": k_ctx,
            "fase": fase,
            "historicos": historicos,
            "prep_meta": prep_meta,     # contadores (alvo, exemplos, truncamentos); nunca texto
            "rag_meta": rag_meta,       # diagnóstico do vazamento (None sem RAG)
        })

    return dataset


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Diagnóstico do SIS-ASTROS: filtragem e split temporal.")
    ap.add_argument("--arquivo", default=None, help=f"CSV do SIS-ASTROS (padrão: {CAMINHO_SIS_ASTROS})")
    ap.add_argument("--split", type=float, default=0.7)
    a = ap.parse_args()
    caminho = resolver_caminho(a.arquivo)
    dividir_temporal(carregar_sis_astros(str(caminho)), a.split, caminho_csv=str(caminho))