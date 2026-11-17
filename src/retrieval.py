"""
src/retrieval.py
================
Recuperação semântica de issues históricas (RAG in-memory) para o MASPE.

Para cada issue a estimar, devolve as K issues do HISTÓRICO mais parecidas, respeitando:
  1. Anti-leakage por IID: a própria issue nunca é retornada (ID != ID_alvo).
  2. Causalidade temporal: só entram issues criadas ANTES da issue alvo.
  (O índice é construído só com o histórico do split 70/30, então o teste nunca é indexado.)

Modelo de embeddings: sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (src/config.py)
  - 768 dimensões; suporta PT-BR, roda em CPU, independente do LLM usado na estimativa.
  - Carregado uma única vez por processo.

Cache: results/cache/embeddings_<chave>.npy (+ _ids.npy)
  - <chave> = hash(conteúdo do CSV + IDs indexados + modelo): mudar o arquivo ou o split
    gera um cache novo, nunca reaproveita embeddings de outro conjunto.
  - Busca = produto escalar com vetores normalizados (similaridade de cosseno) em numpy.

Proteção contra vazamento (além de ID e causalidade temporal):
  3. Conteúdo idêntico (mesmo título+descrição, ID diferente) é sempre excluído.
  4. Quase-duplicatas: candidatos com cosseno >= `limiar_duplicata` são excluídos. O limiar deve ser
     calibrado no bloco de DESENVOLVIMENTO (distribuição de `sim_max` + inspeção de pares), nunca no teste.
  buscar_com_diagnostico() devolve também `sim_max` e as contagens de exclusão, para registrar por item.
"""

import hashlib
import re
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

from src.config import RAIZ_PROJETO, MODELO_EMBEDDINGS, LIMIAR_DUPLICATA
from src.sis_astros import parse_data, hash_arquivo, formatar_exemplo

DIRETORIO_CACHE = RAIZ_PROJETO / "results" / "cache"

_modelo = None                      # SentenceTransformer carregado uma vez
_lock_modelo = threading.Lock()
_indices_memoria: Dict[str, Dict] = {}  # chave do cache -> índice


def obter_modelo_embeddings():
    """Carrega (uma única vez) e devolve o modelo de embeddings."""
    global _modelo
    if _modelo is None:
        with _lock_modelo:
            if _modelo is None:
                from sentence_transformers import SentenceTransformer
                _modelo = SentenceTransformer(MODELO_EMBEDDINGS)
    return _modelo


def _texto_para_embedding(titulo: str, descricao: str) -> str:
    """Título + início da descrição (o título domina; a descrição enriquece sem texto longo)."""
    desc_curta = (descricao or "").strip()[:400]
    titulo = (titulo or "").strip()
    return f"{titulo} - {desc_curta}" if desc_curta else titulo


def _hash_conteudo(titulo, descricao) -> str:
    """Hash do conteúdo normalizado (minúsculas, espaços colapsados) para achar cópias exatas."""
    base = re.sub(r"\s+", " ", f"{titulo or ''}|{descricao or ''}").strip().lower()
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def _parse_data(texto_data):
    """Mantido por compatibilidade: delega para sis_astros.parse_data (datetime UTC ou None)."""
    return parse_data(texto_data)


def _chave_cache(caminho_csv: str, ids: List[str]) -> str:
    h = hashlib.md5()
    h.update(hash_arquivo(caminho_csv).encode())
    h.update(MODELO_EMBEDDINGS.encode())
    h.update("\n".join(ids).encode("utf-8"))
    return h.hexdigest()[:16]


def construir_indice(
    itens: List[Dict],
    caminho_csv: str,
    forcar_rebuild: bool = False,
    diretorio_cache: Optional[Path] = None,
) -> Dict:
    """
    Constrói (ou carrega do cache) o índice de embeddings das issues HISTÓRICAS.

    Args:
        itens: issues do histórico (dicts de sis_astros.carregar_sis_astros).
        caminho_csv: CSV de origem (entra na chave do cache).
        forcar_rebuild: ignora o cache e recalcula.
        diretorio_cache: pasta do cache (padrão: results/cache).

    Returns:
        {"embeddings": np.ndarray (N, D), "itens": list, "datas": list, "modelo": str, "chave": str}
    """
    itens_validos = [it for it in itens if str(it.get("Title", "")).strip()]
    ids = [str(it["ID"]) for it in itens_validos]
    chave = _chave_cache(caminho_csv, ids)

    if not forcar_rebuild and chave in _indices_memoria:
        return _indices_memoria[chave]

    pasta = Path(diretorio_cache) if diretorio_cache else DIRETORIO_CACHE
    pasta.mkdir(parents=True, exist_ok=True)
    caminho_npy = pasta / f"embeddings_{chave}.npy"
    caminho_ids = pasta / f"embeddings_{chave}_ids.npy"

    embeddings = None
    if not forcar_rebuild and caminho_npy.exists() and caminho_ids.exists():
        ids_cache = np.load(str(caminho_ids)).tolist()
        if ids_cache == ids:
            embeddings = np.load(str(caminho_npy))
            print(f"[RAG] Embeddings carregados do cache: {caminho_npy.name}")

    if embeddings is None:
        print(f"[RAG] Gerando embeddings para {len(itens_validos)} issues do histórico...")
        textos = [_texto_para_embedding(it["Title"], it.get("Description", "")) for it in itens_validos]
        embeddings = obter_modelo_embeddings().encode(
            textos, normalize_embeddings=True, show_progress_bar=len(textos) > 50, batch_size=64
        )
        np.save(str(caminho_npy), embeddings)
        np.save(str(caminho_ids), np.array(ids))
        print(f"[RAG] Cache salvo em: {caminho_npy}")

    datas = [parse_data(it.get("Creation_Date")) for it in itens_validos]
    sem_data = sum(d is None for d in datas)
    if sem_data:
        print(f"[RAG] Aviso: {sem_data} issue(s) do histórico sem data válida; "
              f"a causalidade temporal não pode ser garantida para elas.")

    indice = {
        "embeddings": embeddings,
        "itens": itens_validos,
        "datas": datas,
        "hashes": [_hash_conteudo(it.get("Title"), it.get("Description")) for it in itens_validos],
        "modelo": MODELO_EMBEDDINGS,
        "chave": chave,
    }
    _indices_memoria[chave] = indice
    return indice


def buscar_com_diagnostico(
    titulo_alvo: str,
    descricao_alvo: str,
    iid_alvo: str,
    data_alvo: Optional[str],
    indice: Dict,
    k: int = 3,
    limiar_duplicata: Optional[float] = LIMIAR_DUPLICATA,
) -> Tuple[List[Dict], Dict]:
    """
    Retorna (similares, meta). `similares` são as K issues do índice mais parecidas com a alvo.

    Exclusões, nesta ordem:
      1. o próprio IID;
      2. issues criadas na mesma data/hora da alvo ou depois dela (se ambas tiverem data);
      3. conteúdo idêntico ao da alvo (mesmo título+descrição), mesmo com ID diferente;
      4. se `limiar_duplicata` for dado: candidatos com cosseno >= limiar (quase-duplicatas).

    `meta` para registrar por item (relatório de vazamento):
      n_elegiveis, n_conteudo_identico, sim_max (maior cosseno entre os candidatos restantes antes do
      limiar; None se não restou nenhum), n_removidas_limiar, limiar.
    """
    embeddings_base: np.ndarray = indice["embeddings"]
    itens_base: List[Dict] = indice["itens"]
    datas_base = indice.get("datas") or [parse_data(it.get("Creation_Date")) for it in itens_base]
    hashes_base = indice.get("hashes") or [
        _hash_conteudo(it.get("Title"), it.get("Description")) for it in itens_base
    ]

    meta = {"n_elegiveis": 0, "n_conteudo_identico": 0, "sim_max": None,
            "n_removidas_limiar": 0, "limiar": limiar_duplicata}

    dt_alvo = parse_data(data_alvo)
    elegiveis = [
        i for i, it in enumerate(itens_base)
        if str(it["ID"]) != str(iid_alvo)
        and not (dt_alvo is not None and datas_base[i] is not None and datas_base[i] >= dt_alvo)
    ]
    meta["n_elegiveis"] = len(elegiveis)

    hash_alvo = _hash_conteudo(titulo_alvo, descricao_alvo)
    identicas = [i for i in elegiveis if hashes_base[i] == hash_alvo]
    meta["n_conteudo_identico"] = len(identicas)
    elegiveis = [i for i in elegiveis if hashes_base[i] != hash_alvo]
    if not elegiveis:
        return [], meta

    texto_alvo = _texto_para_embedding(titulo_alvo, descricao_alvo)
    vetor_alvo = obter_modelo_embeddings().encode([texto_alvo], normalize_embeddings=True)[0]
    similaridades = embeddings_base[elegiveis] @ vetor_alvo
    meta["sim_max"] = float(similaridades.max())

    if limiar_duplicata is not None:
        manter = similaridades < limiar_duplicata
        meta["n_removidas_limiar"] = int((~manter).sum())
        elegiveis = [i for i, m in zip(elegiveis, manter) if m]
        similaridades = similaridades[manter]
        if not elegiveis:
            return [], meta

    ordem = np.argsort(-similaridades, kind="stable")[: min(k, len(elegiveis))]
    resultados = []
    for pos in ordem:
        item = dict(itens_base[elegiveis[pos]])
        item["similaridade"] = float(similaridades[pos])
        resultados.append(item)
    return resultados, meta


def buscar_issues_similares(
    titulo_alvo: str,
    descricao_alvo: str,
    iid_alvo: str,
    data_alvo: Optional[str],
    indice: Dict,
    k: int = 3,
    limiar_duplicata: Optional[float] = LIMIAR_DUPLICATA,
) -> List[Dict]:
    """Versão compatível: só a lista de similares (veja buscar_com_diagnostico para o diagnóstico)."""
    return buscar_com_diagnostico(
        titulo_alvo, descricao_alvo, iid_alvo, data_alvo, indice, k, limiar_duplicata
    )[0]


def formatar_historicos_rag(similares: List[Dict]) -> List[Dict]:
    """
    Converte o resultado da busca para o formato de prompts.montar_prompt_item:
    título, descrição curta, estimativa planejada (SP), esforço real (h, se houver) e similaridade.
    """
    historicos = []
    for s in similares:
        ex = formatar_exemplo(s)
        ex["similaridade"] = round(float(s.get("similaridade", 0.0)), 3)
        historicos.append(ex)
    return historicos


def resetar_indice():
    """Limpa os índices em memória (útil para testes). O modelo de embeddings é mantido."""
    _indices_memoria.clear()