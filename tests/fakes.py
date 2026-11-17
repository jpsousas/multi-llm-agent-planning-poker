"""
tests/fakes.py
==============
Utilidades compartilhadas pelos testes (sem rede, sem modelo real):
  - FakeEncoder: "embedding" determinístico por saco de palavras (cosseno alto = vocabulário parecido);
  - csv_sis_astros: CSV temporário com o cabeçalho do SIS-ASTROS;
  - patch_retrieval: troca o encoder e a pasta de cache do retrieval por versões de teste.
"""
import csv
import hashlib
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

CABECALHO = ["IID", "Title", "State", "Author", "Assignee", "Time Estimate", "Time Spent",
             "Created At", "Closed At", "Milestone", "Labels", "Description", "URL"]

DIM = 256


class FakeEncoder:
    """Saco de palavras normalizado (hash das palavras -> 256 dimensões). Mesmo interface do SentenceTransformer."""

    def encode(self, textos, normalize_embeddings=True, show_progress_bar=False, batch_size=64):
        un = isinstance(textos, str)
        if un:
            textos = [textos]
        out = np.zeros((len(textos), DIM), dtype=np.float32)
        for i, t in enumerate(textos):
            for w in re.findall(r"\w+", (t or "").lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        if normalize_embeddings:
            n = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.where(n == 0, 1, n)
        return out[0] if un else out


def csv_sis_astros(linhas) -> str:
    """linhas: dicts com chaves do CABEÇALHO (as ausentes ficam vazias). Devolve o caminho (apague depois)."""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8-sig", newline="")
    w = csv.DictWriter(tmp, fieldnames=CABECALHO)
    w.writeheader()
    for l in linhas:
        w.writerow({c: l.get(c, "") for c in CABECALHO})
    tmp.close()
    return tmp.name


def linha(iid, est, dia, titulo=None, desc=None, spent=""):
    return {"IID": str(iid), "Title": titulo or f"Issue {iid}", "Time Estimate": str(est),
            "Time Spent": spent, "Created At": f"2024-01-{dia:02d} 10:00:00 UTC",
            "Description": desc if desc is not None else f"Descrição {iid}", "State": "closed"}


@contextmanager
def patch_retrieval(pasta_cache=None):
    """Encoder falso + cache em pasta temporária (nada é baixado nem gravado no projeto)."""
    import src.retrieval as r
    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(r, "obter_modelo_embeddings", lambda: FakeEncoder()), \
             patch.object(r, "DIRETORIO_CACHE", Path(pasta_cache or tmp)):
            r.resetar_indice()
            yield r
            r.resetar_indice()
