"""
====================
Pré-processamento DETERMINÍSTICO (sem LLM) do texto das issues, aplicado igualmente à issue alvo e aos
exemplos recuperados, para que o prompt caiba no contexto do modelo.

Etapas de `limpar_descricao` (nesta ordem):
  1. blocos de código/log longos ({code}, {noformat}, ```): mantém só as primeiras linhas;
  2. tags HTML conhecidas: removidas (o texto interno é mantido), entidades decodificadas;
  3. imagens markdown -> [imagem]; links markdown -> "texto [link]"; URLs http(s) -> [link]
     (a pontuação final e os parênteses que fecham a URL são preservados, então nada fica desbalanceado);
  4. tokens longos sem espaço: caminhos -> "…/ultimo-segmento"; nomes com pontos (com.x.Classe.metodo)
     -> "…Classe.metodo"; o resto (hashes) -> [token-longo] (o saldo de parênteses/colchetes é mantido);
  5. linhas repetidas consecutivas: removidas; espaços e linhas em branco normalizados.

Constantes (todas fixadas ANTES de qualquer avaliação de erro; derivadas dos dados do SIS-ASTROS, n = 757):
  CARACTERES_POR_TOKEN = 3,0   ~ p5 da razão caracteres/token medida (2,94; mediana 3,45): só vale sem tokenizador
  TOKEN_LONGO = 40             > p99,99 do comprimento de sequências só com letras (34): quase nunca toca palavras
  LINHAS_BLOCO = 6             ~ 250 tokens (10% do orçamento de 2.500 por exemplo) / p75 de tokens por linha (44,6)
                               (valor fixado antes da regra que o reproduz; reportar a sensibilidade)
  LINHA_BLOCO_MAX = 300        ~ 87 tokens por linha, acima do p90 medido (56,4)
  LINHA_REPETIDA_MIN = 15      heurística sem medição: evita colapsar linhas curtas repetidas (ex.: "---")
Não há limite de comprimento de URL (todas viram marcador) nem regras para stack traces ou linhas gigantes fora de
blocos: nenhuma issue do conjunto era afetada.

`truncar_para_orcamento` corta o texto no começo (cabeça) para respeitar um orçamento de tokens.
`preparar_descricao` faz as duas coisas e devolve também contadores agregados (nunca texto) para
relatórios que não podem expor o conteúdo.

Tudo é função pura do texto: mesma entrada -> mesma saída. Os marcadores entre colchetes deixam claro ao
modelo (e ao leitor da transcrição) onde houve omissão.
"""
import html
import math
import os
import re
from typing import Callable, Dict, Optional, Tuple

CARACTERES_POR_TOKEN = 3.0      # estimativa conservadora quando não há tokenizador (~p5 da razão medida)
LINHAS_BLOCO = 6                # linhas mantidas de cada bloco de código/log longo
LINHA_BLOCO_MAX = 300           # caracteres mantidos por linha dentro de um bloco
TOKEN_LONGO = 40                # tokens sem espaço com pelo menos esse tamanho são encurtados
LINHA_REPETIDA_MIN = 15         # só colapsa linhas repetidas com pelo menos esse tamanho

_RE_BLOCO = re.compile(
    r"(?is)\{code(?::[^}]*)?\}(?P<a>.*?)\{code\}"
    r"|\{noformat\}(?P<b>.*?)\{noformat\}"
    r"|```[\w+-]*\n?(?P<c>.*?)```"
)
_RE_HTML = re.compile(
    r"(?i)</?(p|br|div|span|b|i|u|strong|em|ul|ol|li|a|img|table|thead|tbody|tr|td|th|h[1-6]|pre|code|font|style|script|center)\b[^>]*>"
)
_RE_QUEBRA_HTML = re.compile(r"(?i)<br\s*/?>|</(p|div|li|tr|h[1-6])>")
_RE_URL = re.compile(r"https?://\S+")
_RE_IMG_MD = re.compile(r'!\[([^\]]*)\]\(\s*([^)\s]+)(?:\s+"[^"]*")?\s*\)')
_RE_LINK_MD = re.compile(r'(?<!!)\[([^\]]*)\]\(\s*((?:https?://|/|www\.)[^)\s]*)(?:\s+"[^"]*")?\s*\)')
_PONTUACAO_FINAL = ".,;:!?)]}'\""

_tokenizador = None


def contar_tokens(texto: str) -> int:
    """Conta com o tokenizador do Qwen3 se houver `transformers` (e internet/cache); senão estima."""
    global _tokenizador
    if _tokenizador is None:
        _tokenizador = False
        if os.getenv("TOKENIZADOR", "qwen3").lower() != "none":
            try:
                from transformers import AutoTokenizer
                _tokenizador = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
            except Exception:
                _tokenizador = False
    if _tokenizador:
        return len(_tokenizador.encode(texto))
    return math.ceil(len(texto) / CARACTERES_POR_TOKEN)


def _blocos(texto: str, meta: Dict[str, int], linhas_bloco: int) -> str:
    def troca(m: "re.Match") -> str:
        conteudo = next(g for g in (m.group("a"), m.group("b"), m.group("c")) if g is not None)
        linhas = conteudo.strip("\n").splitlines()
        if len(linhas) <= linhas_bloco:
            return m.group(0)                                   # bloco curto: fica como está
        meta["blocos_omitidos"] += 1
        meta["linhas_omitidas"] += len(linhas) - linhas_bloco
        cabeca = "\n".join(l[:LINHA_BLOCO_MAX] for l in linhas[:linhas_bloco])
        return f"[código/log]\n{cabeca}\n[... {len(linhas) - linhas_bloco} linhas omitidas ...]"
    return _RE_BLOCO.sub(troca, texto)


def _saldo(original: str) -> "tuple[str, str]":
    """Prefixo/sufixo de parênteses e colchetes que mantêm o saldo do token original."""
    pre = suf = ""
    for abre, fecha in (("(", ")"), ("[", "]")):
        saldo = original.count(fecha) - original.count(abre)
        if saldo > 0:
            suf += fecha * saldo
        elif saldo < 0:
            pre += abre * (-saldo)
    return pre, suf


def _token_longo(m: "re.Match") -> str:
    w = m.group(0)
    pre, suf = _saldo(w)                                   # preserva o saldo de ( ) [ ] do texto ao redor
    miolo = re.sub(r"[()\[\]]", "", w)
    if "/" in miolo or "\\" in miolo:
        ultimo = re.split(r"[\\/]", miolo.rstrip("/\\"))[-1][:TOKEN_LONGO]
        return f"{pre}…/{ultimo}{suf}"
    partes = [p for p in miolo.split(".") if p]
    if len(partes) >= 3:                                   # nome qualificado: mantém as 2 últimas partes
        return f"{pre}…" + ".".join(partes[-2:])[:TOKEN_LONGO] + suf
    return f"{pre}[token-longo]{suf}"


def limpar_descricao(texto: Optional[str], linhas_bloco: int = LINHAS_BLOCO,
                     meta: Optional[Dict[str, int]] = None) -> str:
    """Limpeza determinística. `meta` (opcional) recebe contadores agregados das operações."""
    if meta is None:
        meta = {}
    for k in ("blocos_omitidos", "linhas_omitidas", "urls", "imagens", "tokens_longos"):
        meta.setdefault(k, 0)

    t = (texto or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")

    t = _blocos(t, meta, linhas_bloco)

    t = _RE_QUEBRA_HTML.sub("\n", t)
    t = _RE_HTML.sub("", t)
    t = html.unescape(t)

    def imagem(m: "re.Match") -> str:
        meta["imagens"] += 1
        return "[imagem]"
    t = _RE_IMG_MD.sub(imagem, t)

    def link_md(m: "re.Match") -> str:
        meta["urls"] += 1
        texto_link, alvo = m.group(1).strip(), m.group(2)
        return f"{texto_link} [link]" if texto_link and texto_link != alvo else "[link]"
    t = _RE_LINK_MD.sub(link_md, t)

    def url(m: "re.Match") -> str:
        u, cauda = m.group(0), ""
        while u and u[-1] in _PONTUACAO_FINAL:               # não engole o ")" que fecha a URL nem a pontuação
            if u[-1] == ")" and u.count("(") >= u.count(")"):
                break                                        # parêntese que pertence à própria URL
            cauda, u = u[-1] + cauda, u[:-1]
        meta["urls"] += 1
        return "[link]" + cauda
    t = _RE_URL.sub(url, t)

    def longo(m: "re.Match") -> str:
        meta["tokens_longos"] += 1
        return _token_longo(m)
    t = re.sub(rf"\S{{{TOKEN_LONGO},}}", longo, t)

    # linhas: remove repetidas consecutivas e normaliza espaços
    linhas, anterior = [], None
    for l in t.split("\n"):
        l = re.sub(r"[ \t]+", " ", l).strip()
        if len(l) >= LINHA_REPETIDA_MIN and l == anterior:
            continue
        linhas.append(l)
        anterior = l
    t = "\n".join(linhas)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def truncar_para_orcamento(texto: str, max_tokens: int,
                           contar: Callable[[str], int] = contar_tokens) -> Tuple[str, bool]:
    """Mantém o começo do texto até caber em `max_tokens`. Devolve (texto, foi_truncado)."""
    if contar(texto) <= max_tokens:
        return texto, False
    marcador = "\n[... texto truncado ...]"
    corte = len(texto)
    while corte > 0:
        corte = int(corte * 0.9)
        cand = texto[:corte]
        pos = max(cand.rfind("\n"), cand.rfind(" "))
        if pos > corte * 0.5:
            cand = cand[:pos]
        cand = cand.rstrip() + marcador
        if contar(cand) <= max_tokens:
            return cand, True
    return marcador.strip(), True


def preparar_descricao(texto: Optional[str], max_tokens: int,
                       contar: Callable[[str], int] = contar_tokens, **kw) -> Tuple[str, Dict[str, int]]:
    """Limpa e trunca. Devolve (texto_pronto, meta) com contadores (sem conteúdo) para relatórios."""
    meta: Dict[str, int] = {}
    original = texto or ""
    limpo = limpar_descricao(original, meta=meta, **kw)
    pronto, truncado = truncar_para_orcamento(limpo, max_tokens, contar)
    meta.update(
        caracteres_antes=len(original),
        caracteres_apos_limpeza=len(limpo),
        caracteres_final=len(pronto),
        tokens_final=contar(pronto),
        truncado=int(truncado),
    )
    return pronto, meta