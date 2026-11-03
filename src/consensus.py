import numpy as np
from typing import List, Dict, Tuple, Optional


def parse_escala(escala_str: str) -> List[int]:
    """Converte a string da escala (ex: '1, 2, 3, 5') em uma lista ordenada de inteiros únicos."""
    return sorted(set(int(x.strip()) for x in escala_str.split(",")))


def arredondar_fibonacci(valor: float, escala: List[int]) -> float:
    """Mapeia um valor numérico contínuo para o ponto mais próximo da escala fornecida."""
    if not escala:
        return valor
    return float(min(escala, key=lambda x: abs(x - valor)))


def agregar(valores: List[float], escala: List[int], metodo: str = "mediana") -> float:
    """Agrega os lances individuais (padrão: mediana) e ajusta para a escala."""
    if not valores:
        return 1.0
    if metodo == "media":
        bruto = float(np.mean(valores))
    elif metodo == "mediana":
        bruto = float(np.median(valores))
    elif metodo == "moda":
        vals, cont = np.unique(valores, return_counts=True)
        bruto = float(vals[np.argmax(cont)])
    else:
        raise ValueError(f"Método de agregação desconhecido: {metodo}")
    return arredondar_fibonacci(bruto, escala)


def desvio_padrao_sp(valores: List[float]) -> float:
    """Desvio padrão populacional (sigma_t) dos Story Points da rodada."""
    if len(valores) <= 1:
        return 0.0
    return float(np.std(valores, ddof=0))


def obter_indice_escala(valor: float, escala: List[int]) -> int:
    """Índice (degrau) do valor mais próximo na escala de Fibonacci."""
    valor_arredondado = int(arredondar_fibonacci(valor, escala))
    try:
        return escala.index(valor_arredondado)
    except ValueError:
        return int(np.argmin([abs(x - valor) for x in escala]))


def distancia_degraus(v1: float, v2: float, escala: List[int]) -> int:
    """Distância ordinal (em degraus de Fibonacci) entre dois valores."""
    return abs(obter_indice_escala(v1, escala) - obter_indice_escala(v2, escala))


def calcular_dispersao_degraus(valores: List[float], escala: List[int]) -> int:
    """Amplitude máxima, em degraus de Fibonacci, entre os lances da rodada (descritivo)."""
    if len(valores) <= 1:
        return 0
    indices = [obter_indice_escala(v, escala) for v in valores]
    return int(max(indices) - min(indices))


def avaliar_convergencia_rodada(
    valores: List[float],
    rodada: int,
    escala: List[int],
    max_rodadas: int = 3,
    lances_atuais: Optional[Dict[str, float]] = None,
    lances_anteriores: Optional[Dict[str, float]] = None,
) -> Tuple[bool, str, int]:
    """
    Regra de parada do Planning Poker multiagente (critério operacional do H5):
      1. Unanimidade em qualquer rodada -> encerra com o valor unânime.
      2. Estagnação na R2 (cada agente repete o lance da R1) -> encerra com a mediana da R2.
      3. Teto na R3 sem unanimidade -> encerra com a mediana da R3.

    Retorna (convergiu, motivo, dispersao_degraus). `convergiu` é True apenas para
    unanimidade ou estagnação; no teto da R3 é False e o grafo encerra pela mediana.
    """
    dispersao = calcular_dispersao_degraus(valores, escala)

    if len(set(valores)) <= 1:
        return True, f"consenso_unanime_r{rodada}", dispersao

    if rodada == 1:
        return False, f"divergencia_r1_abrir_debate (delta={dispersao} degraus)", dispersao

    if rodada == 2:
        estagnou = (
            lances_atuais is not None
            and lances_anteriores is not None
            and set(lances_atuais) == set(lances_anteriores)
            and all(abs(lances_atuais[k] - lances_anteriores[k]) < 1e-5 for k in lances_atuais)
        )
        if estagnou:
            return True, f"estagnacao_r2_mediana (delta={dispersao} degraus)", dispersao
        return False, f"divergencia_r2_avancar_r3 (delta={dispersao} degraus)", dispersao

    return False, f"desempate_mediana_r3 (delta={dispersao} degraus)", dispersao


def identificar_extremos(estimativas_rodada: List[Dict]) -> Dict:
    """Agentes que emitiram os lances mínimo e máximo na rodada."""
    if not estimativas_rodada:
        return {}
    min_est = min(estimativas_rodada, key=lambda x: x["story_points"])
    max_est = max(estimativas_rodada, key=lambda x: x["story_points"])
    return {
        "min": {"agente": min_est["agente"], "papel": min_est["papel"], "story_points": min_est["story_points"]},
        "max": {"agente": max_est["agente"], "papel": max_est["papel"], "story_points": max_est["story_points"]},
        "diferenca": max_est["story_points"] - min_est["story_points"],
    }


def formatar_resumo_debate(estimativas_rodada: List[Dict], rodada: int) -> str:
    """Resumo explicável dos lances e justificativas CoT para a próxima rodada."""
    valores = [e["story_points"] for e in estimativas_rodada]
    sigma = desvio_padrao_sp(valores)
    extremos = identificar_extremos(estimativas_rodada)

    linhas = [f"Lances da Rodada {rodada} (Desvio Padrão sigma = {sigma:.2f}):"]
    for e in estimativas_rodada:
        linhas.append(
            f'- **{e["papel"].capitalize()}** ({e["modelo"]}): {int(e["story_points"])} SP\n'
            f'  *Justificativa CoT:* {e["justificativa"]}'
        )

    if extremos.get("diferenca", 0) > 0:
        linhas.append(
            f"\n*Divergência detectada:* O papel {extremos['min']['papel'].capitalize()} estimou o menor valor "
            f"({int(extremos['min']['story_points'])} SP) e o papel {extremos['max']['papel'].capitalize()} estimou o maior valor "
            f"({int(extremos['max']['story_points'])} SP)."
        )

    return "\n".join(linhas)


if __name__ == "__main__":
    escala = parse_escala("1, 2, 3, 5, 8, 13, 21, 34, 55")
    print("Escala:", escala)
    print("R1 [5,5,5]:", avaliar_convergencia_rodada([5, 5, 5], 1, escala))
    print("R1 [3,5,8]:", avaliar_convergencia_rodada([3, 5, 8], 1, escala))
    lances = {"a": 3.0, "b": 5.0, "c": 8.0}
    print("R2 estagnada:", avaliar_convergencia_rodada([3, 5, 8], 2, escala,
                                                      lances_atuais=lances, lances_anteriores=lances))
    print("R3 sem unanimidade:", avaliar_convergencia_rodada([3, 5, 8], 3, escala))
    print("Mediana:", agregar([3, 5, 8], escala))
