import numpy as np

FIBONACCI = [1, 2, 3, 5, 8, 13, 21]

def arredondar_fibonacci(valor: float) -> float:
    return min(FIBONACCI, key=lambda x: abs(x - valor))

def agregar(valores: list[float], metodo: str = "mediana") -> float:
    """Funcao de agregacao. Parametrizavel para permitir ablacao."""
    if metodo == "media":
        bruto = float(np.mean(valores))
    elif metodo == "mediana":
        bruto = float(np.median(valores))
    elif metodo == "moda": # votacao majoritaria
        vals, cont = np.unique(valores, return_counts=True)
        bruto = float(vals[np.argmax(cont)])
    else:
        raise ValueError(metodo)
    return arredondar_fibonacci(bruto)

def verificar_convergencia(valores: list[float], tolerancia: int = 1) -> bool:
    """Convergiu se todos estao a no maximo `tolerancia` degraus
    de distancia na escala de Fibonacci.
    Trabalhar em degraus, e nao em diferenca absoluta, evita que
    itens grandes (13 vs 21) sejam penalizados injustamente frente
    a itens pequenos (1 vs 2)."""
    if len(set(valores)) == 1:
        return True
    else:
        idx = [FIBONACCI.index(arredondar_fibonacci(v)) for v in valores]
    return (max(idx) - min(idx)) <= tolerancia

def dispersao(valores: list[float]) -> float:
    """Metrica-chave para a analise de ancoragem: se o protocolo
    sequencial reduz a dispersao mais rapido que o simultaneo SEM
    ganho proporcional de acuracia, ha indicio de ancoragem."""
    idx = [FIBONACCI.index(arredondar_fibonacci(v)) for v in valores]
    return float(np.std(idx))

if __name__ == "__main__":
    #meramente testes
    print("Arredondar Fibonacci:")
    for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]:
        print(f"{v} => {arredondar_fibonacci(v)}")
    print("\nAgregacao:")
    print(agregar([1.0, 2.0, 3.0], metodo="media"))
    print(agregar([1.0, 2.0, 3.0], metodo="mediana"))
    print(agregar([1.0, 2.0, 3.0], metodo="moda"))
    print("\nConvergencia:")
    print(verificar_convergencia([1.0, 2.0]))
    print(verificar_convergencia([1.0, 3.0]))
    print(verificar_convergencia([1.0, 5.0]))
    print("\nDispersao:")
    print(dispersao([1.0, 2.0]))
    print(dispersao([1.0, 3.0]))
    print(dispersao([1.0, 5.0]))