from dataclasses import dataclass, field
#modelo padrao em 16 de dezembro de 2026
MODELO_PADRAO = "groq:openai/gpt-oss-120b"

@dataclass
class Condicao:
    nome: str
    agentes: list[tuple[str, str, str]] # (id, papel_id, modelo)
    protocolo: str = "simultaneo" # "simultaneo" | "sequencial"
    deliberacao: bool = False
    max_rodadas: int = 1
    temperatura: float = 0.0

numero_de_agentes = 3 # numero de estimadores: mantem o orcamento igual entre C1, C2 e C3
CONDICOES = {
    
    # C0: uma unica chamada. sem consenso ou piso de comparação
    "C0": Condicao(
        nome="individual",
        agentes=[("unico", "neutro", MODELO_PADRAO)],
    ),
    
    # C1: self-consistency. MESMO modelo, MESMO papel (neutro), N amostras N aka numero_de_agentes.
    # Temperatura > 0 e obrigatoria aqui, senao as N amostras sao identicas.
    "C1": Condicao(
        nome="self_consistency",
        agentes=[(f"amostra_{i}", "neutro", MODELO_PADRAO) for i in range(numero_de_agentes)],
        temperatura=0.7,
    ),
    
    # C2: homogeneo com papeis. MESMO modelo, papeis DIFERENTES.
    "C2": Condicao(
        nome="homogeneo_papeis",
        agentes=[("backend", "backend", MODELO_PADRAO),
                 ("frontend", "frontend", MODELO_PADRAO),
                 ("qa", "qa", MODELO_PADRAO)],

    ),
    
    # C3: heterogeneo com papeis. Modelos DIFERENTES, mesmos papeis.
    "C3": Condicao(
        nome="heterogeneo_papeis",
        agentes=[("backend", "backend", "groq:groq:openai/gpt-oss-120b"),
        ("frontend", "frontend", "openrouter:poolside/laguna-s-2.1:free"),
        ("qa", "qa", "google_genai:gemini-2.5-flash-lite")],
    ),
    
    # Variantes com deliberacao e protocolo, para a RQ de ancoragem
    "C3_delib_simultaneo": Condicao(
    nome="heterogeneo_delib_simultaneo",
        agentes=[("backend", "backend", "groq:groq:openai/gpt-oss-120b"),
                 ("frontend", "frontend", "openrouter:poolside/laguna-s-2.1:free"),
                 ("qa", "qa", "google_genai:gemini-2.5-flash-lite")],
        protocolo="simultaneo",
        deliberacao=True,
        max_rodadas=3
    ),
    "C3_delib_sequencial": Condicao(
        nome="heterogeneo_delib_sequencial",
        agentes=[("backend", "backend", "groq:groq:openai/gpt-oss-120b"),
                 ("frontend", "frontend", "openrouter:poolside/laguna-s-2.1:free"),
                 ("qa", "qa", "google_genai:gemini-2.5-flash-lite")],
        protocolo="sequencial",
        deliberacao=True,
        max_rodadas=3
    ),
}

if __name__ == "__main__":
    #meramente testes
    for condicao_id, condicao in CONDICOES.items():
        print(f"Condicao {condicao_id}: {condicao}")