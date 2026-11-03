from typing import Optional

PROMPT_BASE = """Você é um especialista em desenvolvimento de software participando de uma sessão de Planning Poker de uma equipe ágil.{papel} Sua responsabilidade é estimar o esforço relativo necessário para implementar o item do backlog apresentado, utilizando estritamente a sequência de Fibonacci: {escala_fibonacci}

Critérios de avaliação para estimativa de Story Points:
- Complexidade técnica (arquitetura, algoritmos, dependências de sistemas externos).
- Volume de trabalho (quantidade de código, telas, endpoints, modelos ou testes).
- Incerteza e risco (ambiguidades de requisitos, cenários de borda, impacto potencial em regressão).

Responda OBRIGATORIAMENTE em JSON válido no formato:
{{
  "justificativa": "<raciocínio Chain-of-Thought explicando a decisão técnica com base no seu papel>",
  "story_points": <número inteiro da escala Fibonacci>
}}"""

PAPEIS = {
    "neutro": "",
    "backend": " Sua perspectiva é a de um(a) desenvolvedor(a) Backend sênior, com foco em modelagem de dados, APIs, persistência, regras de negócio, performance e integrações.",
    "frontend": " Sua perspectiva é a de um(a) desenvolvedor(a) Frontend sênior, com foco em interface (UI), experiência de usuário (UX), gestão de estados, responsividade e integração com APIs.",
    "qa": " Sua perspectiva é a de um(a) analista de qualidade (QA) sênior, com foco em estratégias de teste (unitários, integração, regressão), cobertura de requisitos e cenários de borda.",
}

def montar_system_prompt(papel_id: str, escala_fibonacci: str) -> str:
    papel_texto = PAPEIS.get(papel_id, "")
    return PROMPT_BASE.format(papel=papel_texto, escala_fibonacci=escala_fibonacci)

def montar_prompt_item(
    titulo: str,
    descricao: str,
    historicos: Optional[list[dict]] = None,
    contexto_rodada: Optional[str] = None,
    rodada: int = 1,
    protocolo: str = "simultaneo"
) -> str:
    partes = []

    # 1. Contextualização histórica (Few-Shot scale-aware ou RAG semântico).
    #    Os mesmos exemplos vão para todos os agentes e se repetem em todas as rodadas.
    if historicos:
        linhas_hist = []
        tem_rag = any("similaridade" in h for h in historicos)
        header = "### Issues Históricas Mais Parecidas do Projeto (com estimativa e esforço real):" if tem_rag \
                 else "### Exemplos de Referência Histórica do Projeto (Scale-Aware):"
        for h in historicos:
            sp = int(h["story_points"])
            linha = f'- "{h["titulo"]}"'
            if h.get("descricao"):
                linha += f' — {h["descricao"]}'
            linha += f' → Estimativa Planejada em Story Points: {sp} SP'
            if h.get("esforco_real_horas") is not None:
                linha += f' | Esforço Real Gasto em Horas: {h["esforco_real_horas"]:g}h'
            linhas_hist.append(linha)
        partes.append(f"{header}\n" + "\n".join(linhas_hist) + "\n")

    # 2. Contexto de Protocolo (Sequencial na Rodada 1 ou Debate CoT nas Rodadas > 1)
    if contexto_rodada:
        if rodada == 1 and protocolo == "sequencial":
            partes.append(
                "### Estimativas Anteriores da Equipe (Ordem Sequencial):\n"
                f"{contexto_rodada}\n"
                "Considere os lances e justificativas dos colegas apresentados acima ao formular sua estimativa.\n"
            )
        else:
            partes.append(
                "### Contexto do Debate da Rodada Anterior (Divergência Detectada):\n"
                f"{contexto_rodada}\n"
                "Houve divergência entre os membros da equipe. Avalie os argumentos dos outros especialistas, "
                "desenvolva seu raciocínio crítico (Chain-of-Thought) e apresente seu novo lance de consenso "
                "(você pode revisar sua estimativa ou mantê-la justificando o motivo técnico).\n"
            )

    # 3. Item a estimar
    partes.append(
        "### Item do Backlog a Estimar:\n"
        f"**Título:** {titulo}\n"
        f"**Descrição:** {descricao if descricao else 'Sem descrição detalhada fornecida.'}"
    )

    return "\n".join(partes)

if __name__ == "__main__":
    print("--- Teste System Prompt Backend ---")
    print(montar_system_prompt("backend", "1, 2, 3, 5, 8, 13, 21, 34, 55"))
    print("\n--- Teste Item Prompt (Few-Shot, Debate Rodada 2) ---")
    print(montar_prompt_item(
        titulo="Criar endpoint de autenticação OAuth2",
        descricao="Implementar fluxo de autorização com tokens JWT e refresh tokens.",
        historicos=[{"titulo": "CRUD de usuários", "story_points": 3}, {"titulo": "Gateway de Pagamento", "story_points": 13}],
        contexto_rodada="Backend estimou 13 SP: APIs complexas.\nFrontend estimou 3 SP: Apenas formulário de login.",
        rodada=2,
        protocolo="simultaneo"
    ))