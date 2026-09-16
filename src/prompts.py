PROMPT_BASE = """Voce é um estimador de esforco participando de uma sessão de Planning Poker de uma equipe ágil.{papel} Estime o esforço necessário para implementar o item apresentado, usando a escala de Fibonacci: 1, 2, 3, 5, 8, 13, 21. Considere complexidade, volume de trabalho e incerteza. Responda APENAS com JSON válido no formato: {{"story_points": <numero>, "justificativa": "<ate 2 frases>"}}""" 

PAPEIS = {
    "neutro": "",
    "backend": "Sua perspectiva é a de uma pessoa desenvolvedora de backend, "
    "atenta a modelagem de dados, APIs e integrações.",
    "frontend": "Sua perspectiva é a de uma pessoa desenvolvedora de frontend, "
    "atenta a interface, estados de tela e experiência de uso.",
    "qa": "Sua perspectiva é a de uma pessoa analista de qualidade, "
    "atenta a casos de teste, regressão e cenarios de borda.",
}

def montar_system_prompt(papel_id: str) -> str:
    return PROMPT_BASE.format(papel=PAPEIS[papel_id])

def montar_prompt_item(titulo, descricao, historicos=None, contexto_rodada=None):
    partes = []
    if historicos:
        linhas = "\n".join(f'- "{h["titulo"]}" => {h["story_points"]} SP'for h in historicos)
        partes.append(f"Itens ja estimados neste projeto:\n{linhas}\n")
    if contexto_rodada:
        partes.append(f"Estimativas da rodada anterior:\n{contexto_rodada}\n" "Considere os argumentos e reestime.\n")
    partes.append(f"Item a estimar:\nTitulo: {titulo}\nDescricao: {descricao}")
    return "\n".join(partes)

if __name__ == "__main__":
    #meramente testes
    print(PROMPT_BASE + "\n\n")
    print(PAPEIS.__str__() + "\n\n")
    print(montar_prompt_item("Titulo do item", "Descricao do item", historicos=[{"titulo": "Item 1", "story_points": 5}, {"titulo": "Item 2", "story_points": 8}], contexto_rodada="Item 1 => 5 SP\nItem 2 => 8 SP"))
    print(montar_system_prompt("backend"))