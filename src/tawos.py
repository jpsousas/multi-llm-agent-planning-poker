import mysql.connector, json
import os
from dotenv import load_dotenv

load_dotenv()

db_password = os.getenv("PASSWORD")

def conectar():
    return mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password=db_password,
        database="tawos_db"
    )

def extrair_itens(projeto: str, limite=None):
    """Extrai issues estimadas, ordenadas por data. A ordenacao eh obrigatoria para nao haver vazamento temporal"""
    sql = """
       SELECT i.ID, i.Title, i.Description, i.Story_Point, i.Resolution_Time_Minutes, i.Creation_Date
       FROM Issue i
       JOIN Project p ON i.Project_ID = p.ID
       WHERE p.Name = %s
        AND i.Story_Point IS NOT NULL
        AND i.Story_Point > 0
        AND i.Story_Point <= 100
       ORDER BY i.Creation_Date ASC
    """
    conn = conectar()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(sql, (projeto,))
        linhas = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    return linhas[:limite] if limite else linhas

def montar_dataset(projeto, n_historico=5, split= 0.7):
    """Para cada item de teste, aenxa apenas itens anteriores a ele para exemplo few-shot."""
    itens = extrair_itens(projeto)
    corte = int(len(itens) * split)
    treino, teste = itens[:corte], itens[corte:]

    dataset = []
    for item in teste:
        anteriores = [t for t in treino if t["Creation_Date"] < item["Creation_Date"]]
        historicos = _amostrar_por_escala(anteriores, n_historico)
        dataset.append({
            "issue_id": str(item["ID"]),
            "titulo": item["Title"],
            "descricao": (item["Description"] or "")[:2000],
            "ground_truth": float(item["Story_Point"]),
            "esforco_real_min": float(item["Resolution_Time_Minutes"]),
            "historicos": [{"titulo": h["Title"], "story_points": float(h["Story_Point"])} for h in historicos],
        })
    return dataset

def _amostrar_por_escala(itens, n):
    if not itens:
        return []
    import numpy as np
    sps = [float(i["Story_Point"]) for i in itens]
    faixas = np.linspace(min(sps), max(sps), n + 1)
    escolhidos = []
    for a,b in zip(faixas[:-1], faixas[1:]):
        candidatos = [i for i in itens if a<= float(i["Story_Point"]) <= b]
        if candidatos:
            escolhidos.append(candidatos[len(candidatos) // 2])
    return escolhidos[:n]

if __name__ == "__main__":
    #meramente testes
    dataset = montar_dataset("Projeto Exemplo", n_historico=3, split=0.7)
    print(json.dumps(dataset, indent=2))