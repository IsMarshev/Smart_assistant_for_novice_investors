query = "Что делать Если пришел маржин-колл"
input_em = model.encode(sentences=f"query: {query}", batch_size=1, device = device).tolist()

results = invest_qa_emb.query(
    query_embeddings=[input_em],
    n_results=5
)
print(results)