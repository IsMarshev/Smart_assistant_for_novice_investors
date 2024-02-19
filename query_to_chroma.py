import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.HttpClient(host='localhost', port=8000)
collection = client.get_collection("my_collection")

embeddddd = SentenceTransformer("intfloat/multilingual-e5-small")
query = "Чем отличается акции от облигаций"
input_em = embeddddd.encode(sentences=f"query: {query}", batch_size=1).tolist()

results = collection.query(
    query_embeddings=[input_em],
    n_results=5
)
print(results)