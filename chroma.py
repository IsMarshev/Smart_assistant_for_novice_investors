from sentence_transformers import SentenceTransformer
import pandas as pd
from chromadb.utils import embedding_functions
from tqdm import tqdm
import chromadb

client = chromadb.PersistentClient(path="/database/data")
# client = chromadb.HttpClient(host='localhost', port=8000)
print('База данных подключена') 
# invest_qa_emb = client.create_collection(name=db_name, metadata={'hnsw:space': 'cosine'})
collection = client.create_collection(name="my_collection", embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="intfloat/multilingual-e5-small"), metadata={'hnsw:space': 'cosine'})
print('Конфигурация коллекции создана')
df = pd.read_excel('data/data.xlsx')
collection.add(
    documents=df.Answer.to_list(),
    metadatas=[{'question': q} for q in df.Theme.to_list()],
    ids=[str(i) for i in range(len(df))]
)
print('Данные загружены')
# embeddddd = SentenceTransformer("intfloat/multilingual-e5-small")
# query = "Что делать Если пришел маржин-колл"
# input_em = embeddddd.encode(sentences=f"query: {query}", batch_size=1).tolist()

# results = collection.query(
#     query_embeddings=[input_em],
#     n_results=5
# )
# print(results)