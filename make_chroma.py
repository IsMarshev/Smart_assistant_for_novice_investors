from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
device = 'cuda'
model = SentenceTransformer("intfloat/multilingual-e5-small")

def make_passage(ans):
    return f'passage: {ans}'

def make_query(q):
    return f'query: {q}'

def make_db(df_path, db_name):
    df = pd.read_excel(df_path)

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        documents.append(row['Answer'])
        embedding = model.encode(sentences=make_passage(row['Answer']), batch_size=1, device = device).tolist()
        embeddings.append(embedding)
        metadatas.append({'source': row['Theme']})
        ids.append(str(index + 1))

    client = chromadb.Client(Settings(persist_directory='db/'))
    invest_qa_emb = client.create_collection(name=db_name, metadata={'hnsw:space': 'cosine'})

    invest_qa_emb.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
if __name__=='__main__':
    df_path = 'data\data.xlsx'
    db_name = "invest_qa_emb"
    make_db(df_path=df_path, db_name=db_name)
