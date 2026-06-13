
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

embedding_model = SentenceTransformerEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

collection = db._collection

print("Total vectors:", collection.count())

results = db.similarity_search("machine learning", k=5)

for doc in results:
    print(doc.metadata)