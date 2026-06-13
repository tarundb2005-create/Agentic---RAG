
# BASIC RAG CHAT APPLICATION

import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_ollama import ChatOllama

#-----------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_PATH = os.path.join(BASE_DIR, "chroma_db")

#-----------------------------------------------------------------------

#load embedding model

embedding_model = SentenceTransformerEmbeddings(
     model_name="BAAI/bge-small-en-v1.5"
)

#----------------------------------------------------------------------

# CONNECT TO CHROMADB

print("Connecting To ChromaDB...")

db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embedding_model
)

print("Connected Successfully")

#----------------------------------------------------------------------

# LOAD QWEN MODEL

print("Loading Qwen 2.5...")

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

print("Readyyy!!!")

#--------------------------------------------------------------------------------

# CHAT LOOP

while True:

    query = input("\nAsk Question: ")

    if query.lower() == "exit":
        break

    #-----------------------------------------------------------------------
    # RETRIEVE DOCUMENTS

    docs = db.similarity_search(
        query,
        k=5
    )

    print("\nRetrieved Documents:", len(docs))

    #-----------------------------------------------------------------------
    # BUILD CONTEXT

    context = ""

    for i, doc in enumerate(docs):

        print("\n" + "="*50)

        print(f"Document {i+1}")

        print(doc.metadata)

        print()

        print(doc.page_content[:300])

        context += doc.page_content + "\n\n"

    #---------------------------------------------------------------------------

    #building prompt

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.

If the answer is not present in the context,
say:

"I could not find that information in the documents."

Context:
{context}

Question:
{query}

Answer:
"""

    #----------------------------------------------------------------------

    #generate response

    print("\nGenerating Answer...\n")

    response = llm.invoke(prompt)

    print("\nAnswer:\n")

    print(response.content)

    #-------------------------------------------------------------------------------------------

    print("\nSources:\n")

    for doc in docs[:3]:

        print(
            f"{doc.metadata.get('source','Unknown')} "
            f"(Page {doc.metadata.get('page',0)+1})"
        )

    print("\n" + "="*80)



        