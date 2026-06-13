

# HYBRID RETRIEVAL

#1.Using Vector Database
#2Using Bm25 Search

import pickle 
import os 
from reranker import rerank
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

#-------------------------------------------------------------------------
#Using Paths

# Get project root directory
BASE_DIR = os.path.dirname( 
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Path where ChromaDB vectors are stored
DB_PATH = os.path.join(
    BASE_DIR,"chroma_db"
)

# Path where chunked documents were saved
CHUNK_PATH = os.path.join(
    BASE_DIR,"chunks.pkl"
)

#----------------------------------------------------------------------

# LOAD CHUNKS
print("Loading chunks...")

# Open chunks.pkl in binary read mode
with open(CHUNK_PATH, "rb") as f:

    # Load all chunked documents
    chunks = pickle.load(f)

# Display total number of chunks
print("Chunks Loaded:", len(chunks))

#------------------------------------------------------------------------
# BUILD BM25 INDEX

# Empty list to store tokenized chunks
tokenized_chunks = []

# Process every chunk
for chunk in chunks:
    # Convert text to lowercase
    # Split into words (tokens)
    tokenized_chunks.append(
        chunk.page_content.lower().split()
    )

# Create BM25 search index
bm25 = BM25Okapi(tokenized_chunks)
print("BM25 Ready")

#-------------------------------------------------------------------
# LOAD CHROMA VECTOR DATABASE
# Load embedding model

embedding_models = SentenceTransformerEmbeddings(
    model_name = "BAAI/bge-small-en-v1.5"
)

# Connect to existing Chroma database
db = Chroma(
    # Location of stored vectors
    persist_directory= DB_PATH,

    # Embedding model used during retrieval
    embedding_function = embedding_models
)
print("Chroma Ready")


#----------------------------------------------------------------------
# HYBRID SEARCH FUNCTION

def hybrid_search(query, k=5):

    """
    Performs Hybrid Retrieval

    Parameters:
    ----------
    query : str
        User question

    k : int
        Number of top documents from
        Vector Search and BM25

    Returns:
    -------
    combined_docs
        Unique documents retrieved from
        both retrieval methods
    """

# VECTOR SEARCH
# Convert query into embedding
# Search nearest vectors in ChromaDB

def hybrid_search(query, k=5):
    vector_docs = db.similarity_search(query,k=k)
    # BM25 SEARCH
    # # Tokenize user query
    query_tokens = query.lower().split()

# Compute BM25 score for every chunk
    scores = bm25.get_scores(query_tokens)

# Sort chunk indices based on score
# Highest score first

    ranked_indices = sorted(
        range(len(scores)),
        key = lambda i: scores[i],
        reverse=True 
        )

    # Store top BM25 documents
    bm25_docs = []

    # Take top-k chunks
    for idx in ranked_indices[:k]:
        bm25_docs.append(chunks[idx])


#----------------------------------------------------------------
    # MERGE RESULTS
    # Final result list
    combined_docs = []

    # Used to remove duplicate chunks
    seen_texts = set()

    # Combine vector and BM25 results
    for doc in vector_docs + bm25_docs:
        # Extract chunk text
        text = doc.page_content

        # Add only if not already present
        if text not in seen_texts:
            seen_texts.add(text)
            combined_docs.append(doc)

        # Return unique retrieved chunks
    # Rerank documents
    combined_docs = rerank(
        query,combined_docs,top_k=3
    )
    return combined_docs


# ==========================================================
# INTERACTIVE TEST LOOP
# ==========================================================

while True:

    # Ask user a question
    query = input("\nQuestion: ")

    # Exit condition
    if query.lower() == "exit":
        break

    # Retrieve documents using hybrid search
    docs = hybrid_search(query)

    print("\nRetrieved Documents:", len(docs))

    # Display top 5 retrieved chunks
    for i, doc in enumerate(docs[:5]):

        print("\n" + "=" * 60)

        # Ranking position
        print("Rank:", i + 1)

        # Metadata such as source file/page
        print(doc.metadata)

        print()

        # Show first 300 characters
        # of retrieved chunk
        print(doc.page_content[:300])

print("\nTop Sources:\n")

for doc in docs[:5]:
    print(doc.metadata.get("source"))