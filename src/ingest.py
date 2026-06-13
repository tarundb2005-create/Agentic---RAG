#RAG ingestion pipeline

# Purpose:
# 1. Load all PDFs from data/pdfs
# 2. Split documents into chunks
# 3. Generate embeddings using BGE
# 4. Store embeddings in ChromaDB

#-----------------------------------------------------

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma


#-----------------------------------------------------------
#load of documents

print("\n Loading Pdf Files... \n")

all_docs = []
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
pdf_folder = os.path.join(BASE_DIR, "data")

for file in os.listdir(pdf_folder):
    if file.endswith('.pdf'):
        path = os.path.join(pdf_folder, file)


        #continues even the pdf is empty
        if os.path.getsize(path) == 0:
            print(f"Skipping empty PDF: {file}")
            continue

        # Initialize PDF loader
        loader = PyPDFLoader(path)

        # Load all pages from the PDF
        # Each page becomes a separate Document object
        docs = loader.load()

        # Add source metadata to every page
        # This helps identify which PDF the page came from
        for doc in docs:
            doc.metadata['source'] = file

        all_docs.extend(docs)

# Display total number of pages loaded from all PDFs
print("Page Loaded:",len(all_docs))

#------------------------------------------------------------------

# Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,       # Maximum size of each chunk (1000 characters)
    chunk_overlap = 200     # Overlap between consecutive chunks,helps preserve context across chunks
)

# Split all document pages into smaller chunks
chunks = text_splitter.split_documents(all_docs) 

print("Chunks = ",len(chunks))

#------------------------------------------------------

# Load embedding model
# BGE Small converts text chunks into numerical vectors
embeddingmodel = SentenceTransformerEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

#---------------------------------------------------------
print("DB Path:", DB_PATH)

import shutil

if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)

print("Old database deleted")

# Create Chroma vector database
db = Chroma.from_documents(

    chunks,
    embeddingmodel,
    persist_directory=DB_PATH
)

# Print success message after vector database creation
print("Vector Database Created")


#----------------------------------------------------------------------------

#pickle python package 
import pickle

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)