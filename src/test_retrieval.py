
# TEST RETRIEVAL

# Import embedding model wrapper from LangChain Community
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Import Chroma vector database integration
from langchain_community.vectorstores import Chroma

print("Embedding Model Is Loading...")

# This converts text into numerical vectors.
embedding_model = SentenceTransformerEmbeddings( model_name="BAAI/bge-small-en-v1.5")

# Display status message
print("Connecting to Chroma")

# Connect to the existing Chroma database stored on disk.
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function= embedding_model
)

# Convert the vector database into a retriever object.
retriever = db.as_retriever(
     search_kwargs={"k":5}    ## Return top 5 most similar chunks
)

# Ask the user to enter a question.
query = input('\n Ask a Question: \n')

# Perform similarity search.
results = retriever.invoke(query)

# Print heading
print("Retrived Chunks:")

# Loop through each retrieved document chunk
for i,doc in enumerate(results):

    # Print separator line
    print("="*50)

    # Display result number
    print(f"result {i+1}")

    #Display all information about the process(eg. chunks,chunks size, how many pdf)
    print('Metadata:',doc.metadata)

    # Blank line for readability
    print()

    # Print first 500 characters of the retrieved chunk
    print(doc.page_content[:500])

    # Add spacing after each result
    print("\n")