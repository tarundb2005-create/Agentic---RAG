
# ==========================================================
# CROSS ENCODER RERANKER TEST
# ==========================================================

from sentence_transformers import CrossEncoder

print("Loading Reranker Model...")

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

print("Model Loaded")

query = "What is Machine Learning?"

documents = [

    "Machine Learning is a subset of Artificial Intelligence.",

    "Table of Contents Chapter 1 Chapter 2 Chapter 3",

    "The weather is good today.",

    "Machine Learning allows systems to learn from data."

]

pairs = []

for doc in documents:

    pairs.append(
        [query, doc]
    )

scores = reranker.predict(pairs)

print("\nScores:\n")

for doc, score in zip(documents, scores):

    print(f"{score:.4f}")
    print(doc)
    print()