

from sentence_transformers import CrossEncoder

print("Loading Reranker...")

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

print("Reranker Ready")

def rerank(query, docs, top_k=3):

    pairs = [
        [query, doc.page_content]
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for score, doc in ranked[:top_k]]