
# ==========================================================
# AGENTIC RAG ENGINE
# ==========================================================

from langchain_ollama import ChatOllama

from src.hybrid_retrieval import hybrid_search
from src.memory import add_to_memory, get_memory
from src.web_search import web_search
from src.router import route_query

print("Loading Qwen...")

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

print("Qwen Ready")

def ask_agent(question):

    route = route_query(question)

    context = ""

    sources = []

    # --------------------------------------------------
    # PDF SEARCH
    # --------------------------------------------------

    if route == "pdf":

        docs = hybrid_search(question)

        for doc in docs:

            context += doc.page_content + "\n\n"

            sources.append(
                f"{doc.metadata.get('source','Unknown')} "
                f"(Page {doc.metadata.get('page',0)+1})"
            )

    # --------------------------------------------------
    # WEB SEARCH
    # --------------------------------------------------

    else:

        results = web_search(question)

        for result in results:

            context += f"""
Title: {result['title']}

Content:
{result['body']}
"""

            sources.append(
                result["link"]
            )

    history = get_memory()

    prompt = f"""
You are a helpful AI assistant.

Conversation History:
{history}

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    answer = response.content

    add_to_memory(
        question,
        answer
    )

    return answer, sources

import time

start = time.time()

# your code

end = time.time()

print(f"Total Time: {end-start:.2f} sec")