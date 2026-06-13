
# ==========================================================
# AGENTIC RAG CHATBOT V4
# ==========================================================

import os

from langchain_ollama import ChatOllama

from hybrid_retrieval import hybrid_search
from memory import add_to_memory, get_memory
from web_search import web_search
from router import route_query

# ==========================================================
# LOAD QWEN
# ==========================================================

print("Loading Qwen 2.5...")

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

print("Qwen Ready")

# ==========================================================
# CHAT LOOP
# ==========================================================

while True:

    query = input("\nAsk Question: ")

    if query.lower() == "exit":
        break

    # ------------------------------------------------------
    # ROUTER
    # ------------------------------------------------------

    route = route_query(query)
    print("\nROUTE SELECTED =", route)
    print(f"\nRoute Selected: {route}")

    context = ""

    sources = []

    # ------------------------------------------------------
    # PDF ROUTE
    # ------------------------------------------------------

    if route == "pdf":

        docs = hybrid_search(query)

        print(f"\nRetrieved Docs: {len(docs)}")

        for doc in docs:

            context += doc.page_content + "\n\n"

            sources.append(
                f"{doc.metadata.get('source','Unknown')} "
                f"(Page {doc.metadata.get('page',0)+1})"
            )

    # ------------------------------------------------------
    # WEB ROUTE
    # ------------------------------------------------------

    else:

        results = web_search(query)

        print(f"\nWeb Results: {len(results)}")

        for result in results:

            context += f"""
Title: {result['title']}

Content:
{result['body']}
"""

            sources.append(
                result["link"]
            )

    # ------------------------------------------------------
    # MEMORY
    # ------------------------------------------------------

    history = get_memory()

    # ------------------------------------------------------
    # PROMPT
    # ------------------------------------------------------

    prompt = f"""
You are a helpful AI assistant.

Use the conversation history and context
to answer the user's question.

If the answer is not available,
say so clearly.

Conversation History:
{history}

Context:
{context}

Question:
{query}

Answer:
"""

    # ------------------------------------------------------
    # GENERATE ANSWER
    # ------------------------------------------------------

    print("\nGenerating Answer...\n")

    response = llm.invoke(prompt)

    answer = response.content

    print(answer)

    # ------------------------------------------------------
    # SAVE MEMORY
    # ------------------------------------------------------

    add_to_memory(
        query,
        answer
    )

    # ------------------------------------------------------
    # SOURCES
    # ------------------------------------------------------

    print("\nSources:\n")

    for source in sources[:5]:

        print("-", source)

    print("\n" + "="*80)