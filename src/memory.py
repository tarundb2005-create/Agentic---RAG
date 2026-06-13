# ==========================================================
# SIMPLE CONVERSATION MEMORY
# ==========================================================

chat_history = []

def add_to_memory(question, answer):

    chat_history.append({
        "question": question,
        "answer": answer
    })

def get_memory():

    memory_text = ""

    for item in chat_history:

        memory_text += f"""
User: {item['question']}
Assistant: {item['answer']}
"""

    return memory_text