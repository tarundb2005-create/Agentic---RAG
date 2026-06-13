from langchain_ollama import ChatOllama

print("Loading model...")

llm = ChatOllama(
    model="qwen2.5:3b"
)

print("Sending prompt...")

response = llm.invoke("Who are you?")

print(response.content)