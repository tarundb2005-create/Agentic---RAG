
from web_search import web_search

results = web_search(
    "Latest AI trends 2026"
)

for i, result in enumerate(results):

    print("\n" + "="*50)

    print(f"Result {i+1}")

    print("Title:", result["title"])

    print("Body:", result["body"])

    print("Link:", result["link"])