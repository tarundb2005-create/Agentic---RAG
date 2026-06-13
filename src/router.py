

# ==========================================================
# AGENT ROUTER
# ==========================================================

def route_query(query):

    query = query.lower()

    web_keywords = [

        "latest",
        "today",
        "current",
        "news",
        "recent",
        "2026",
        "2027",
        "trend",
        "update"
    ]

    for keyword in web_keywords:

        if keyword in query:

            return "web"

    return "pdf"