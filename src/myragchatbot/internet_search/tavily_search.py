import os
from typing import List
from dotenv import load_dotenv
from tavily import TavilyClient

# Load environment
load_dotenv()

# Ambil API key dari .env
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# Inisialisasi client
if TAVILY_API_KEY:
    client = TavilyClient(api_key=TAVILY_API_KEY)
else:
    client = None
    print("[TAVILY]API key not found in environment!")


def run_tavily_search(query: str, max_results: int = 3) -> List[str]:
    if client is None:
        return ["Tavily API key not found."]

    try:
        print(f"[TAVILY] Searching: '{query}' (max_results={max_results})")
        results = client.search(query=query, max_results=max_results)

        snippets = []
        for result in results.get("results", []):
        
            snippet = f"{result['title']}\n{result['content']}\nSource: {result['url']}"
            snippets.append(snippet)

        print("[TAVILY] Retrieved", len(snippets), "results")
        return snippets

    except Exception as e:
        print("[TAVILY] Error during search:", e)
        return [f" Tavily search failed: {e}"]

    if client is None:
        print("[TAVILY] Skipping search because API key is missing.")
        return ["Tavily API key not found. Internet search disabled."]

    try:
        print(f"[TAVILY]Searching: '{query}' (max_results={max_results})")
        results = client.search(query=query, max_results=max_results)

        snippets = []
        for result in results.get("results", []):
            snippet = f"{result['title']}\n{result['snippet']}\nSource: {result['url']}"
            snippets.append(snippet)

        print(f"[TAVILY]Retrieved {len(snippets)} results")
        return snippets

    except Exception as e:
        print(f"[TAVILY] Error during search: {e}")
        return [f"Tavily search failed: {str(e)}"]
