import os
from tavily import TavilyClient
from typing import List

from dotenv import load_dotenv
load_dotenv()
client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def run_tavily_search(query: str, max_results: int = 3) -> List[str]:
    results = client.search(query=query, max_results=max_results)

    snippets = []
    for result in results["results"]:
        snippet = f"{result['title']}\n{result['snippet']}\nSource: {result['url']}"
        snippets.append(snippet)

    return snippets
