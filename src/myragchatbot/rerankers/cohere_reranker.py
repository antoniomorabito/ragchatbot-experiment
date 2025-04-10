import os
from typing import List, Dict
from langchain_core.documents import Document
import cohere


class CohereReranker:
    def __init__(self, threshold: float = 0.4, model: str = "rerank-english-v2.0"):
        self.client = cohere.Client(api_key=os.environ["COHERE_API_KEY"])
        self.threshold = threshold
        self.model = model

    def rerank(self, query: str, docs: List[Document]) -> List[Dict]:
        if not docs:
            return []

        # Prepare inputs for Cohere Rerank
        passages = [doc.page_content for doc in docs]

        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=passages,
            top_n=len(passages)
        )

        # Combine scores with original docs
        reranked_results = []
        for idx, res in enumerate(response.results):
            score = res.relevance_score
            if score >= self.threshold:
                reranked_results.append({
                    "relevance_score": score,
                    "document": docs[res.index]
                })

        return reranked_results
