from typing import List, Dict

def precision_at_k(ranked_docs: List[Dict], k: int) -> float:
    """
    Hitung Precision@K berdasarkan hasil reranking.
    Assumes each item in ranked_docs has a 'is_relevant' boolean field.
    """
    if k == 0 or not ranked_docs:
        return 0.0

    top_k = ranked_docs[:k]
    relevant_count = sum(1 for doc in top_k if doc.get("is_relevant", False))
    return relevant_count / k

def recall_at_k(ranked_docs: List[Dict], total_relevant: int, k: int) -> float:
    """
    Hitung Recall@K
    """
    if total_relevant == 0:
        return 0.0

    top_k = ranked_docs[:k]
    retrieved_relevant = sum(1 for doc in top_k if doc.get("is_relevant", False))
    return retrieved_relevant / total_relevant

def average_precision(ranked_docs: List[Dict]) -> float:
    """
    Mean Average Precision: hitung precision setiap kali menemukan dokumen relevan.
    """
    relevant_found = 0
    precision_sum = 0.0

    for i, doc in enumerate(ranked_docs):
        if doc.get("is_relevant", False):
            relevant_found += 1
            precision = relevant_found / (i + 1)
            precision_sum += precision

    return precision_sum / relevant_found if relevant_found > 0 else 0.0

def evaluate_reranking(ranked_docs: List[Dict], k: int = 5) -> Dict:
    total_relevant = sum(1 for doc in ranked_docs if doc.get("is_relevant", False))

    return {
        "precision@k": precision_at_k(ranked_docs, k),
        "recall@k": recall_at_k(ranked_docs, total_relevant, k),
        "MAP": average_precision(ranked_docs)
    }

# ------------------
# Example dummy usage:
if __name__ == "__main__":
    example_docs = [
        {"document": "Doc 1", "relevance_score": 0.9, "is_relevant": True},
        {"document": "Doc 2", "relevance_score": 0.85, "is_relevant": False},
        {"document": "Doc 3", "relevance_score": 0.8, "is_relevant": True},
        {"document": "Doc 4", "relevance_score": 0.75, "is_relevant": False},
        {"document": "Doc 5", "relevance_score": 0.7, "is_relevant": True},
    ]

    results = evaluate_reranking(example_docs, k=3)
    for metric, value in results.items():
        print(f"{metric}: {value:.2f}")
