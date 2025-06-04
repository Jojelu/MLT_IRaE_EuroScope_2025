from sklearn.metrics import ndcg_score
import numpy as np

def compute_ndcg(results, relevant_ids, score_key="score", k=10):
    """
    results: list of dicts with 'doc_id' and score_key
    relevant_ids: set or list of relevant doc_ids (as str)
    score_key: 'score', 'dense_score', or 'hybrid_score'
    k: cutoff for NDCG
    """
    # Get the list of doc_ids in ranked order
    doc_ids = [str(doc["doc_id"]) for doc in results[:k]]
    # Relevance: 1 if in relevant_ids, else 0
    y_true = np.array([[1 if doc_id in relevant_ids else 0 for doc_id in doc_ids]])
    # Scores from the retrieval system
    y_score = np.array([[doc[score_key] for doc in results[:k]]])
    return ndcg_score(y_true, y_score)

