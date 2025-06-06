from sklearn.metrics import ndcg_score
import json
import numpy as np
from data_loader import load_data_from_json
from retrieval import save_results_to_json

def compute_ndcg(results, evaluation_set, score_key="score", k=10):
    """
    results: list of dicts with 'doc_id' and score_key
    relevant_ids: set or list of relevant doc_ids (as str)
    score_key: 'score', 'dense_score', or 'hybrid_score'
    k: cutoff for NDCG
    """
    # Get the list of doc_ids in ranked order
    doc_ids = [doc["doc_id"] for doc in results[:k]]
    # Relevance: 1 if in relevant_ids, else 0
    y_true = np.array([[1 if doc_id in evaluation_set else 0 for doc_id in doc_ids]])
    # Scores from the retrieval system
    y_score = np.array([[doc[score_key] for doc in results[:k]]])
    return ndcg_score(y_true, y_score)


def main():
    sparse_results = load_data_from_json(f"Data/results/sparse_results.json")
    dense_results = load_data_from_json(f"Data/results/dense_results.json")
    hybrid_results = load_data_from_json(f"Data/results/hybrid_results.json")
    evaluation_set = load_data_from_json("Data/evaluation_set.json")
    queries = list(evaluation_set.keys()) 

    ndcg_scores = {}

    for query in queries:
        ndcg_scores[query] = {}
        for top_k in sparse_results[query].keys():
            top_k_int = int(top_k)
            ndcg_scores[query][top_k] = {}

            # Sparse
            sparse = sparse_results[query].get(str(top_k), [])
            ndcg_sparse = compute_ndcg(sparse, evaluation_set[query], score_key="score", k=top_k_int)
            ndcg_scores[query][top_k]['sparse'] = ndcg_sparse

            # Dense
            dense = dense_results[query].get(str(top_k), [])
            ndcg_dense = compute_ndcg(dense, evaluation_set[query], score_key="score", k=top_k_int)
            ndcg_scores[query][top_k]['dense'] = ndcg_dense

            # Hybrid (iterate over alphas)
            ndcg_scores[query][top_k]['hybrid'] = {}
            for alpha in hybrid_results[query][str(top_k)]:
                hybrid = hybrid_results[query][str(top_k)][alpha]
                ndcg_hybrid = compute_ndcg(hybrid, evaluation_set[query], score_key="hybrid_score", k=top_k_int)
                ndcg_scores[query][top_k]['hybrid'][alpha] = ndcg_hybrid
    # Save the NDCG scores to a JSON file
    save_results_to_json(ndcg_scores, "Data/results/ndcg_scores.json")
    print("NDCG scores computed and saved to Data/results/ndcg_scores.json")

if __name__ == "__main__":
    main()









           