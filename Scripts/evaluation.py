from sklearn.metrics import ndcg_score
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
    sparse_results = load_data_from_json("Data/results/sparse_results_v5.json")
    dense_results = load_data_from_json("Data/results/dense_results_v5.json")
    hybrid_results = load_data_from_json("Data/results/hybrid_results_v5.json")
    evaluation_set = load_data_from_json("Data/evaluation_set.json")
    queries = list(evaluation_set.keys()) 

    ndcg_scores = {}

    top_ks = [10, 20, 50, 100]

    for query in queries:
        ndcg_scores[query] = {}
        # Always use the top 100 results
        sparse = sparse_results[query].get("100", [])
        dense = dense_results[query].get("100", [])
        hybrid_dict = hybrid_results[query].get("100", {})

        for k in top_ks:
            ndcg_scores[query][str(k)] = {}

            # Sparse
            ndcg_sparse = compute_ndcg(sparse[:k], evaluation_set[query], score_key="score", k=k)
            ndcg_scores[query][str(k)]['sparse'] = ndcg_sparse

            # Dense
            ndcg_dense = compute_ndcg(dense[:k], evaluation_set[query], score_key="score", k=k)
            ndcg_scores[query][str(k)]['dense'] = ndcg_dense

            # Hybrid (iterate over alphas)
            ndcg_scores[query][str(k)]['hybrid'] = {}
            for alpha in hybrid_dict:
                hybrid = hybrid_dict[alpha][:k]
                ndcg_hybrid = compute_ndcg(hybrid, evaluation_set[query], score_key="hybrid_score", k=k)
                ndcg_scores[query][str(k)]['hybrid'][alpha] = ndcg_hybrid

    # Save the NDCG scores to a JSON file
    save_results_to_json(ndcg_scores, "Data/results/ndcg_scores_v5.json")
    print("NDCG scores computed and saved to Data/results/ndcg_scores_v5.json")


if __name__ == "__main__":
    main()