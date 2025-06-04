from retriever import SpladeRetriever, DenseRetriever
from indexer import SQLiteIndex, ChromaClient
from sklearn.metrics import ndcg_score
import numpy as np
from chromadb import HttpClient


def merge_and_rerank(sparse_results, dense_results, alpha=0.5, top_k=10):
    """
    alpha: weight for sparse score (0 = only dense, 1 = only sparse)
    """

    # Correctly format dense results into a list of dictionaries
    # Each dictionary represents a retrieved document with its id, metadata, and distance
    formatted_dense_results = []
    # Check if dense_results is not empty and has the expected structure
    print("dense_results =", dense_results)
    if dense_results and "ids" in dense_results and dense_results["ids"] and \
        "metadatas" in dense_results and dense_results["metadatas"] and \
        "distances" in dense_results and dense_results["distances"]:

        # Iterate through the retrieved items
        # Dense results structure: {"ids": [["id1", "id2"]], "metadatas": [[{}, {}]], "distances": [[0.1, 0.2]]}
        # We iterate through the inner lists
        for id_list, metadata_list, distance_list in zip(dense_results["ids"], dense_results["metadatas"], dense_results["distances"]):
            for doc_id, metadata, distance in zip(id_list, metadata_list, distance_list):
              formatted_dense_results.append({
                "id": str(doc_id),  # Ensure ID is a string
                "metadata": metadata,
                "score": 1 - distance # Use 1 - distance as a score (higher is better)
              })

    # Print the length of formatted_dense_results here
    print("Formatted dense results:", len(formatted_dense_results))

    # Build dicts for fast lookup # inserted .strip
    sparse_dict = {str(doc["doc_id"]).strip(): doc for doc in sparse_results}
    # Corrected: Use the formatted_dense_results list to build the dense_dict
    dense_dict = {str(doc["id"]).strip(): doc for doc in formatted_dense_results}

    # Union of all doc IDs
    all_ids = set(sparse_dict.keys()) | set(dense_dict.keys())
    merged = []
    for doc_id in all_ids:
        sparse_score = sparse_dict.get(doc_id, {}).get("score", 0)
        dense_score = dense_dict.get(doc_id, {}).get("score", 0)
        # Weighted sum
        hybrid_score = alpha * sparse_score + (1 - alpha) * dense_score
        # Use available metadata
        title = sparse_dict.get(doc_id, {}).get("title") or dense_dict.get(doc_id, {}).get("title", "")
        content = sparse_dict.get(doc_id, {}).get("content") or dense_dict.get(doc_id, {}).get("content", "")
        merged.append({
            "doc_id": doc_id,
            "title": title,
            "content": content,
            "hybrid_score": hybrid_score,
            "sparse_score": sparse_score,
            "dense_score": dense_score
        })
    # Sort by hybrid score
    merged = sorted(merged, key=lambda x: x["hybrid_score"], reverse=True)[:top_k]
    return merged

# Example usage:
# relevant_ids = {"123", "456", ...}
# ndcg_sparse = compute_ndcg(sparse_results, relevant_ids, score_key="score")
# ndcg_dense = compute_ndcg(dense_results, relevant_ids, score_key="score")
# ndcg_hybrid = compute_ndcg(hybrid_results, relevant_ids, score_key="hybrid_score")


def main(query="Romania Elections"):
    sparse_retriever = SpladeRetriever(model_name="naver/splade-cocondenser-ensembledistil")
    sparse_results = sparse_retriever.retrieve(
        query=query,
        sqlite_index=SQLiteIndex(db_file="index_sqlite.db"),
        top_k=10
    )
    print("\nSparse Retrieval Results:")
    for res in sparse_results:
        print(f"Doc ID: {res['doc_id']}, Score: {res['score']:.4f}, Title: {res['title']}")
    
    
    # Dense Retrieval #
    client = ChromaClient(host="arbeit.cba.media", port=8099)
    collection = client.get_collection(name="chromadb-alex")
    dense_retriever = DenseRetriever(chroma_client=client)
    dense_results = dense_retriever.retrieve(query=query, collection=collection, top_k=10)
    for res in dense_results:
        print(f"doc_num: {res['id']}")
        print(f"Title: {res['title']}")
        print(f"Distanz: {res['score']:.3f}")
        print("-" * 80)
        hybrid_results = merge_and_rerank(sparse_results, dense_results, alpha=0.5, top_k=10)
        print("\nHybrid Retrieval Results:")
    for res in hybrid_results:
        print(f"Doc ID: {res['doc_id']}, Hybrid Score: {res['hybrid_score']:.4f}, Title: {res['title']}")

if __name__ == "__main__":
    main()