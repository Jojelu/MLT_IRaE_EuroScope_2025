from retriever import SpladeRetriever, DenseRetriever
from indexer import SQLiteIndexer, ChromaClient
from sklearn.metrics import ndcg_score
import numpy as np
from chromadb import HttpClient
from set_device import set_device


def merge_and_rerank(sparse_results, dense_results, alpha=0.5, top_k=10):
    """
    alpha: weight for sparse score (0 = only dense, 1 = only sparse)
    """

    # Build dicts for fast lookup # inserted .strip
    sparse_dict = {str(doc["doc_id"]).strip(): doc for doc in sparse_results}

    dense_dict = {str(doc["doc_id"]).strip(): doc for doc in dense_results}

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
    device = set_device()
    sparse_retriever = SpladeRetriever(model_name="naver/splade-cocondenser-ensembledistil", device=device)
    sparse_results = sparse_retriever.retrieve(
        query=query,
        sqlite_index=SQLiteIndexer(db_file="index_sqlite_v4.db"),
        top_k=10
    )
    print("\nSparse Retrieval Results:")
    for res in sparse_results:
        print(f"Doc ID: {res['doc_id']}, Score: {res['score']:.4f}, Title: {res['title']}")
    
    
    # Dense Retrieval #
    client = ChromaClient(url="arbeit.cba.media", port=8099, device=device)
    collection = client.get_collection(name="chromadb-alex")
    dense_retriever = DenseRetriever(chroma_client=client)
    dense_results = dense_retriever.retrieve(query=query, collection=collection, top_k=10)
    print("\nDense Retrieval Results:")
    for res in dense_results:
        print(f"Doc ID: {res['doc_id']}, Distanz: {res['score']:.4f}, Title: {res['title']}")
        
    hybrid_results = merge_and_rerank(sparse_results, dense_results, alpha=0.5, top_k=10)
    print("\nHybrid Retrieval Results:")
    for res in hybrid_results:
        print(f"Doc ID: {res['doc_id']}, Hybrid Score: {res['hybrid_score']:.4f}, Title: {res['title']}")

if __name__ == "__main__":
    main()