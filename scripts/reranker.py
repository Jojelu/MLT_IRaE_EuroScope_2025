
class HybridReranker:
    def __init__(self, alpha=0.5):
        """
        alpha: weight for sparse score (0 = only dense, 1 = only sparse)
        """
        self.alpha = alpha

    def merge_and_rerank(self, sparse_results, dense_results, top_k=10):
        # Build a dict for fast lookup
        sparse_dict = {str(doc["doc_id"]): doc for doc in sparse_results}
        dense_dict = {str(doc["id"]): doc for doc in dense_results}

        # Union of all doc IDs
        all_ids = set(sparse_dict.keys()) | set(dense_dict.keys())
        merged = []
        for doc_id in all_ids:
            sparse_score = sparse_dict.get(doc_id, {}).get("score", 0)
            dense_score = dense_dict.get(doc_id, {}).get("score", 0)
            # Weighted sum
            hybrid_score = self.alpha * sparse_score + (1 - self.alpha) * dense_score
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