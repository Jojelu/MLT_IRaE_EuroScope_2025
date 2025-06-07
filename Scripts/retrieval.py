from retriever import SparseRetriever, DenseRetriever
from database import SpladeEncoder, ChromaClient
from set_device import set_device
from data_loader import load_data_from_json
import json
from pathlib import Path



def min_max_normalize(scores):
    min_score = min(scores)
    max_score = max(scores)
    range_s = max((max_score - min_score), 1e-6)
    return [(s - min_score) / (range_s) for s in scores]

def merge_and_rerank(sparse_results, dense_results, alpha=0.5, top_k=10):
    """
    alpha: weight for sparse score (0 = only dense, 1 = only sparse)
    """

    # Build dicts for fast lookup # 
    sparse_dict = {str(doc["doc_id"]).strip(): doc for doc in sparse_results}
    dense_dict = {str(doc["doc_id"]).strip(): doc for doc in dense_results}

    # Union of all doc IDs
    all_ids = set(sparse_dict.keys()) | set(dense_dict.keys())
    """overlap = set(sparse_dict.keys()) & set(dense_dict.keys())
    if not overlap:
        print("There is no overlap between sparse and dense results. Merging will not be effective.")
    else:
        print(f"Overlap between sparse and dense results: {overlap}")"""
    
    # Gather all scores for normalization
    sparse_scores = [sparse_dict.get(doc_id, {}).get("score", 0) for doc_id in all_ids]
    dense_scores = [dense_dict.get(doc_id, {}).get("score", 0) for doc_id in all_ids]
    
    # Normalize
    norm_sparse = dict(zip(all_ids, min_max_normalize(sparse_scores)))
    norm_dense = dict(zip(all_ids, min_max_normalize(dense_scores)))

    
    merged = []
    for doc_id in all_ids:
        sparse_score = norm_sparse[doc_id]
        dense_score = norm_dense[doc_id]
        # Weighted sum
        hybrid_score = alpha * sparse_score + (1 - alpha) * dense_score
        # Use available metadata
        title = sparse_dict.get(doc_id, {}).get("title") or dense_dict.get(doc_id, {}).get("title", "")
        #content = sparse_dict.get(doc_id, {}).get("content") or dense_dict.get(doc_id, {}).get("content", "")
        merged.append({
            "doc_id": doc_id,
            "title": title,
            #"content": content,
            "hybrid_score": hybrid_score,
            "sparse_score": sparse_score,
            "dense_score": dense_score
        })
    # Sort by hybrid score
    merged = sorted(merged, key=lambda x: x["hybrid_score"], reverse=True)[:top_k]
    return merged

def save_results_to_json(results, filename):
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=4)


def main():
    device = set_device()
    evaluation_set = load_data_from_json("Data/evaluation_set.json")
    queries = list(evaluation_set.keys()) 
    all_results_sparse = {}
    all_results_dense = {}
    all_results_hybrid = {}

    encoder = SpladeEncoder()
    sparse_retriever = SparseRetriever()
    client = ChromaClient(url="arbeit.cba.media", port=8099, device=device)
    collection = client.get_collection(name="chromadb-alex")
    dense_retriever = DenseRetriever(chroma_client=client)
    print("Starting retrieval process...")
    for query in queries:
        all_results_sparse[query] = {}
        all_results_dense[query] = {}
        all_results_hybrid[query] = {}
        query_vec = encoder.encode(query)
        top_k = 100
        print(f"Processing query: {query} with top_k={top_k}")
        
        # Sparse Retrieval #            
        sparse_results = [{"doc_id": str(doc[0]), "score": doc[1]}
        for doc in sparse_retriever.search(query_vec, top_k=top_k)
        ]
        all_results_sparse[query][str(top_k)] = sparse_results
    
    
        # Dense Retrieval #
        dense_results = dense_retriever.retrieve(query=query, collection=collection, top_k=top_k)
        all_results_dense[query][str(top_k)] = dense_results
        

        # Hybrid Retrieval (for each alpha)
        all_results_hybrid[query][str(top_k)] = {}
        for alpha in [0.1, 0.3, 0.5, 0.7]:
            print(f"  Merging results with alpha={alpha}")
            hybrid_results = merge_and_rerank(sparse_results, dense_results, alpha=alpha, top_k=top_k)
            all_results_hybrid[query][str(top_k)][str(alpha)] = hybrid_results

    save_results_to_json(all_results_sparse, "Data/results/sparse_results_v5.json")
    save_results_to_json(all_results_dense, "Data/results/dense_results_v5.json")
    save_results_to_json(all_results_hybrid, "Data/results/hybrid_results_v5.json")
    print("Results saved to JSON files.")
    sparse_retriever.close()

def test():
    device=set_device() 
    query=(input("What are you interested in?\n"))
    top_k=(int(input("How many results do you want?\n")))
    encoder = SpladeEncoder()
    sparse_retriever = SparseRetriever()
    client = ChromaClient(url="arbeit.cba.media", port=8099, device=device)
    collection = client.get_collection(name="chromadb-alex")
    dense_retriever = DenseRetriever(chroma_client=client)
    query_vec = encoder.encode(query)
    #sparse = [{"doc_id": str(doc[0]), "score": doc[1]}
        #for doc in sparse_retriever.search(query_vec, top_k=top_k)]
    dense = dense_retriever.retrieve(query=query, collection=collection, top_k=top_k)
    #hyrid = merge_and_rerank(sparse, dense, alpha=0.05, top_k=top_k)
    print(f"Results for query: {query}")
    print(f"Top {top_k} results:")
    for i, doc in enumerate(dense):
        print(f"{i+1}. {doc['title']}")


if __name__ == "__main__":  
    test()