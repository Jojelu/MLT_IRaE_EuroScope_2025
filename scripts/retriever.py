from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np
from collections import defaultdict
from pprint import pprint
from chromadb import HttpClient
from indexer import SQLiteIndexer
from sentence_transformers import SentenceTransformer


class SpladeRetriever:
    def __init__(self, model_name="naver/splade-cocondenser-ensembledistil", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def encode_query(self, query: str) -> dict:
        # SPLADE encodes queries as sparse vectors (log(1+ReLU))
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Take the max over the sequence dimension (SPLADE pooling)
            sparse_vec = torch.log(1 + torch.relu(outputs.logits)).max(dim=1).values.squeeze()
        # Convert to numpy and get nonzero indices
        sparse_vec = sparse_vec.cpu().numpy()
        nonzero = np.nonzero(sparse_vec)[0]
        scores = {self.tokenizer.convert_ids_to_tokens([i])[0]: float(sparse_vec[i]) for i in nonzero}
        return scores

    def retrieve(self, query: str, sqlite_index: SQLiteIndexer, top_k=10, title_weight=2.0, content_weight=1.0):
        query_sparse = self.encode_query(query)
        # Get all terms and their IDs from the DB
        sqlite_index.cur.execute("SELECT term_id, term FROM terms")
        word_map = {row[1]: row[0] for row in sqlite_index.cur.fetchall()}
        # Build a doc score dict
        doc_scores = defaultdict(float)
        for token, q_weight in query_sparse.items():
            term_id = word_map.get(token)
            if term_id is None:
                continue
            # Get all postings for this word_id, including section
            sqlite_index.cur.execute("SELECT doc_id, section, count FROM postings WHERE term_id = ?", (term_id,))
            for doc_id, section, count in sqlite_index.cur.fetchall():
                weight = title_weight if section == "title" else content_weight
                doc_scores[doc_id] += q_weight * count * weight
        # Get top_k doc_ids
        top_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        # Fetch document titles
        results = []
        for doc_id, score in top_docs:
            sqlite_index.cur.execute("SELECT title FROM documents WHERE doc_id = ?", (doc_id,))
            row = sqlite_index.cur.fetchone()
            if row:
                results.append({"doc_id": doc_id, "title": row[0], "score": score})
        return results


class DenseRetriever:
    def __init__(self, chroma_client):
        self.chroma_client = chroma_client  

    def retrieve(self, query: str, collection, top_k=10):
        query_embedding = self.chroma_client.encode(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )
        # Format results for consistency
        docs = []
        for i in range(len(results["ids"][0])):
            docs.append({
                "doc_id": results["ids"][0][i],
                "title": results["metadatas"][0][i].get("title", ""),
                #"content": results["documents"][0][i],
                "score": -results["distances"][0][i]  # Negative distance = higher similarity
            })
        return docs


