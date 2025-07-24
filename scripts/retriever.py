import sqlite3
import numpy as np
from collections import defaultdict

class SparseRetriever:
    def __init__(self, db_path="Databases/splade_index.db"):
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()

    def search(self, query_tokens: dict, top_k=10, title_weight=2.0, content_weight=1.0) -> list:
        doc_scores = defaultdict(float)
        for token, q_weight in query_tokens.items():
            self.cur.execute("SELECT doc_id, weight, section FROM inverted_index WHERE token = ?", (token,))
            for doc_id, d_weight, section in self.cur.fetchall():
                weight = title_weight if section == "title" else content_weight
                doc_scores[doc_id] += q_weight * d_weight * weight

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def get_document(self, doc_id: str):
        self.cur.execute("SELECT title, content FROM documents WHERE doc_id = ?", (doc_id,))
        return self.cur.fetchone()

    def close(self):
        self.conn.close()

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
