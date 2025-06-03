from bs4 import BeautifulSoup
from collections import defaultdict
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer

class SQLiteIndexer:
    def __init__(self, db_file: str = "index_sqlite.db") -> None:
        self._index: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._documents: dict[str, str] = {}

        self.con = sqlite3.connect(db_file)
        self.cur = self.con.cursor()

        self.initialize_tables()

    def drop_tables(self):
        self.cur.execute("DROP TABLE IF EXISTS document;")
        self.cur.execute("DROP TABLE IF EXISTS term;")
        self.cur.execute("DROP TABLE IF EXISTS posting;")
        self.cur.connection.commit()
        print("🧹 Old tables dropped.")

    def initialize_tables(self) -> None:
        self.cur.execute("CREATE TABLE IF NOT EXISTS document (doc_id INTEGER PRIMARY KEY, title TEXT UNIQUE, content TEXT, doc_len INTEGER)")
        self.cur.execute("CREATE TABLE IF NOT EXISTS term (word_id INTEGER PRIMARY KEY, word TEXT UNIQUE)")
        self.cur.execute(
            "CREATE TABLE IF NOT EXISTS posting (word_id INTEGER, doc_id INTEGER, section TEXT, count INTEGER, unique(word_id, doc_id, section))"
        )
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_word ON term (word)")
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_word_doc_section ON posting (word_id, doc_id, section)")

    def extract_tokens(self, text: str, ngram_range=(1, 2)) -> dict:
        """Return a dict of ngram:frequency for the given text."""
        vectorizer = CountVectorizer(ngram_range=ngram_range)
        if not isinstance(text, str) or not text.strip():
            return {}

        try:
            X = vectorizer.fit_transform([text])
            vocab = vectorizer.get_feature_names_out()
            counts = X.toarray().flatten()
            return dict(zip(vocab, counts))
        except Exception as e:
            print(f"Tokenization error: {e}")
            return {}


    def index(self, title: str, content: str, ngram_range=(1, 2)) -> None:
        doc_len = len(title) + len(content)
        self.cur.execute("INSERT OR IGNORE INTO document (title, content, doc_len) VALUES (?, ?, ?)",(title, content, doc_len))
        self.cur.execute("SELECT doc_id FROM document WHERE title = ?", (title,))
        doc_id = self.cur.fetchone()[0]

        for section, text in [("title", title), ("content", content)]:
            tokens_freq = self.extract_tokens(text, ngram_range=ngram_range)
            for token, freq in tokens_freq.items():
                self.cur.execute("INSERT OR IGNORE INTO term (word) VALUES (?)", (token,))
                self.cur.execute("SELECT word_id FROM term WHERE word = ?", (token,))
                word_id = self.cur.fetchone()[0]

                self.cur.execute(
                    "SELECT count FROM posting WHERE word_id = ? AND doc_id = ? AND section = ?",
                    (word_id, doc_id, section)
                )
                result = self.cur.fetchone()
                if result:
                    count = result[0] + freq
                    self.cur.execute(
                        "UPDATE posting SET count = ? WHERE word_id = ? AND doc_id = ? AND section = ?",
                        (count, word_id, doc_id, section)
                    )
                else:
                    self.cur.execute(
                        "INSERT INTO posting (word_id, doc_id, count) VALUES (?, ?, ?, ?)",
                        (word_id, doc_id, section, freq)
                        )
        self.con.commit()

    def close(self):
        self.cur.close()
        self.con.close()

class ChromaClient:
    def __init__(self, url, port, model_name="all-MiniLM-L6-v2", device=None):
        self.client = HttpClient(url, port)
        self.model = SentenceTransformer(model_name, device=device)
    
    def create_collection(self, collection_name):
        return self.client.create_collection(name=collection_name)

    def add_documents(self, collection, documents):
        embeddings = self.model.encode([doc['content'] for doc in documents], show_progress_bar=True)
        collection.add(
            documents=[doc['content'] for doc in documents],
            metadatas=[{'title': doc['title']} for doc in documents],
            ids=[doc['id'] for doc in documents],
            embeddings=embeddings
        )
    
    def encode(self, text):
        return self.model.encode([text])[0]
    