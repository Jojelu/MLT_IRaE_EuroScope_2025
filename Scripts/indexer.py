from collections import defaultdict
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import spacy
import re
nlp = spacy.load("en_core_web_sm")

class SQLiteIndexer:
    def __init__(self, db_file: str = "index_sqlite.db") -> None:
        self._index: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._documents: dict[str, str] = {}

        self.con = sqlite3.connect(db_file)
        self.cur = self.con.cursor()

        self.initialize_tables()

    def drop_tables(self):
        self.cur.execute("DROP TABLE IF EXISTS documents;")
        self.cur.execute("DROP TABLE IF EXISTS terms;")
        self.cur.execute("DROP TABLE IF EXISTS postings;")
        self.cur.connection.commit()
        print("🧹 Old tables dropped.")

    def initialize_tables(self) -> None:
        self.cur.execute("CREATE TABLE IF NOT EXISTS documents (doc_id INTEGER PRIMARY KEY, title TEXT, content TEXT, doc_len INTEGER, UNIQUE(title, content))")
        self.cur.execute("CREATE TABLE IF NOT EXISTS terms (term_id INTEGER PRIMARY KEY, term TEXT UNIQUE)")
        self.cur.execute("""CREATE TABLE IF NOT EXISTS postings (term_id INTEGER, doc_id INTEGER, section TEXT, 
                         count INTEGER, 
                         FOREIGN KEY(term_id) REFERENCES terms(term_id),
                         FOREIGN KEY(doc_id) REFERENCES documents(doc_id),
                         UNIQUE(term_id, doc_id, section))
                    """)
        self.con.commit()
        #self.cur.execute("CREATE INDEX IF NOT EXISTS idx_word ON term (word)")
        #self.cur.execute("CREATE INDEX IF NOT EXISTS idx_word_doc_section ON posting (word_id, doc_id, section)")

    def _extract_tokens(self, text: str, ngram_range=(1, 2)) -> dict:
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

    def _preprocess_text(self, text, lemmatize_and_remove_stopwords=False):
        preprocessed_text = text.lower()
        # Remove punctuation and numbers
        preprocessed_text = re.sub(r"[^a-zA-Z\s]", " ", preprocessed_text)
        # Remove extra whitespace
        preprocessed_text = re.sub(r"\s+", " ", preprocessed_text).strip()
        # Process with spaCy
        doc = nlp(preprocessed_text)
        # Lemmatize, remove stopwords, keep alphabetic and numeric tokens, remove short tokens
        
        if lemmatize_and_remove_stopwords:
            preprocessed_text = " ".join([token.lemma_ for token in doc if not token.is_stop])
        else:
            preprocessed_text = doc.text
        return preprocessed_text
        
    def index(self, title: str, content: str, ngram_range=(1, 2), lemmatize_and_remove_stopwords=False) -> None:
        original_title = title
        original_content = content
        doc_len = len(title) + len(content)

        # Insert document if not exists
        self.cur.execute(
            "INSERT OR IGNORE INTO documents (title, content, doc_len) VALUES (?, ?, ?)",
            (original_title, original_content, doc_len)
        )
        self.cur.execute(
            "SELECT doc_id FROM documents WHERE title = ? AND content = ?",
            (original_title, original_content)
        )
        doc_id = self.cur.fetchone()[0]

        term_cache = {}
        #postings_to_insert = []
        agg_postings = defaultdict(int)

        for section, text in [("title", self._preprocess_text(title, lemmatize_and_remove_stopwords)),
                            ("content", self._preprocess_text(content, lemmatize_and_remove_stopwords))]:
            tokens_freq = self._extract_tokens(text, ngram_range=ngram_range)
            for token, freq in tokens_freq.items():
                if token not in term_cache:
                    self.cur.execute("INSERT OR IGNORE INTO terms (term) VALUES (?)", (token,))
                    term_id = self.cur.lastrowid
                    if not term_id:
                        self.cur.execute("SELECT term_id FROM terms WHERE term = ?", (token,))
                        term_id = self.cur.fetchone()[0]
                    term_cache[token] = term_id
                term_id = term_cache[token]
                #postings_to_insert.append((term_id, doc_id, section, int(freq)))
                agg_postings[(term_id, doc_id, section)] += int(freq)

        postings_to_insert = [(tid, did, sec, cnt) for (tid, did, sec), cnt in agg_postings.items()]
        def batch_insert(cursor, sql, data, batch_size=1000):
            for i in range(0, len(data), batch_size):
                cursor.executemany(sql, data[i:i+batch_size])
        if postings_to_insert:
            sql = """
                INSERT INTO postings (term_id, doc_id, section, count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(term_id, doc_id, section) DO UPDATE SET count = excluded.count
            """
            batch_insert(self.cur, sql, postings_to_insert, batch_size=1000)
            self.con.commit()
        print(f"Indexed {len(term_cache)} unique terms.")

        """for section, text in [("title", preprocessed_title), ("content", preprocessed_content)]:
            tokens_freq = self.extract_tokens(text, ngram_range=ngram_range)
            for token, freq in tokens_freq.items():
                print(f"Indexing token: {token} (freq: {freq}) in section: {section}")
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
                        "INSERT INTO posting (word_id, doc_id, section, count) VALUES (?, ?, ?, ?)",
                        (word_id, doc_id, section, freq)
                    )
        self.con.commit()"""
    
    def get_all_documents(self):
        """Fetch all documents from the document table as a list of dicts."""
        self.cur.execute("SELECT doc_id, title, content FROM document")
        rows = self.cur.fetchall()
        docs = []
        for row in rows:
            docs.append({
                "doc_id": row[0],
                "title": row[1],
                "content": row[2]
            })
        return docs

    def save_docs_to_csv(self, filename):
        self.cur.execute("SELECT doc_id, title, doc_len FROM documents")
        rows = self.cur.fetchall()
        columns = [desc[0] for desc in self.cur.description]
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} documents to {filename}")

    def close(self):
        self.cur.close()
        self.con.close()




"""class ChromaClient:
    def __init__(self, url, port, model_name="all-MiniLM-L6-v2", device=None):
        self.client = HttpClient(url, port)
        self.model = SentenceTransformer(model_name, device=device)
    
    def create_collection(self, collection_name):
        return self.client.create_collection(name=collection_name)
    
    def get_collection(self, collection_name):
        return self.client.get_collection(name=collection_name)

    def add_documents(self, collection, documents):
         # Ensure each document has an 'id' field as string
        for doc in documents:
            doc["id"] = str(doc["doc_id"])
        embeddings = self.model.encode([doc['content'] for doc in documents], show_progress_bar=True)
        collection.add(
            documents=[doc['content'] for doc in documents],
            metadatas=[{'title': doc['title']} for doc in documents],
            ids=[doc['id'] for doc in documents],
            embeddings=embeddings
        )
    
    def encode(self, text):
        return self.model.encode([text])[0]"""


def main():
    clean_file = "Data/en_clean_df.csv"
    english_df = pd.read_csv(clean_file)
    sqlite_indexer = SQLiteIndexer(db_file="index_sqlite_v4.db")
    print("SQLiteIndexer initialized.")
    sqlite_indexer.drop_tables()
    sqlite_indexer.initialize_tables()
    print("SQLiteIndexer tables created.")
    for _, row in english_df.iterrows():
        sqlite_indexer.index(title=row['title'], content=row['content'], ngram_range=(1, 2), lemmatize_and_remove_stopwords=True)
    title_list="Data/documents_en.csv"
    sqlite_indexer.save_docs_to_csv(title_list)
    sqlite_indexer.close()
    print("Documents indexed in SQLiteIndexer.")
    
if __name__ == "__main__":
    main()