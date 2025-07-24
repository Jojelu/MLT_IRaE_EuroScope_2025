import sqlite3
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from data_loader import load_data_from_json, load_data_from_url
from preprocessors import PreDataFrameProcessor, DataFrameProcessor
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import argparse


# -----------------------------
# SPLADE Encoder Class
# -----------------------------
class SpladeEncoder:
    def __init__(self, model_name="naver/splade-cocondenser-ensembledistil", device="cpu", top_k=50):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.top_k = top_k

    def encode(self, text: str) -> dict:
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(0)
            scores = torch.log(1 + F.relu(logits)).sum(0)
            topk = scores.topk(k=self.top_k)
            token_ids = topk.indices.tolist()
            weights = topk.values.tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            return dict(zip(tokens, weights))


# -----------------------------
# SQLite Indexer Class
# -----------------------------
class SQLiteIndexer:
    def __init__(self, db_path="database/splade_index.db"):
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()
        self._setup_tables()

    def _setup_tables(self):
        # Added the 'section' column to the inverted_index table
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS inverted_index (
            token TEXT,
            doc_id TEXT,
            section TEXT,
            weight REAL
        )""")
        # Updated the index to include section for potentially faster lookups by token and section
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_token_section ON inverted_index(token, section)")
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id INTEGER PRIMARY KEY,
            title TEXT,
            content TEXT
        )""")
        self.conn.commit()

    # Modified add_document to accept separate token weights for title and content
    def add_document(self, doc_id: str, title: str, content: str, title_token_weights: dict, content_token_weights: dict):
        # Insert token weights for the title
        for token, weight in title_token_weights.items():
            self.cur.execute("""
                INSERT INTO inverted_index (token, doc_id, section, weight) VALUES (?, ?, ?, ?)
            """, (token, doc_id, "title", float(weight))) # Insert with section 'title'

        # Insert token weights for the content
        for token, weight in content_token_weights.items():
             self.cur.execute("""
                INSERT INTO inverted_index (token, doc_id, section, weight) VALUES (?, ?, ?, ?)
            """, (token, doc_id, "content", float(weight))) # Insert with section 'content'


        # Save original text (assuming doc_id is unique and can be primary key)
        # If doc_id is not guaranteed to be unique, you might need to adjust the table definition
        self.cur.execute("""
            INSERT INTO documents (doc_id, title, content) VALUES (?, ?, ?)
        """, (doc_id, title, content))
        self.conn.commit()


    def close(self):
        self.conn.close()

class ChromaClient:
    def __init__(self, url, port, model_name="all-MiniLM-L6-v2", device=None):
        self.client = HttpClient(host=url, port=port)
        self.model = SentenceTransformer(model_name, device=device)

    def get_collection(self, name):
        return self.client.get_collection(name=name)
    
    def add_documents(self, collection, documents):
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
        return self.model.encode([text])[0]


def convert_splade_to_chroma(splade_db_file, chroma_url="arbeit.cba.media", chroma_port=8099):
    # Load documents from SQLite
    sqlite_indexer = SQLiteIndexer(db_file=splade_db_file)
    docs = sqlite_indexer.get_all_documents()

    # Prepare docs with 'id' field for Chroma
    for doc in docs:
        doc["id"] = str(doc["doc_id"])

    # Initialize Chroma DB client
    chroma_client = ChromaClient(url=chroma_url, port=chroma_port)
    collection = chroma_client.create_collection("documents")

    # Add documents to Chroma DB using your method
    chroma_client.add_documents(collection, docs)
    print("Documents imported to Chroma DB.")

# -----------------------------
#Create_Splade_DB
# -----------------------------

def create_splade_db(
    splade_db_file,
    stats_filename,
    english_df_filename,
    load_from_url=False,
    json_path=None,
    url=None,    
):
    if load_from_url:
        raw_data = load_data_from_url(url)
        print("Raw data loaded from URL.")
    else:
        if json_path is None:
            raise ValueError("json_path must be provided when load_from_url=False")
        raw_data = load_data_from_json(json_path)
        print("Raw data loaded from JSON file.")

    preDFdata = PreDataFrameProcessor(
        data_list=raw_data,
        title_key="title",
        content_key="content",
        value_key="value",
        language="en"
    )
    print("PreDataFrameProcessor initialized.")
    titles, contents = preDFdata.extract_monolingual_content()
    print("Monolingual content extracted.")
    plain_contents = preDFdata.html2str(contents)
    print("HTML content cleaned and converted to plain text.")
    df_en = preDFdata.create_dataframe(titles, plain_contents)
    print("DataFrame created.")

    df_processor = DataFrameProcessor(
        df=df_en,
        title_key="title",
        content_key="content",
        value_key="value",
        language="en"
    ) 
    print("DataFrameProcessor initialized with DataFrame.")  
    df_cleaned = df_processor.preprocess_df(stats_filename=stats_filename, english_df_filename=english_df_filename)
    print("After Dataframe cleaning:")
    print(df_cleaned.head(2))
    #df_cleaned["text"] = df_cleaned["title"].fillna('') + df_cleaned["content"].fillna('')
    encoder = SpladeEncoder()
    indexer = SQLiteIndexer(db_path=splade_db_file)
    
    for idx, row in df_cleaned.iterrows():
        doc_id = idx+1
        title_weight = encoder.encode(row["title"])
        content_weight = encoder.encode(row["content"])
        indexer.add_document(
            doc_id,
            row["title"],
            title_weight,
            row["content"],
            content_weight
        )
    indexer.close()
    print(f"SPLADE index created and saved to {splade_db_file}")


def main():
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--database_file', type=str, help='Path to the database file')
    parser.add_argument('--stats_file', type=str, help='Path to the statistics file')
    parser.add_argument('--english_df_file', type=str, help='Path to the English DataFrame file')

    args = parser.parse_args()

    splade_db_file = args.database_file
    stats_filename = args.stats_file
    english_df_filename = args.english_df_file

    #print(f"Creating database: {database_file}")
    #print(f"Saving statistics to: {stats_file}")
    #print(f"Saving English DataFrame to: {english_df_file}")
    #splade_db_file = "database/splade_index_update.db"
    #stats_filename = "data/splade_stats_1807.txt"
    #english_df_filename = "data/english_splade_df_1807.csv"

    
    create_splade_db(
        splade_db_file=splade_db_file,
        stats_filename=stats_filename,
        english_df_filename=english_df_filename,
        load_from_url=False,
        json_path="data/repco_raw_data.json",
        url=None
    )
if __name__ == "__main__":
    main()