import os

from data_loader import load_data_from_json
from preprocessors import PreDataFrameProcessor, DataFrameProcessor
from indexer import SQLiteIndexer, ChromaClient
from chromadb import HttpClient
from set_device import get_least_used_gpu
gpu_id = get_least_used_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_sqlite_db(
    json_path="Data/repco_raw_data.json",
    stats_filename="Data/stats_en.json",
    english_df_filename="Data/en_clean_df.csv",
    sqlite_db_file="index_sqlite.db"
):
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
    print("After DataFrameProcessor.preprocess_df:")
    print(df_cleaned.head(2))

    sqlite_indexer = SQLiteIndexer(db_file=sqlite_db_file)
    print("SQLiteIndexer initialized.")
    sqlite_indexer.drop_tables()
    sqlite_indexer.initialize_tables()
    print("SQLiteIndexer tables created.")
    for index, row in df_cleaned.iterrows():
        sqlite_indexer.index(title=row['title'], content=row['content'], ngram_range=(1, 2), lemmatize_and_remove_stopwords=True)
    sqlite_indexer.save_docs_to_csv(filename="Data/documents_en.csv")
    sqlite_indexer.close()
    print("Documents indexed in SQLiteIndexer.")

def convert_sqlite_to_chroma(sqlite_db_file="index_sqlite.db", chroma_url="arbeit.cba.media", chroma_port=8099):
    # Load documents from SQLite
    sqlite_indexer = SQLiteIndexer(db_file=sqlite_db_file)
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

if __name__ == "__main__":
    #create_sqlite_db()
    #convert_sqlite_to_chroma()  # Uncomment to run conversion after SQL DB creation
    client = HttpClient(host="arbeit.cba.media", port=8099)
    collection = client.get_collection(name="chromadb-alex")  

    # Example: Retrieve documents by query or ID
    results = collection.get(ids=["123", "456"], include=["documents", "metadatas"])
    print(results)





















"""raw_data = load_data_from_json("Data/repco_raw_data.json")
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
print("Sample titles:", titles[:2])

plain_content = preDFdata.html2str(contents)
print("HTML content cleaned and converted to plain text.")
print("Sample plain content:", plain_content[:2])

df_en = preDFdata.create_dataframe(titles, plain_content)
print("DataFrame created.")
print(df_en.head(2))

df_processor = DataFrameProcessor(
    df=df_en,
    title_key="title",
    content_key="content",
    value_key="value",
    language="en"
) 
print("DataFrameProcessor initialized with DataFrame.")  
df_cleaned = df_processor.preprocess_df(stats_filename="Data/stats_en.json", english_df_filename="Data/en_clean_df.csv")
print("After DataFrameProcessor.preprocess_df:")
print(df_cleaned.head(2))
sqlite_indexer = SQLiteIndexer()
print("SQLiteIndexer initialized.")
sqlite_indexer.drop_tables()
sqlite_indexer.initialize_tables()
print("SQLiteIndexer tables created.")
for index, row in df_cleaned.iterrows():
    sqlite_indexer.index(title=row['title'], content=row['content'], ngram_range=(1, 2), lemmatize_and_remove_stopwords=True)
sqlite_indexer.save_docs_to_csv(filename="Data/documents_en.csv")
sqlite_indexer.close()
print("Documents indexed in SQLiteIndexer.")"""

