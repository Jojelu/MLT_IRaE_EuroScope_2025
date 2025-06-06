from data_loader import load_data_from_url, load_data_from_json
from preprocessors import PreDataFrameProcessor, DataFrameProcessor
from indexer import SQLiteIndexer, ChromaClient
from chromadb import HttpClient
from set_device import set_device



def create_sqlite_db(
    sqlite_db_file,
    stats_filename,
    english_df_filename,
    title_list,
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

    sqlite_indexer = SQLiteIndexer(db_file=sqlite_db_file)
    print("SQLiteIndexer initialized.")
    sqlite_indexer.drop_tables()
    sqlite_indexer.initialize_tables()
    print("SQLiteIndexer tables created.")
    for index, row in df_cleaned.iterrows():
        sqlite_indexer.index(title=row['title'], content=row['content'], ngram_range=(1, 2), lemmatize_and_remove_stopwords=True)
    sqlite_indexer.save_docs_to_csv(title_list)
    sqlite_indexer.close()
    print("Documents indexed in SQLiteIndexer.")

def convert_sqlite_to_chroma(sqlite_db_file, chroma_url="arbeit.cba.media", chroma_port=8099):
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
    set_device()  # Set the device for GPU usage if available
    sqlite_db_file = "index_sqlite_v5.db"
    stats_filename = "Data/stats_en_v2.csv"
    english_df_filename = "Data/en_clean_df_v2.csv"
    title_list = "Data/doc_titles_v2.csv"
    create_sqlite_db(sqlite_db_file, stats_filename, english_df_filename, title_list,load_from_url=True, url="https://repco.arbeit.cba.media/graphql")

    #client = HttpClient(host="arbeit.cba.media", port=8099)
    #collection = client.get_collection(name="chromadb-alex")  

    # Example: Retrieve documents by query or ID
    #results = collection.get(ids=["123", "456"], include=["documents", "metadatas"])
    #print(results)





















