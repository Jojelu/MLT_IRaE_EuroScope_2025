import os
import subprocess
import torch
from data_loader import load_data_from_url
from preprocessors import PreDataFrameProcessor, DataFrameProcessor
from indexer import SQLiteIndexer, ChromaClient
from retriever import DenseRetriever, SpladeRetriever
from reranker import HybridReranker

def get_least_used_gpu():
    # Query GPU memory usage
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader']
    )
    # Parse the result
    memory_used = [int(x) for x in result.decode('utf-8').strip().split('\n')]
    # Get the index of the GPU with the least memory used
    return int(min(range(len(memory_used)), key=lambda i: memory_used[i]))

# Set the most unused GPU as default
gpu_id = get_least_used_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# Now torch and transformers will use the selected GPU by default
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using GPU: {gpu_id}")

if __name__ == "__main__":

    url = "https://repco.arbeit.cba.media/graphql"
    raw_data = load_data_from_url(url)
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
    plain_content = preDFdata.html2str(contents)
    print("HTML content cleaned and converted to plain text.")
    df_en = preDFdata.create_dataframe(plain_content, titles)
    print("DataFrame created.")
    df_processor = DataFrameProcessor(
        df=df_en,
        title_key="title",
        content_key="content",
        value_key="value",
        language="en"
    ) 
    print("DataFrameProcessor initialized with DataFrame.")  
    df_cleaned = df_processor.preprocess_df(lemmatize_and_remove_stopwords=True, stats_filename="Data/stats_en.json")
    sqlite_indexer = SQLiteIndexer(db_path="indexer.db")
    print("SQLiteIndexer initialized.")
    sqlite_indexer.drop_tables()
    sqlite_indexer.create_tables()
    print("SQLiteIndexer tables created.")
    for index, row in df_cleaned.iterrows():
        sqlite_indexer.index(title=row['title'], content=row['content'], ngram_range=(1, 2))
    sqlite_indexer.close()
    print("Documents indexed in SQLiteIndexer.")
    
    chroma_client = ChromaClient(url="arbeit.cba.media", port=8099, model_name="all-MiniLM-L6-v2", device=device)
    print("ChromaClient initialized.")
    collection = chroma_client.create_collection(collection_name="documents")
    print("Chroma collection created.")
    chroma_client.add_documents(collection, df_cleaned.to_dict(orient='records'))
