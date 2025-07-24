import os
import time
import json
import requests
import trafilatura
from pathlib import Path
from tqdm import tqdm


NDJSON_PATH = Path("data/cleaned/repco_cleaned_chunks.ndjson")
SQLITE_PATH = Path("data/sqlite/repco_chunks.db")


# Optional SQLite setup (for indexing chunks)
USE_SQLITE = False
if USE_SQLITE:
    import sqlite3
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS chunks (
        doc_id TEXT,
        paragraph_index INTEGER,
        text TEXT,
        source_url TEXT
    )''')
    conn.commit()

def ensure_data_dirs():
    for sub in ["cleaned", "sqlite"]:
        Path(f"data/{sub}").mkdir(parents=True, exist_ok=True)


def clean_html(html_content):
    """Extract main content from HTML using Trafilatura."""
    text = trafilatura.extract(
        html_content,
        include_comments=False,
        include_tables=False,
        favor_recall=True
    )
    return text.strip() if text else ""


def chunk_text_by_paragraph(cleaned_text, doc_id, pubdate,source_url=None):
    """Split plain text into paragraph-like chunks."""
    paragraphs = [p.strip() for p in cleaned_text.split("\n\n") if p.strip()]
    chunks = []
    for idx, para in enumerate(paragraphs):
        chunks.append({
            "doc_id": doc_id,
            "paragraph_index": idx,
            "text": para,
            "pubDate": pubdate,
            "source_url": source_url,
            "format": "text/plain"
        })
    return chunks


def stream_graphql_data(url, batch_size=100, repo_dids=None):
    offset = 0
    total_count = None

    while True:
        query = f"""
        query MyQuery {{
            contentItems(
                filter: {{
                    revision: {{ repoDid: {{ in: {json.dumps(repo_dids)} }} }}
                    title: {{ containsKey: \"en\" }}
                    pubDate: {{ greaterThanOrEqualTo: \"2022-02-01\" }}
                }}
                first: {batch_size}
                offset: {offset}
                orderBy: PUB_DATE_ASC
            ) {{
                nodes {{
                    uid
                    title
                    content
                    pubDate
                    contentUrl
                    revision {{ repo {{ name }} }}
                }}
                totalCount
            }}
        }}
        """

        response = requests.post(url, json={"query": query}, headers={"Content-Type": "application/json"})
        if response.status_code != 200:
            print(f"Error: Status code {response.status_code}")
            break

        data = response.json()
        items = data.get("data", {}).get("contentItems", {}).get("nodes", [])

        if total_count is None:
            total_count = data.get("data", {}).get("contentItems", {}).get("totalCount", 0)
            print(f"Total records to fetch: {total_count}")

        for item in items:
            yield item

        offset += batch_size
        if len(items) < batch_size:
            break
        time.sleep(1)


def main():
    ensure_data_dirs()
    repo_dids = ["did:key:z6Mkf93GdfHNc1FZMvzHFFf58iSsLCZx6acZdoEUP5q1oKdB"]
    url = "https://repco.arbeit.cba.media/graphql"  
    target_lang = "en"
    with open(NDJSON_PATH, "w", encoding="utf-8") as ndjson_file:
        for item in tqdm(stream_graphql_data(url, repo_dids=repo_dids), desc="Fetching & cleaning"):
            # Extract title and content in the specified language
            doc_id = item.get("uid", "")
            pub_date = item.get("pubDate", "")
            source_url = item.get("contentUrl", {}).get(target_lang, {})
            title = item.get("title", {}).get(target_lang, {}).get("value", "")
            html_content = item.get("content", {}).get(target_lang, {}).get("value", "")

            # Track language and metadata
            language = target_lang
            format = "text/html"
            repoName = item.get("revision", {}).get("repo", {}).get("name", "")


            cleaned_text = clean_html(html_content)
            if not cleaned_text:
                continue

            paragraph_chunks = chunk_text_by_paragraph(cleaned_text, doc_id, source_url)

            for chunk in paragraph_chunks:
                json.dump(chunk, ndjson_file, ensure_ascii=False)
                ndjson_file.write("\n")

                if USE_SQLITE:
                    cur.execute("INSERT INTO chunks VALUES (?, ?, ?, ?)", (
                        chunk["doc_id"],
                        chunk["paragraph_index"],
                        chunk["text"],
                        chunk["source_url"]
                    ))

        if USE_SQLITE:
            conn.commit()
            conn.close()

    print(f"✅ Finished. Data saved to {NDJSON_PATH}")


if __name__ == "__main__":
    main()
