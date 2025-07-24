import trafilatura

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
            "source_url": source_url,
            "format": "text/plain"
        })
    return chunks