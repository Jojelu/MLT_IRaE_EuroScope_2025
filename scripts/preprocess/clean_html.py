import sys
import json
from bs4 import BeautifulSoup

def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    paragraphs = [p.get_text(strip=True).replace(u'\xa0', ' ') for p in soup.find_all("p")]
    return "\n\n".join(paragraphs).strip()

def main(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    for item in data_list:
        raw_html = item.get("content", "")
        item["clean_text"] = clean_html(raw_html)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)

    print(f"✅ Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_repco.py input.json output.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)
