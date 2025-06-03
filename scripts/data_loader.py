import requests
import time
import json

def load_data_from_url(url,batch_size=100, offset=0, total_count=None):

    data_list = []

    while True:
        query = f"""
        query MyQuery {{
        contentItems(
            filter: {{
            revision: {{
                repoDid: {{
                in: [
                    "did:key:z6Mkf93GdfHNc1FZMvzHFFf58iSsLCZx6acZdoEUP5q1oKdB"
                ]
                }}
            }}
            title: {{ containsKey: "en" }}
            pubDate: {{ greaterThanOrEqualTo: "2022-02-01" }}
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
            revision {{
                repo {{
                name
                }}
            }}
            }}
            totalCount
        }}
        }}
        """

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(url, json={"query": query}, headers=headers)
        if response.status_code != 200:
            print(f"Error: Status code {response.status_code}")
            break

        data = response.json()
        items = data.get("data", {}).get("contentItems", {}).get("nodes", [])


        if total_count is None:
            total_count = data.get("data", {}).get("contentItems", {}).get("totalCount", 0)
            print(f"Total records to fetch: {total_count}")

        for item in items:
            item["repoName"] = item.get("revision", {}).get("repo", {}).get("name")
            item.pop("revision", None)
            item["language"] = "en"
            item["format"] = "text/html"
        data_list.extend(items)
        print(f"Fetched {len(data_list)} / {total_count} records...")

        if len(items) < batch_size:
            print("✅ All records fetched.")
            break

        offset += batch_size
        time.sleep(1)
        with open("Data/repco_raw_data.json", "w", encoding="utf-8") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        return data_list

if __name__ == "__main__":
    url = "https://repco.arbeit.cba.media/graphql"
    load_data_from_url(url)