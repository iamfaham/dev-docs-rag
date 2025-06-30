import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json

BASE_URL = "https://react.dev/learn"
DOMAIN = "https://react.dev"

visited = set()
docs = []


def extract_links_from_main():
    res = requests.get(BASE_URL)
    soup = BeautifulSoup(res.text, "html.parser")
    links = soup.select("a[href^='/learn']")
    unique_links = set(urljoin(DOMAIN, link["href"]) for link in links)
    return list(unique_links)


def extract_content_from_url(url):
    if url in visited:
        return
    print("Crawling:", url)
    visited.add(url)

    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")

        content_div = soup.select_one("main")  # cleaner than whole body
        if not content_div:
            return

        paragraphs = content_div.find_all(["h1", "h2", "h3", "p", "pre", "code", "li"])
        text = "\n".join(
            p.get_text().strip() for p in paragraphs if p.get_text().strip()
        )

        if text:
            docs.append(
                {
                    "url": url,
                    "title": soup.title.string if soup.title else "No Title",
                    "content": text,
                }
            )
    except Exception as e:
        print("Error at:", url, str(e))


# Start crawling
for link in extract_links_from_main():
    extract_content_from_url(link)

# Save to JSON
with open("react_docs_raw.json", "w") as f:
    json.dump(docs, f, indent=2)
