from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

with open("react_docs_raw.json", "r") as f:
    raw_docs = json.load(f)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

chunked_docs = []
for doc in raw_docs:
    chunks = splitter.split_text(doc["content"])
    for i, chunk in enumerate(chunks):
        chunked_docs.append(
            {"content": chunk, "title": doc["title"], "url": doc["url"]}
        )

# Save final chunked data
with open("react_docs_chunks.json", "w") as f:
    json.dump(chunked_docs, f, indent=2)
