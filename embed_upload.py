# embed_upload.py
import os
import json
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone

load_dotenv()

# Load data
with open("react_docs_chunks.json", "r") as f:
    docs = json.load(f)

# Init Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to index
index_name = os.getenv("PINECONE_INDEX")
if index_name not in pc.list_indexes().names():
    raise ValueError("Index not found. Please create it first in Pinecone dashboard.")

index = pc.Index(index_name)

# Embedder
embedder = HuggingFaceEmbeddings(
    model_name="intfloat/e5-large-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Prepare LangChain-style documents
from langchain.schema import Document

formatted_docs = [
    Document(
        page_content=doc["content"], metadata={"url": doc["url"], "title": doc["title"]}
    )
    for doc in docs
]

# Upload in smaller batches
batch_size = 50
total_docs = len(formatted_docs)

print(f"Uploading {total_docs} documents in batches of {batch_size}...")

for i in range(0, total_docs, batch_size):
    batch = formatted_docs[i : i + batch_size]
    print(
        f"Uploading batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} ({len(batch)} documents)"
    )

    vectorstore = LangchainPinecone.from_documents(
        documents=batch, embedding=embedder, index_name=index_name
    )

print("✅ Uploaded to Pinecone!")
