from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from appwrite_service import appwrite_service
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chunk_and_save_docs(url=None):
    """Chunk documents and save to Appwrite database"""
    try:
        # Load raw documents from Appwrite storage
        logger.info("Loading raw documents from Appwrite storage...")
        raw_docs = appwrite_service.get_raw_docs_from_storage(url)

        if not raw_docs:
            logger.error(
                "No raw documents found in Appwrite storage. Please run crawl_docs.py first to populate the storage."
            )
            return False

        logger.info(f"Loaded {len(raw_docs)} raw documents")

        # Initialize text splitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

        # Chunk the documents
        chunked_docs = []
        for doc in raw_docs:
            chunks = splitter.split_text(doc["content"])
            for i, chunk in enumerate(chunks):
                chunked_docs.append(
                    {"content": chunk, "title": doc["title"], "url": doc["url"]}
                )

        logger.info(f"Created {len(chunked_docs)} chunks")

        # Save to Appwrite storage (optimized method)
        success = appwrite_service.save_chunks(chunked_docs, url)

        if success:
            logger.info("Successfully saved chunks to Appwrite storage")
            return True
        else:
            logger.error("Failed to save chunks to Appwrite")
            return False

    except Exception as e:
        logger.error(f"Error in chunk_and_save_docs: {str(e)}")
        return False


if __name__ == "__main__":
    # Get URL from command line argument if provided
    url = None
    if len(sys.argv) > 1:
        url = sys.argv[1]
        print(f"🔪 Chunking documents for URL: {url}")
    else:
        print("🔪 Chunking documents (no URL provided, using default)")

    success = chunk_and_save_docs(url)
    if success:
        print("✅ Chunking completed successfully!")
    else:
        print("❌ Chunking failed!")
