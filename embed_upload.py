# embed_upload.py
import os
import json
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_core.documents import Document
from appwrite_service import appwrite_service
import logging
import time
import torch

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_device():
    """Detect the best available device for computation"""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    else:
        device = "cpu"
        logger.info("💻 Using CPU for computation")
    return device


def embed_and_upload_chunks(url=None, batch_size=200, use_gpu=True):
    """Embed and upload document chunks to Pinecone using LangChain with GPU acceleration"""
    try:
        # Detect and configure device
        if use_gpu:
            device = detect_device()
        else:
            device = "cpu"
            logger.info("🔧 GPU disabled by user, using CPU")

        # Initialize embeddings model with GPU support
        logger.info(f"🧠 Initializing embeddings model on {device.upper()}")
        embedder = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large-v2",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Initialize Pinecone
        index_name = os.getenv("PINECONE_INDEX")

        # Load chunks from Appwrite
        logger.info("📚 Loading chunks from Appwrite...")
        chunks = appwrite_service.get_all_chunks(url)

        if not chunks:
            logger.error(
                "❌ No chunks found in Appwrite database. Please run chunk_docs.py first."
            )
            return False

        logger.info(f"📊 Loaded {len(chunks)} chunks")

        # Prepare LangChain-style documents
        formatted_docs = [
            Document(
                page_content=chunk["content"],
                metadata={
                    "url": chunk.get("url", ""),
                    "title": chunk["title"],
                    "chunk_id": chunk.get("chunk_id", ""),
                },
            )
            for chunk in chunks
        ]

        # Configure batch processing
        total_docs = len(formatted_docs)
        logger.info(f"🔄 Processing {total_docs} documents in batches of {batch_size}")
        logger.info(f"🎯 Device: {device.upper()}")

        start_time = time.time()
        successful_batches = 0
        failed_batches = 0

        for i in range(0, total_docs, batch_size):
            batch = formatted_docs[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_docs + batch_size - 1) // batch_size

            try:
                logger.info(
                    f"📤 Uploading batch {batch_num}/{total_batches} ({len(batch)} documents)"
                )

                # Use LangChain Pinecone to upload batch
                vectorstore = LangchainPinecone.from_documents(
                    documents=batch, embedding=embedder, index_name=index_name
                )

                successful_batches += 1
                logger.info(f"✅ Batch {batch_num} completed successfully")

                # Clear GPU cache if using GPU
                if device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                failed_batches += 1
                logger.error(f"❌ Batch {batch_num} failed: {str(e)}")

                # Clear GPU cache on error too
                if device == "cuda":
                    torch.cuda.empty_cache()

        # Final summary
        total_time = time.time() - start_time
        logger.info(f"🎉 Upload completed in {total_time:.2f} seconds")
        logger.info(
            f"📊 Results: {successful_batches} successful, {failed_batches} failed batches"
        )
        logger.info(
            f"📈 Average speed: {(successful_batches * batch_size)/total_time:.1f} docs/second"
        )

        # Save completion status if upload was successful
        if failed_batches == 0 and url:
            logger.info(f"💾 Saving completion status for {url}")
            success = appwrite_service.save_completion_status(url, total_docs)
            if success:
                logger.info(f"✅ Completion status saved successfully")
            else:
                logger.warning(f"⚠️  Failed to save completion status")

        return failed_batches == 0

    except Exception as e:
        logger.error(f"❌ Error uploading embeddings: {str(e)}")
        return False


def upload_embeddings_to_pinecone(url=None, batch_size=200, use_gpu=True):
    """Legacy function - kept for backward compatibility"""
    return embed_and_upload_chunks(url, batch_size, use_gpu)


if __name__ == "__main__":
    # For standalone execution, you can pass a URL as command line argument
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else None
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    use_gpu = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else True

    logger.info("🚀 Starting GPU-accelerated LangChain Pinecone upload")
    logger.info(f"🔧 Configuration: batch_size={batch_size}, use_gpu={use_gpu}")

    start_time = time.time()
    success = embed_and_upload_chunks(url, batch_size, use_gpu)
    total_time = time.time() - start_time

    logger.info(f"⏱️  Total upload time: {total_time:.2f} seconds")

    if success:
        print("✅ Embedding upload completed successfully!")
    else:
        print("❌ Embedding upload failed!")
