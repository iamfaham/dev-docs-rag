#!/usr/bin/env python3
"""
Manual Documentation Processing Script

This script allows you to manually run the crawl, chunk, and embed upload processes separately.
Each step can be run independently and provides clear feedback on the status.
"""

import os
import sys
from crawl_docs import crawl_documentation
from chunk_docs import chunk_and_save_docs
from embed_upload import embed_and_upload_chunks
from appwrite_service import appwrite_service
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_documentation_status(url):
    """Check if documentation exists and show detailed status"""
    print(f"\n📊 Checking status for: {url}")
    print("=" * 50)

    # Check if raw docs exist
    raw_exists = appwrite_service.docs_already_exist(url)
    print(f"📄 Raw documents: {'✅ Exists' if raw_exists else '❌ Not found'}")

    # Check if chunks exist
    chunks_exist = appwrite_service.chunks_already_exist(url)
    print(f"🔪 Chunks: {'✅ Exists' if chunks_exist else '❌ Not found'}")

    # Check if embeddings exist (this would require checking Pinecone)
    # For now, we'll assume if chunks exist, embeddings might exist
    print(f"🧠 Embeddings: {'✅ Likely exists' if chunks_exist else '❌ Not found'}")

    if raw_exists and chunks_exist:
        print("\n🎉 Documentation is fully processed and ready for questions!")
        return True
    elif raw_exists and not chunks_exist:
        print("\n⚠️  Raw documents exist but need to be chunked and embedded.")
        return False
    else:
        print("\n❌ Documentation needs to be crawled first.")
        return False


def run_crawl_step(url):
    """Run the crawl step"""
    print(f"\n🕷️  Starting crawl for: {url}")
    print("=" * 50)

    success = crawl_documentation(url)

    if success:
        print("✅ Crawl completed successfully!")
        print(f"📄 Raw documents saved to storage")
    else:
        print("❌ Crawl failed!")

    return success


def run_chunk_step(url):
    """Run the chunk step"""
    print(f"\n🔪 Starting chunking for: {url}")
    print("=" * 50)

    success = chunk_and_save_docs(url)

    if success:
        print("✅ Chunking completed successfully!")
        print(f"🔪 Documents chunked and saved to storage")
    else:
        print("❌ Chunking failed!")

    return success


def run_embed_step(url, batch_size=200, use_gpu=True):
    """Run the embed and upload step with GPU acceleration"""
    print(f"\n🧠 Starting embedding and upload for: {url}")
    print("=" * 50)
    print(f"🔧 Batch size: {batch_size}")
    print(f"🚀 GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")

    success = embed_and_upload_chunks(url, batch_size=batch_size, use_gpu=use_gpu)

    if success:
        print("✅ Embedding and upload completed successfully!")
        print(f"🧠 Embeddings created and uploaded to Pinecone")
    else:
        print("❌ Embedding and upload failed!")

    return success


def run_full_pipeline(url):
    """Run the complete pipeline"""
    print(f"\n🚀 Running complete pipeline for: {url}")
    print("=" * 50)

    # Step 1: Crawl
    print("\n📋 Step 1: Crawling documentation...")
    if not run_crawl_step(url):
        return False

    # Step 2: Chunk
    print("\n📋 Step 2: Chunking documents...")
    if not run_chunk_step(url):
        return False

    # Step 3: Embed and upload
    print("\n📋 Step 3: Creating embeddings and uploading...")
    if not run_embed_step(url):
        return False

    print("\n🎉 Complete pipeline finished successfully!")
    return True


def main():
    """Main function with interactive menu"""
    print("🔧 Manual Documentation Processing Tool")
    print("=" * 50)

    # Predefined documentation URLs
    docs = {
        "1": ("React", "https://react.dev/learn"),
        "2": ("Python", "https://docs.python.org/3/"),
        "3": ("Node.js", "https://nodejs.org/en/docs/"),
        "4": ("Vue.js", "https://vuejs.org/guide/"),
        "5": ("Django", "https://docs.djangoproject.com/en/stable/"),
        "6": ("FastAPI", "https://fastapi.tiangolo.com/"),
        "7": ("Docker", "https://docs.docker.com/"),
        "8": ("Kubernetes", "https://kubernetes.io/docs/"),
        "9": ("MongoDB", "https://docs.mongodb.com/"),
        "10": ("PostgreSQL", "https://www.postgresql.org/docs/"),
    }

    print("\n📚 Available documentation sets:")
    for key, (name, url) in docs.items():
        print(f"  {key}. {name} - {url}")

    print("\n🔧 Processing options:")
    print("  A. Check status")
    print("  B. Run crawl only")
    print("  C. Run chunk only")
    print("  D. Run embed only")
    print("  E. Run full pipeline")
    print("  Q. Quit")

    while True:
        try:
            choice = (
                input(
                    "\n🎯 Enter your choice (1-10 for docs, A-E for actions, Q to quit): "
                )
                .strip()
                .upper()
            )

            if choice == "Q":
                print("👋 Goodbye!")
                break

            elif choice in docs:
                doc_name, doc_url = docs[choice]
                print(f"\n📖 Selected: {doc_name} - {doc_url}")

                action = (
                    input(
                        "What would you like to do? (A=status, B=crawl, C=chunk, D=embed, E=full): "
                    )
                    .strip()
                    .upper()
                )

                if action == "A":
                    check_documentation_status(doc_url)
                elif action == "B":
                    run_crawl_step(doc_url)
                elif action == "C":
                    run_chunk_step(doc_url)
                elif action == "D":
                    run_embed_step(doc_url)
                elif action == "E":
                    run_full_pipeline(doc_url)
                else:
                    print("❌ Invalid action choice!")

            elif choice in ["A", "B", "C", "D", "E"]:
                # For these actions, we need a URL
                url = input("Enter the documentation URL: ").strip()
                if not url:
                    print("❌ URL is required!")
                    continue

                if choice == "A":
                    check_documentation_status(url)
                elif choice == "B":
                    run_crawl_step(url)
                elif choice == "C":
                    run_chunk_step(url)
                elif choice == "D":
                    run_embed_step(url)
                elif choice == "E":
                    run_full_pipeline(url)

            else:
                print("❌ Invalid choice! Please enter 1-10, A-E, or Q.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
