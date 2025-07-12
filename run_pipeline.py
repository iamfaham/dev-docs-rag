#!/usr/bin/env python3
"""
Complete Processing Pipeline

This script runs the entire pipeline: crawl → chunk → embed upload
All in one file with proper data passing between steps.
"""

import os
import sys
import time
import logging
from crawl_docs import crawl_documentation
from chunk_docs import chunk_and_save_docs
from embed_upload import embed_and_upload_chunks
from appwrite_service import appwrite_service

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_documentation_status(url):
    """Check current status of documentation processing"""
    print(f"\n📊 Checking status for: {url}")
    print("=" * 50)

    # Check if raw docs exist
    raw_exists = appwrite_service.docs_already_exist(url)
    print(f"📄 Raw documents: {'✅ Exists' if raw_exists else '❌ Not found'}")

    # Check if chunks exist
    chunks_exist = appwrite_service.chunks_already_exist(url)
    print(f"🔪 Chunks: {'✅ Exists' if chunks_exist else '❌ Not found'}")

    # Determine overall status
    if raw_exists and chunks_exist:
        print("\n🎉 Documentation is fully processed and ready for questions!")
        return "complete"
    elif raw_exists and not chunks_exist:
        print("\n⚠️  Raw documents exist but need to be chunked and embedded.")
        return "needs_chunking"
    else:
        print("\n❌ Documentation needs to be crawled first.")
        return "needs_crawling"


def run_crawl_step(url, force=False):
    """Run the crawl step"""
    print(f"\n🕷️  Step 1: Crawling documentation")
    print("=" * 50)
    print(f"📚 URL: {url}")

    if not force:
        # Check if already exists
        if appwrite_service.docs_already_exist(url):
            print("✅ Raw documents already exist, skipping crawl...")
            return True

    success = crawl_documentation(url)

    if success:
        print("✅ Crawl completed successfully!")
        print(f"📄 Raw documents saved to storage")
        return True
    else:
        print("❌ Crawl failed!")
        return False


def run_chunk_step(url, force=False):
    """Run the chunk step"""
    print(f"\n🔪 Step 2: Chunking documents")
    print("=" * 50)
    print(f"📚 URL: {url}")

    if not force:
        # Check if chunks already exist
        if appwrite_service.chunks_already_exist(url):
            print("✅ Chunks already exist, skipping chunking...")
            return True

    success = chunk_and_save_docs(url)

    if success:
        print("✅ Chunking completed successfully!")
        print(f"🔪 Documents chunked and saved to storage")
        return True
    else:
        print("❌ Chunking failed!")
        return False


def run_embed_step(url, force=False, batch_size=200, use_gpu=True):
    """Run the embed and upload step with GPU acceleration"""
    print(f"\n🧠 Step 3: Creating embeddings and uploading")
    print("=" * 50)
    print(f"📚 URL: {url}")
    print(f"🔧 Batch size: {batch_size}")
    print(f"🚀 GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")

    success = embed_and_upload_chunks(url, batch_size=batch_size, use_gpu=use_gpu)

    if success:
        print("✅ Embedding and upload completed successfully!")
        print(f"🧠 Embeddings created and uploaded to Pinecone")
        return True
    else:
        print("❌ Embedding and upload failed!")
        return False


def run_complete_pipeline(url, force=False, batch_size=200, use_gpu=True):
    """Run the complete pipeline: crawl → chunk → embed with GPU acceleration"""
    print("🚀 Complete Documentation Processing Pipeline")
    print("=" * 60)
    print(f"📚 Processing: {url}")
    print(f"🔄 Force reprocess: {'Yes' if force else 'No'}")
    print(f"🔧 Batch size: {batch_size}")
    print(f"🚀 GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")

    start_time = time.time()

    # Step 1: Crawl
    print(f"\n📋 Step 1: Crawling documentation...")
    if not run_crawl_step(url, force=force):
        print("❌ Pipeline failed at crawl step!")
        return False

    # Step 2: Chunk
    print(f"\n📋 Step 2: Chunking documents...")
    if not run_chunk_step(url, force=force):
        print("❌ Pipeline failed at chunk step!")
        return False

    # Step 3: Embed and upload
    print(f"\n📋 Step 3: Creating embeddings and uploading...")
    if not run_embed_step(url, force=force, batch_size=batch_size, use_gpu=use_gpu):
        print("❌ Pipeline failed at embed step!")
        return False

    total_time = time.time() - start_time
    print(f"\n🎉 Complete pipeline finished successfully!")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📈 Average speed: {total_time/3:.1f} seconds per step")

    return True


def main():
    """Main function with command line interface"""

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

    print("🚀 Documentation Processing Pipeline")
    print("=" * 60)

    # Check command line arguments
    if len(sys.argv) > 1:
        # Direct URL provided
        url = sys.argv[1]
        force = sys.argv[2].lower() == "true" if len(sys.argv) > 2 else False
        batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 200
        use_gpu = sys.argv[4].lower() == "true" if len(sys.argv) > 4 else True

        print(f"📚 URL: {url}")
        print(f"🔄 Force: {'Yes' if force else 'No'}")
        print(f"🔧 Batch size: {batch_size}")
        print(f"🚀 GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")

        # Check current status
        status = check_documentation_status(url)

        if status == "complete" and not force:
            print("\n✅ Documentation is already fully processed!")
            response = (
                input("🔄 Do you want to reprocess anyway? (y/N): ").strip().lower()
            )
            if response != "y":
                print("👋 Exiting...")
                return

        # Run pipeline
        success = run_complete_pipeline(
            url, force=force, batch_size=batch_size, use_gpu=use_gpu
        )

        if success:
            print("\n🎉 Pipeline completed successfully!")
        else:
            print("\n❌ Pipeline failed!")

        return

    # Interactive mode
    print("\n📚 Available documentation sets:")
    for key, (name, url) in docs.items():
        print(f"  {key}. {name} - {url}")

    print("\n🔧 Options:")
    print("  A. Check status only")
    print("  B. Run complete pipeline")
    print("  C. Run individual steps")
    print("  Q. Quit")

    while True:
        try:
            choice = (
                input(
                    "\n🎯 Enter your choice (1-10 for docs, A-C for actions, Q to quit): "
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

                # Check status first
                status = check_documentation_status(doc_url)

                action = (
                    input(
                        "What would you like to do? (A=status, B=pipeline, C=steps): "
                    )
                    .strip()
                    .upper()
                )

                if action == "A":
                    # Status already shown above
                    pass
                elif action == "B":
                    # Get pipeline parameters
                    force_input = input("Force reprocess? (y/N): ").strip().lower()
                    force = force_input == "y"

                    batch_size_input = input("Batch size (default 200): ").strip()
                    batch_size = int(batch_size_input) if batch_size_input else 200

                    gpu_input = input("Use GPU acceleration? (Y/n): ").strip().lower()
                    use_gpu = gpu_input != "n"

                    success = run_complete_pipeline(
                        doc_url, force=force, batch_size=batch_size, use_gpu=use_gpu
                    )

                    if success:
                        print("\n🎉 Pipeline completed successfully!")
                    else:
                        print("\n❌ Pipeline failed!")

                elif action == "C":
                    # Individual steps
                    step = input("Which step? (1=crawl, 2=chunk, 3=embed): ").strip()

                    if step == "1":
                        run_crawl_step(doc_url)
                    elif step == "2":
                        run_chunk_step(doc_url)
                    elif step == "3":
                        batch_size_input = input("Batch size (default 200): ").strip()
                        batch_size = int(batch_size_input) if batch_size_input else 200

                        gpu_input = (
                            input("Use GPU acceleration? (Y/n): ").strip().lower()
                        )
                        use_gpu = gpu_input != "n"

                        run_embed_step(doc_url, batch_size=batch_size, use_gpu=use_gpu)
                    else:
                        print("❌ Invalid step choice!")

            elif choice in ["A", "B", "C"]:
                # For these actions, we need a URL
                url = input("Enter the documentation URL: ").strip()
                if not url:
                    print("❌ URL is required!")
                    continue

                if choice == "A":
                    check_documentation_status(url)
                elif choice == "B":
                    force_input = input("Force reprocess? (y/N): ").strip().lower()
                    force = force_input == "y"

                    batch_size_input = input("Batch size (default 200): ").strip()
                    batch_size = int(batch_size_input) if batch_size_input else 200

                    gpu_input = input("Use GPU acceleration? (Y/n): ").strip().lower()
                    use_gpu = gpu_input != "n"

                    success = run_complete_pipeline(
                        url, force=force, batch_size=batch_size, use_gpu=use_gpu
                    )

                    if success:
                        print("\n🎉 Pipeline completed successfully!")
                    else:
                        print("\n❌ Pipeline failed!")
                elif choice == "C":
                    step = input("Which step? (1=crawl, 2=chunk, 3=embed): ").strip()

                    if step == "1":
                        run_crawl_step(url)
                    elif step == "2":
                        run_chunk_step(url)
                    elif step == "3":
                        batch_size_input = input("Batch size (default 200): ").strip()
                        batch_size = int(batch_size_input) if batch_size_input else 200

                        gpu_input = (
                            input("Use GPU acceleration? (Y/n): ").strip().lower()
                        )
                        use_gpu = gpu_input != "n"

                        run_embed_step(url, batch_size=batch_size, use_gpu=use_gpu)
                    else:
                        print("❌ Invalid step choice!")

            else:
                print("❌ Invalid choice! Please enter 1-10, A-C, or Q.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
