# 🤖 Developer Documentation RAG Assistant

A powerful Retrieval-Augmented Generation (RAG) system that transforms any developer documentation into an intelligent Q&A assistant. Built with GPU acceleration, hybrid retrieval, and modern NLP techniques.

## 🎯 What It Does

Transform documentation websites into intelligent chatbots that can answer questions about the content. Simply provide a documentation URL (like React docs, Python docs, etc.), and the system will:

1. **Crawl** the documentation website
2. **Process** content into searchable chunks
3. **Generate** vector embeddings with GPU acceleration
4. **Store** everything in a vector database
5. **Provide** a chat interface for Q&A

## ✨ Key Features

- 🚀 **GPU Acceleration** - 3-5x faster processing with automatic CUDA detection
- 🧠 **Hybrid Retrieval** - Combines dense vectors (Pinecone) + sparse search (BM25) + re-ranking
- 📚 **Universal Support** - Works with any documentation website
- ⚡ **Smart Caching** - Processes once, use forever
- 🎨 **Clean Interface** - Modern Gradio web UI
- 🔧 **Flexible Usage** - Web interface, Python API, or Jupyter notebook

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/dev-docs-rag.git
cd dev-docs-rag
pip install -r requirements.txt
```

### 2. Set Up Environment

Create a `.env` file with your API keys:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=your_pinecone_index_name
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
APPWRITE_ENDPOINT=your_appwrite_endpoint
APPWRITE_PROJECT_ID=your_project_id
APPWRITE_API_KEY=your_api_key
APPWRITE_DATABASE_ID=your_database_id
APPWRITE_COLLECTION_ID=your_collection_id
APPWRITE_BUCKET_ID=your_bucket_id
```

### 3. Process Documentation

We have 2 options because one is easier to run on local and other one is easier to run on a server.

**Option A: Complete Pipeline (Recommended)**

```bash
python run_pipeline.py
# Follow the interactive prompts
```

**Option B: Jupyter Notebook**

```bash
jupyter notebook documentation_pipeline.ipynb
```

### 4. Start Chatting

Once processing is complete, launch the web interface to ask questions:

```bash
python app.py
# Open http://localhost:7860
```

## 🛠️ How It Works

```
Documentation URL → Crawl → Chunk → Embed → Store → Chat Interface
```

1. **Crawl**: Extracts content from documentation websites
2. **Chunk**: Splits content into manageable pieces with overlap
3. **Embed**: Generates vector embeddings (GPU-accelerated)
4. **Store**: Saves to Pinecone vector database + Appwrite storage
5. **Retrieve**: Hybrid search (vector + keyword + re-ranking)
6. **Generate**: LLM creates answers with retrieved context

## 📋 Requirements

### Services Needed

- **Pinecone** - Vector database (free tier available)
- **OpenRouter** - LLM API access (pay-per-use)
- **Appwrite** - Storage and database (free tier available)

### Hardware

- **GPU**: NVIDIA GPU recommended (falls back to CPU)
- **RAM**: 8GB+ for large documentation sets
- **Storage**: Minimal (data stored in cloud)

## 🔧 Configuration

### GPU Settings

- **Batch Size**: 200 (default), increase for better GPUs
- **Auto-detection**: Automatically uses GPU if available
- **Memory Management**: Automatic cache clearing

### Processing Options

- **Force Reprocess**: Reprocess existing documentation
- **URL Filtering**: Filter results by documentation source
- **Status Tracking**: Database-backed completion tracking

## 🔍 Advanced Usage

### Manual Step-by-Step Processing

```bash
python crawl_docs.py https://react.dev/learn
python chunk_docs.py https://react.dev/learn
python embed_upload.py https://react.dev/learn 200 true
```

### Python API

```python
from embed_upload import embed_and_upload_chunks
from rag_pipeline import process_question_with_relevance_check

# Process documentation
embed_and_upload_chunks("https://react.dev/learn", batch_size=200, use_gpu=True)

# Ask questions
answer = process_question_with_relevance_check(
    "How do React hooks work?",
    selected_url="https://react.dev/learn"
)
```

## 🏗️ Project Structure

```
dev-docs-rag/
├── app.py                          # Web chat interface
├── run_pipeline.py                 # Complete pipeline
├── documentation_pipeline.ipynb    # Jupyter notebook
├── crawl_docs.py                   # Web scraper
├── chunk_docs.py                   # Document chunking
├── embed_upload.py                 # GPU-accelerated embedding
├── rag_pipeline.py                 # RAG with hybrid retrieval
├── appwrite_service.py             # Database integration
├── manual_process.py               # Step-by-step processing
└── requirements.txt                # Dependencies
```

## 🐛 Troubleshooting

**GPU Not Working?**

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Memory Issues?**

- Reduce batch size for smaller GPUs
- Check GPU memory usage

**API Errors?**

- Verify all API keys in `.env`
- Check service quotas and limits

**Processing Failures?**

- Ensure documentation URL is accessible
- Try with `force_reprocess=True`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test with different documentation sets
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 🙏 Acknowledgments

- Built with [Gradio](https://gradio.app/) for the web interface
- Powered by [Pinecone](https://pinecone.io/) for vector storage
- Uses [OpenRouter](https://openrouter.ai/) for LLM access
- Storage provided by [Appwrite](https://appwrite.io/)

---

`Note - This system can work with other (crawlable) websites as well, but is optimised for developer documentations specifically.`

---

**⚡ Ready to get started?** Run `python run_pipeline.py` and follow the prompts!
