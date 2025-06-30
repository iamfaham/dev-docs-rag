# React Documentation RAG System

A Retrieval-Augmented Generation (RAG) system that provides intelligent answers to React-related questions using the official React documentation.

## Features

- **Web Scraping**: Automatically crawls React documentation
- **Document Chunking**: Splits documentation into searchable chunks
- **Vector Embeddings**: Uses Hugging Face embeddings for semantic search
- **Pinecone Vector Store**: Stores and retrieves document embeddings
- **OpenRouter Integration**: Uses various LLM models for answer generation
- **Interactive CLI**: Command-line interface for asking questions

## Project Structure

```
dev-docs-rag/
├── crawl_docs.py          # Web scraper for React docs
├── chunk_docs.py          # Document chunking utility
├── embed_upload.py        # Embedding generation and upload
├── rag_pipeline.py        # Main RAG pipeline
├── app.py                 # Simple web interface
├── react_docs_raw.json    # Raw scraped documentation
├── react_docs_chunks.json # Chunked documentation
└── requirements.txt       # Python dependencies
```

## Setup

### 1. Environment Variables

Create a `.env` file with your API keys:

```bash
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=your_pinecone_index_name
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Initialize Vector Store

```bash
# Step 1: Crawl React documentation
python crawl_docs.py

# Step 2: Chunk the documents
python chunk_docs.py

# Step 3: Generate embeddings and upload to Pinecone
python embed_upload.py
```

## Usage

### Command Line Interface

```bash
python rag_pipeline.py
```

This will start an interactive session where you can ask questions about React.

### Web Interface

```bash
python app.py
```

Access the web interface at `http://localhost:5000`

## API Endpoints

- `GET /`: Web interface
- `POST /ask`: Ask a question (JSON: `{"question": "your question"}`)

## Models Used

- **Embeddings**: `intfloat/e5-large-v2` (Hugging Face)
- **LLM**: Configurable via OpenRouter (default: Claude 3.5 Sonnet)
- **Vector Store**: Pinecone

## Deployment

### GitHub

This repository is ready for GitHub deployment. The `.gitignore` file ensures sensitive data is not committed.

### Hugging Face Spaces

For Hugging Face Spaces deployment, additional files may be needed:

- `app.py` (already exists) - Main application
- `requirements.txt` (already exists) - Dependencies
- Environment variables should be set in Hugging Face Spaces settings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Notes

- Large JSON files (`react_docs_*.json`) are excluded from Git by default
- Environment variables are kept secure and not committed
- The system uses Pinecone for vector storage, requiring a Pinecone account
