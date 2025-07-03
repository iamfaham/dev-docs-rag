# React Documentation RAG System

A Retrieval-Augmented Generation (RAG) system that provides intelligent answers to React-related questions using the official React documentation.

## 🚀 Live Demo

**Try the live demo on Hugging Face Spaces:** [React Docs Assistant](https://huggingface.co/spaces/YOUR_USERNAME/react-docs-rag)

## Features

- **Web Scraping**: Automatically crawls React documentation
- **Document Chunking**: Splits documentation into searchable chunks
- **Vector Embeddings**: Uses Hugging Face embeddings for semantic search
- **Pinecone Vector Store**: Stores and retrieves document embeddings
- **OpenRouter Integration**: Uses various LLM models for answer generation
- **Interactive Web Interface**: Beautiful Gradio interface with send button
- **Command Line Interface**: For direct question asking

## Project Structure

```
dev-docs-rag/
├── crawl_docs.py          # Web scraper for React docs
├── chunk_docs.py          # Document chunking utility
├── embed_upload.py        # Embedding generation and upload
├── rag_pipeline.py        # Main RAG pipeline
├── app.py                 # Gradio web interface
├── requirements.txt       # Python dependencies
└── README.md             # This file
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

### Web Interface (Recommended)

```bash
python app.py
```

Access the web interface at `http://localhost:7860`

### Command Line Interface

```bash
python rag_pipeline.py
```

This will start an interactive session where you can ask questions about React.

## Deployment

### Hugging Face Spaces

This project is configured for easy deployment on Hugging Face Spaces:

1. **Fork this repository** to your GitHub account
2. **Go to [Hugging Face Spaces](https://huggingface.co/spaces)**
3. **Click "Create new Space"**
4. **Choose "Gradio" as the SDK**
5. **Connect your GitHub repository**
6. **Set environment variables** in the Space settings:
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX`
   - `OPENROUTER_API_KEY`
   - `OPENROUTER_MODEL`
7. **Deploy!** The Space will automatically build and deploy your app

### Required Files for HF Spaces

The following files are already configured for Hugging Face Spaces:

- ✅ `app.py` - Main Gradio application
- ✅ `requirements.txt` - Python dependencies
- ✅ `rag_pipeline.py` - RAG pipeline logic
- ✅ `react_docs_chunks.json` - Document chunks (included in repo)

## Models Used

- **Embeddings**: `intfloat/e5-large-v2` (Hugging Face)
- **LLM**: Configurable via OpenRouter (default: Claude 3.5 Sonnet)
- **Vector Store**: Pinecone
- **Re-ranking**: Cross-encoder/ms-marco-MiniLM-L-6-v2

## API Endpoints

- `GET /`: Web interface
- `POST /api/predict`: Gradio API endpoint

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Troubleshooting

### Gradio App Issues

#### Output Truncation

If the Gradio app is not showing complete responses:

1. **Check response length**: The app automatically truncates responses longer than 8000 characters
2. **Increase chat height**: The chat interface has been configured with a 500px height for better visibility
3. **Scroll through responses**: Long responses are scrollable within the chat interface
4. **Check browser console**: Look for any JavaScript errors that might affect display

#### Performance Issues

- **Long response times**: The RAG pipeline processes 4 sub-questions, which can take time
- **Memory usage**: Large responses may consume significant memory
- **API rate limits**: Check your OpenRouter API usage and limits

#### Gradio Compatibility Issues

If you encounter Gradio parameter errors:

1. **Invalid parameters**: The app has been updated to use only valid Gradio 5.x parameters
2. **Type warnings**: The chatbot now uses the modern "messages" format
3. **Version compatibility**: Tested with Gradio 5.35.0

### Common Solutions

1. **Restart the app**: `python app.py`
2. **Clear browser cache**: Refresh the page or clear browser cache
3. **Check environment variables**: Ensure all API keys are properly set
4. **Test with simple questions**: Try basic questions first to verify functionality

### Testing

Run the test script to verify app functionality:

```bash
python test_app.py
```

This will test the chat function and show response length information.

## License

MIT License - see LICENSE file for details.

## Notes

- Large JSON files (`react_docs_*.json`) are included for Hugging Face Spaces deployment
- Environment variables are kept secure and not committed
- The system uses Pinecone for vector storage, requiring a Pinecone account
- Responses are limited to 4000 tokens to prevent extremely long outputs
- The app includes a prominent send button and clear chat functionality
