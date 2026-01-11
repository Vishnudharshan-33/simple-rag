# Simple RAG System 🤖

A production-ready Retrieval-Augmented Generation (RAG) system built with Python, OpenAI, and Pinecone. This project demonstrates how to build an intelligent document Q&A system that can answer questions based on PDF documents.

## 🌟 Features

- **PDF Processing**: Extract and process text from PDF documents
- **Intelligent Chunking**: Split documents into overlapping chunks for better context retention
- **Vector Embeddings**: Generate semantic embeddings using OpenAI's embedding models
- **Vector Database**: Store and retrieve embeddings efficiently using Pinecone
- **Contextual Q&A**: Answer questions using GPT-4 with retrieved document context
- **Modular Architecture**: Clean, maintainable code structure with separation of concerns

## 🏗️ Architecture

The system operates in two phases:

### Offline Phase (Data Processing)
```
PDF → Text Extraction → Chunking → Embedding → Vector Storage
```

### Online Phase (Query Processing)
```
User Query → Query Embedding → Similarity Search → LLM Response
```

## 📋 Prerequisites

- Python 3.12+
- OpenAI API Key
- Pinecone API Key
- UV package manager (recommended) or pip

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/simplerag.git
cd simplerag
```

### 2. Install Dependencies

Using UV (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=your_index_name_here
```

### 4. Prepare Your Document

Place your PDF file in the `data/` directory:
```bash
mkdir -p data
cp your-document.pdf data/sample.pdf
```

### 5. Index Your Document (Offline Phase)

```bash
python dataprocessor.py
```

This will:
- Extract text from your PDF
- Create overlapping chunks
- Generate embeddings
- Store everything in Pinecone

### 6. Query Your Document (Online Phase)

```bash
python queryprocessor.py
```

Or use it programmatically:
```python
from queryprocessor import process_user_query

process_user_query("What is the main topic of the document?")
```

## 📁 Project Structure

```
simplerag/
├── pdfreader.py          # PDF text extraction
├── chunker.py            # Text chunking with overlap
├── embedder.py           # OpenAI embedding generation
├── vectorstore.py        # Pinecone vector operations
├── llm.py                # GPT-4 query processing
├── dataprocessor.py      # Offline pipeline orchestration
├── queryprocessor.py     # Online query pipeline
├── pyproject.toml        # Project dependencies
├── .env                  # Environment variables (create this)
├── data/                 # PDF documents directory
└── README.md             # This file
```

## 🔧 Configuration

### Chunking Parameters
Adjust in `chunker.py`:
- `chunk_size`: Size of each text chunk (default: 500 characters)
- `chunk_overlap`: Overlap between chunks (default: 100 characters)

### Retrieval Parameters
Adjust in `vectorstore.py`:
- `top_k`: Number of similar chunks to retrieve (default: 3)

### LLM Parameters
Adjust in `llm.py`:
- `model`: OpenAI model to use (default: "gpt-4o-mini")
- `temperature`: Response randomness (default: 0.4)

## 💡 How It Works

1. **Document Ingestion**: PDF documents are read and text is extracted page by page
2. **Chunking**: Text is split into overlapping chunks to maintain context
3. **Embedding**: Each chunk is converted to a vector embedding using OpenAI
4. **Storage**: Embeddings are stored in Pinecone with metadata
5. **Query Processing**: User queries are embedded and matched against stored chunks
6. **Answer Generation**: Retrieved chunks provide context for GPT-4 to generate accurate answers

## 🎯 Use Cases

- Internal documentation Q&A systems
- Research paper analysis
- Legal document search
- Customer support knowledge bases
- Educational content assistance

## 🛠️ Tech Stack

- **Language**: Python 3.12+
- **Embeddings**: OpenAI text-embedding-3-small
- **LLM**: OpenAI GPT-4o-mini
- **Vector Database**: Pinecone
- **PDF Processing**: PyPDF
- **Package Manager**: UV

## 📊 Example Usage

```python
# Index a document
from dataprocessor import run
run()

# Query the document
from queryprocessor import process_user_query

process_user_query("What are the key findings?")
process_user_query("Explain the methodology used")
process_user_query("What are the conclusions?")
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Connect

- **GitHub**: [yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your Name](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- OpenAI for embedding and LLM APIs
- Pinecone for vector database infrastructure
- The open-source community for inspiration

---

**Built with ❤️ for the AI community**

*If you find this project helpful, please consider giving it a ⭐ on GitHub!*
