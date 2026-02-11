# PDF-QA-Bot 🦙

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their content using Llama 3.3-70B.

## Features

- **PDF Document Upload**: Upload any PDF file through an intuitive Streamlit interface
- **Intelligent Document Processing**: Automatically chunks and embeds documents using HuggingFace embeddings
- **Vector Storage**: Uses ChromaDB for efficient semantic search
- **Powerful LLM**: Leverages Llama 3.3-70B (70 billion parameters) via Groq for accurate answers
- **Context-Aware Responses**: Retrieves relevant document sections to provide accurate, grounded answers

## How It Works

1. **Document Upload**: User uploads a PDF file
2. **Text Extraction**: The document is loaded using UnstructuredPDFLoader
3. **Text Chunking**: Document is split into manageable chunks (2000 chars with 200 char overlap)
4. **Embedding**: Each chunk is converted to vector embeddings using HuggingFace models
5. **Vector Storage**: Embeddings are stored in ChromaDB for fast retrieval
6. **Question Answering**: When a question is asked:
   - Relevant document chunks are retrieved using semantic search
   - Context is passed to Llama 3.3-70B via Groq
   - The model generates an answer based on the document content

## Technology Stack

- **Frontend**: Streamlit
- **LLM**: Llama 3.3-70B (via Groq API)
- **Embeddings**: HuggingFace Embeddings
- **Vector Database**: ChromaDB
- **Framework**: LangChain
- **Document Processing**: Unstructured

## Prerequisites

- Python 3.8+
- Groq API key (get one at [https://console.groq.com](https://console.groq.com))

## Installation

1. Clone the repository:
git clone <your-repo-url>
cd PDF-QA-Bot
