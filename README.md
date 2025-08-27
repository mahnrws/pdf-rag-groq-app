An interactive system that lets users upload PDF files and ask natural language questions about their content.
It leverages Retrieval-Augmented Generation (RAG) with FAISS, SentenceTransformers, and Groq LLMs, with a simple Gradio-based frontend.

FEATURES:
Upload a PDF and extract its text automatically.
Ask questions in plain English about the uploaded document.
Uses FAISS for efficient similarity search on text chunks.
Generates accurate answers with Groq LLMs.
Clean and minimal interface built with Gradio.

REQUIREMENTS AND LIBRARIES:
Python 3.9+
PyPDF2
 – PDF text extraction
 
SentenceTransformers
 – text embeddings
 
FAISS
 – similarity search
 
Groq API
 – LLM responses
 
Gradio
 – user interface
