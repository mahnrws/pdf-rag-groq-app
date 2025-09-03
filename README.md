# PDF RAG with Groq LLM

This project is an interactive **PDF Question Answering system** that leverages **Retrieval-Augmented Generation (RAG)** with **Groq LLMs**. It extracts text from uploaded PDFs, creates semantic embeddings with `sentence-transformers`, indexes them using **FAISS**, and queries the **Groq API (LLaMA 3.3 70B)** for accurate answers grounded in the retrieved context. The frontend is built with **Gradio** for a simple browser-based interface.

---

## Features

* Upload a **PDF file**
* Extract and chunk text automatically
* Perform **natural language QnA** on document content
* Uses **SentenceTransformer (all-MiniLM-L6-v2)** for embeddings
* Retrieves top relevant chunks with **FAISS vector search**
* Queries **Groq LLaMA-3.3-70B** for high-quality answers
* User-friendly interface powered by **Gradio**

---

## Requirements

* Python 3.8+
* Groq API key

Install dependencies:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
faiss-cpu
numpy
gradio
PyPDF2
sentence-transformers
groq
```

---

## Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/mahnrws/pdf-rag-groq-app.git
   cd pdf-rag-groq-app
   ```

2. Export your **Groq API key** as an environment variable:

   ```bash
   export GROQ_API_KEY="your_api_key_here"
   ```

   On Windows (PowerShell):

   ```powershell
   setx GROQ_API_KEY "your_api_key_here"
   ```

3. Run the app:

   ```bash
   python app.py
   ```

4. Open the local Gradio link in your browser.

---

## Usage

1. Upload a PDF file
2. Type your question in plain English
3. The system will:

   * Extract text from the PDF
   * Chunk and embed text with `sentence-transformers`
   * Retrieve top matches using FAISS
   * Generate an answer using Groq’s **LLaMA 3.3 70B** model

---

## Example Workflow

* Upload `research_paper.pdf`
* Ask: *"What methods were used in this study?"*
* The system retrieves the most relevant text chunks and provides a grounded answer.

---

## Project Structure

```
pdf-rag-groq-app/
│
├── doc reader.py              # Main application script
└── README.md           # Project documentation
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
