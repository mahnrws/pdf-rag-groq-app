import os
import gradio as gr
from PyPDF2 import PdfReader
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set in environment variables!")

client = Groq(api_key=groq_api_key)

def load_pdfs(pdf_paths):
    texts = []
    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
        except Exception as e:
            print(f" Error reading {pdf_path}: {e}")
    return texts

def chunk_text(texts, chunk_size=500):
    chunks = []
    for text in texts:
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    return np.array(embeddings)

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def query_rag(query, chunks, index):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, 3)
    retrieved_chunks = [chunks[i] for i in indices[0]]

    context = "\n\n".join(retrieved_chunks)
    prompt = f"Answer the following question based ONLY on the context:\n\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def process_pdfs_and_answer(filepaths, question):
    if not filepaths:
        return "Please upload at least one PDF."

    texts = load_pdfs([filepaths])
    if not texts:
        return "Could not extract any text from the PDFs."

    chunks = chunk_text(texts)
    embeddings = embed_chunks(chunks)
    index = build_index(embeddings)

    return query_rag(question, chunks, index)

with gr.Blocks() as demo:
    gr.Markdown("## PDF Questions & Answers")
    file_input = gr.File(label="Upload PDF", type="filepath", file_types=[".pdf"])
    question_input = gr.Textbox(label="Ask a question")
    output = gr.Textbox(label="Answer")

    btn = gr.Button("Get Answer")
    btn.click(fn=process_pdfs_and_answer, inputs=[file_input, question_input], outputs=output)

demo.launch()
