from transformers import pipeline
from fastapi import FastAPI
from rag_pipeline import load_pdf, split_text, create_embeddings
from vector_store import VectorStore
from sentence_transformers import SentenceTransformer

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")
store = VectorStore()

generator = pipeline("text-generation", model="gpt2")
@app.get("/")
def home():
    return {"message": "AI Research Assistant Running"}


@app.post("/upload")
def upload_pdf(path: str):

    text = load_pdf(path)

    chunks = split_text(text)

    embeddings = create_embeddings(chunks)

    store.add(embeddings, chunks)

    return {"message": "PDF processed successfully"}


@app.get("/ask")
def ask(question: str):

    query_embedding = model.encode(question)

    results = store.search(query_embedding)

    context = " ".join(results)

    prompt = f"Answer the question using this context: {context} Question: {question}"

    response = generator(prompt, max_length=200)

    return {"answer": response[0]["generated_text"]}