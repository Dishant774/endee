from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def load_pdf(path):

    reader = PdfReader(path)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text


def split_text(text, chunk_size=500):

    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks


def create_embeddings(chunks):

    embeddings = model.encode(chunks)

    return embeddings