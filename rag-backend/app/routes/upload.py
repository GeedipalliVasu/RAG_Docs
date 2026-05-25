from fastapi import APIRouter, UploadFile, File
import os

from app.ingestion.pdf_parser import extract_text_from_pdf
from app.ingestion.chunker import chunk_text
from app.retrieval.embedder import generate_embeddings
from app.retrieval.vector_store import create_faiss_index

router = APIRouter()

UPLOAD_DIR = "data"

os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:

        content = await file.read()

        f.write(content)

    extracted_text = extract_text_from_pdf(file_path)

    chunks=chunk_text(extracted_text)

    embeddings = generate_embeddings(chunks)

    index = create_faiss_index(embeddings)

    return {
        "filename": file.filename,
        "text_preview": extracted_text[:1000],
        "sample_chunk": chunks[0],
        "embedding_dimension": embeddings.shape[1],
        "faiss_vectors":index.ntotal
    }