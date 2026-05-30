from fastapi import APIRouter, UploadFile, File
import os

from app.ingestion.pdf_parser import extract_text_from_pdf
from app.ingestion.chunker import chunk_text
from app.retrieval.embedder import generate_embeddings
from app.retrieval.vector_store import create_faiss_index
from app.retrieval.retriever import similar_search_chunks
from app.retrieval.embedder import model
from app.retrieval.bm25_retriever import create_bm25_index, bm25_search
from app.reranking.reranker import rerank_chunks
from app.rag.generator import generate_answer

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

    bm25=create_bm25_index(chunks)

    embeddings = generate_embeddings(chunks)

    index = create_faiss_index(embeddings)

    query = "What is this document about?"

    results=similar_search_chunks(query,model,index,chunks)

    bm25_results=bm25_search(query,bm25,chunks)

    combined_results=list(set(results+bm25_results))

    reranked_results=rerank_chunks(query,combined_results)  

    top_chunks=[chunk for chunk,score in reranked_results]

    final_answer=generate_answer(query,top_chunks)

    return {
        "filename": file.filename,
        "total_chunks":len(chunks),
        "faiss_vectors":index.ntotal,
        "semantic_results":results,
        "bm25_results":bm25_results,
        "reranked_results":reranked_results,
        "query": query,
        "final_answer": final_answer
    }