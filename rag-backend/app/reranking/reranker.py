from sentence_transformers import CrossEncoder

reranker_model=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_chunks(query,chunks,top_k=3):
    pairs=[(query,chunk) for chunk in chunks]

    scores=reranker_model.predict(pairs)

    ranked_results=sorted(zip(chunks,scores),key=lambda x:x[1],reverse=True)

    top_results=[(chunk,float(score)) for chunk,score in ranked_results[:top_k]]

    return top_results