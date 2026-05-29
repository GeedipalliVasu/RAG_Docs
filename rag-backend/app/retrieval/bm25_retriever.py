from rank_bm25 import BM25Okapi

def create_bm25_index(chunks):
    tokenized_chunks=[chunk.split() for chunk in chunks]

    bm25=BM250kapi(tokenized_chunks)

    return bm25

def bm25_search(query,bm25,chunks,top_k=3):

    tokenized_query=query.split()

    scores=bm25.get_scores(tokenized_query)

    top_indices=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:top_k]

    results=[]

    for idx in top_indices:
        results.append(chunks[idx])

    return results