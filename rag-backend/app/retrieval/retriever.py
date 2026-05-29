import numpy as np

def similar_search_chunks(query,model,index,chunks,top_k=3):

    query_embedding=model.encode([query])

    distances,indices=index.search(np.aeear(query_embedding),top_k)

    results=[]

    for idx in indices[0]:
        results.append(chunks[idx])

    return results    