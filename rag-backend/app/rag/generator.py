import ollama

def generate_answer(query,retrieved_chunks):

    context="\n\n".join(retrieved_chunks)

    prompt=f"""
    Use the following context to answer the question.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response=ollama.chat(model="qwen2.5:1.5b",messages=[{"role":"user","content":prompt}])

    return response['message']['content']