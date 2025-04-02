import ollama, os


PROMPT = 'You are a helpful AI assistant. Use the following context to answer the query \
    as accurately as possible. If the context is not relevant to the query, say "I don\'t know". \
    \n\nContext: {context}\n\nQuery: {query}\n\nAnswer:'


def list_pdfs(dname):
    return [os.path.join(dname, f) for f in os.listdir(dname) if f.endswith('.pdf')]


def generate_rag_response(model, query, context):
    def format_context(r):
        return f"From {r.get('file', 'Unknown file')} (page {r.get('page', 'Unknown page')}, \
            chunk {r.get('chunk', 'Unknown chunk')}) with similarity {float(r.get('similarity', 0)):.2f}"
    context_str = '\n'.join(list(map(format_context, context)))
    prompt_fmt = PROMPT.format(context=context_str, query=query)
    response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt_fmt}])
    return response['message']['content']