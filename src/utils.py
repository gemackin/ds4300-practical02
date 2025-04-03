import ollama, os, string, re
import nltk # For finding stop words
nltk.download('stopwords')
from nltk.corpus import stopwords


PROMPT = 'You are a helpful AI assistant. Use the following context to answer the query \
as accurately as possible. If the context is not relevant to the query, say "I don\'t know". \
\n\nContext: {context}\n\nQuery: {query}\n\nAnswer:'

STOP_WORDS = set(stopwords.words('english'))


# Provide a list of the absolute paths of all PDFs in a directory
def list_pdfs(dname):
    return [os.path.join(dname, f) for f in os.listdir(dname) if f.endswith('.pdf')]


# Query the LLM for a response given the context provided by the vector database
def generate_rag_response(model, query, context):
    def format_context(r):
        return f"From {r.get('file')} (page {r.get('page')}, chunk {r.get('chunk')}) \
        with similarity {float(r.get('similarity', 0)):.2f}:\n{r.get('text')}"
    context_str = '\n\n'.join(list(map(format_context, context)))
    prompt_fmt = PROMPT.format(context=context_str, query=query)
    # print(prompt_fmt, '\n') # Debugging prompts
    response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt_fmt}])
    return response['message']['content']


# "Preprocessing" function that just splits by whitespace
# Used during testing to compare to other forms of preprocessing
def dont_preprocess_text(text):
    return text.split()


# Preprocessing text into words by restricting valid characters and removing stop words
# This went through multiple iterations during testing before we settled on this
def preprocess_text(text):
    def restrict_characters(word):
        return ''.join([x for x in word if x in string.printable and x not in ',()[]{}'])
    text = [restrict_characters(word.strip('.\'\"-`')) for word in re.split('[\s/]', text)]
    return [word for word in text if word and word not in STOP_WORDS]