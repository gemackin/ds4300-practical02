import ollama, fitz, os
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer


# Extract text from a PDF file by page
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# Split text into chunks of a given size with overlap
def split_text_into_chunks(words, chunk_size, chunk_overlap):
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = ' '.join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


# Represents a vector database instance
class Database(ABC):
    # Creates a new database
    @abstractmethod
    def __init__(self, port, **kwargs): pass


    # Helper function for setting object fields
    def config(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    

    def initialize_embedding_model(self, model):
        if model in ('nomic-embed-text'):
            self.embedding_model = model # Ollama model (not SentenceTransformer)
        else: self.embedding_model = SentenceTransformer(model)


    # Clears all stored indices in the database
    @abstractmethod
    def clear(self): pass


    # Indexes a PDF into the database
    def index(self, file):
        stem = os.path.basename(file)[:-4]
        get_metrics, store_metrics = [], []
        for page_num, text in extract_text_from_pdf(file):
            text = self.preprocess(text) # Convert text to words
            chunks = split_text_into_chunks(text, self.chunk_size, self.chunk_overlap)
            for chunk_num, chunk in enumerate(chunks):
                embedding = chunk # We allow the option to skip embedding and just pass it raw
                if self.use_embedding:
                    _get_embedding = self.track(self.get_embedding, get_metrics)
                    embedding = _get_embedding(chunk)
                _store_embedding = self.track(self.store_embedding, store_metrics)
                _store_embedding(embedding, chunk, file=stem, page=str(page_num), chunk=str(chunk_num))
        if self.use_embedding: self.log('index', 'embed', stem, *get_metrics)
        self.log('index', 'store', stem, *store_metrics)
        if hasattr(self, 'log'): print(f'Indexed "{stem}"')

    
    # Searches the database for context to a non-embedded query
    def search(self, query, k=3):
        embedding = query # We allow the option to skip embedding and just pass it raw
        if self.use_embedding:
            embedding, get_metrics = self.track(self.get_embedding)(query)
            self.log('search', 'embed', query, *get_metrics)
        response, query_metrics = self.track(self.query_embedding)(embedding, query, k)
        self.log('search', 'query', query, *query_metrics)
        if hasattr(self, 'log'): print(f'Queried "{query}"')
        return response


    # Returns a text embedding using the initialized model
    def get_embedding(self, text):
        if not isinstance(self.embedding_model, str):
            return self.embedding_model.encode(text)
        return ollama.embeddings(model=self.embedding_model, prompt=text)['embedding']
    

    # Sets up a file to log time/memory usage of the database's functions
    def initialize_logger(self, file, track, prefix=''):
        if isinstance(file, str):
            exists = os.path.isfile(file)
            file = open(file, 'a')
            if not exists: # If the file didn't exist prior, add a header
                file.write('run,process,part,argument,duration')
        self.logger, self.log_prefix, self.track = file, prefix, track

    
    # Logs the tracking metrics for a function into the log file
    def log(self, *args):
        if not hasattr(self, 'log'): return
        self.logger.write(f'\n{self.log_prefix},{",".join(map(str, args))}')


    # Indexes a single chunk of a PDF into the database
    @abstractmethod
    def store_embedding(self, embedding, **kwargs): pass


    # Queries the database for context to an embedded query
    @abstractmethod
    def query_embedding(self, embedding, k=3): pass