import ollama, fitz, os
from abc import ABC, abstractmethod
from ..metrics import track


# Extract text from a PDF file by page
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# Split text into chunks of a given size with overlap
def split_text_into_chunks(text, chunk_size, chunk_overlap):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = ' '.join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


# Represents a vector database instance
class Database(ABC):
    # Creates a new database
    @abstractmethod
    def __init__(self, port): pass


    # Helper function for setting object fields
    def config(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


    # Clears all stored indices in the database
    @abstractmethod
    def clear(self): pass


    # Indexes a PDF into the database
    def index(self, file):
        get_metrics, store_metrics = [], []
        for page_num, text in extract_text_from_pdf(file):
            chunks = split_text_into_chunks(text, self.chunk_size, self.chunk_overlap)
            for chunk_num, chunk in enumerate(chunks):
                _get_embedding = track(self.get_embedding, get_metrics)
                embedding = _get_embedding(chunk)
                _store_embedding = track(self.store_embedding, store_metrics)
                _store_embedding(embedding, file=file, page=str(page_num), chunk=str(chunk_num))
        self.log('index', 'embed', os.path.basename(file)[:-4], *get_metrics)
        self.log('index', 'store', os.path.basename(file)[:-4], *store_metrics)

    
    # Searches the database for context to a non-embedded query
    def search(self, query, k=3):
        embedding, get_metrics = track(self.get_embedding)(query)
        self.log('search', 'embed', query, *get_metrics)
        response, query_metrics = track(self.query_embedding)(embedding, k)
        self.log('search', 'query', query, *query_metrics)
        return response


    # Returns a text embedding using the initialized model
    def get_embedding(self, text):
        return ollama.embeddings(model=self.embedding_model, prompt=text)['embedding']
    

    # Sets up a file to log time/memory usage of the database's functions
    def initialize_logger(self, file, prefix=''):
        if isinstance(file, str):
            exists = os.path.isfile(file)
            file = open(file, 'a')
            if not exists: # If the file didn't exist prior, add a header
                file.write('run,process,part,argument,duration')
        self.log, self.log_prefix = file, prefix

    
    # Logs the tracking metrics for a function into the log file
    def log(self, *args):
        if not hasattr(self, 'log'): return
        self.log.write(f'\n{self.prefix},{",".join(map(str, args))}')


    # Indexes a single chunk of a PDF into the database
    @abstractmethod
    def store_embedding(self, embedding, **kwargs): pass


    # Queries the database for context to an embedded query
    @abstractmethod
    def query_embedding(self, embedding, k=3): pass