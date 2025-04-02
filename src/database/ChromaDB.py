import chromadb
from .Database import Database


INDEX_NAME = 'embedding_index'


# Represents a ChromaDB instance
class ChromaDB(Database):
    def __init__(self, embedding_model, port=8000):
        self.embedding_model = embedding_model
        self.client = chromadb.HttpClient(host='chroma', port=port)
        self.clear() # Creates the collection for indexing


    def clear(self):
        self.client.reset()
        self.collection = self.client.create_collection(INDEX_NAME)
    
    
    def store_embedding(self, embedding, **kwargs):
        key = f'{"_".join([f"{k}_{v}" for k, v in kwargs.items()])}'
        self.collection.add(documents=[embedding], id=[key])


    def query_embedding(self, embedding, k=3):
        return self.collection.query(query_texts=[embedding], n_results=k)