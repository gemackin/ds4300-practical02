import chromadb
from chromadb.utils import embedding_functions as embf
from .Database import Database


INDEX_NAME = 'embedding_index'


# Represents a ChromaDB instance
class ChromaDB(Database):
    def __init__(self, port=8000, **kwargs):
        self.config(**kwargs)
        self.client = chromadb.HttpClient(host='localhost', port=port)
        self.use_embedding = True
        self.clear()
        self._create_collection()


    def clear(self):
        try: self.client.delete_collection(INDEX_NAME) # Resetting not allowed
        except: pass # Triggers if the collection doesn't exist
    

    def _create_collection(self):
        self.collection = self.client.create_collection(INDEX_NAME, embedding_function=None)
    
    
    def store_embedding(self, embedding, raw, **kwargs):
        key = f'{"_".join([f"{k}_{v}" for k, v in kwargs.items()])}'
        self.collection.add(documents=[raw], embeddings=[embedding], ids=[key], metadatas=[kwargs])


    def query_embedding(self, embedding, raw, k=3):
        results = self.collection.query(query_texts=[raw], query_embeddings=[embedding], n_results=k) # dict of lists
        results = {k: v[0] for k, v in results.items() if isinstance(v, list)} # All of the items are lists of lists?
        return [results['metadatas'][i] | {'similarity': results['distances'][i], 'text': results['documents'][i]} for i in range(k)]

        