import json
from pymilvus import MilvusClient, connections, FieldSchema, CollectionSchema, DataType, Collection
from .Database import Database


VECTOR_DIM = 768
INDEX_NAME = 'embedding_index'

SCHEMA = CollectionSchema([
    FieldSchema(name='id', dtype=DataType.VARCHAR, max_length=80, is_primary=True),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
    FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name='file', dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name='page', dtype=DataType.VARCHAR, max_length=5),
    FieldSchema(name='chunk', dtype=DataType.VARCHAR, max_length=5),
])


# Represents a Milvus database instance
class MilvusDB(Database):
    def __init__(self, port=19530, **kwargs):
        self.config(**kwargs)
        connections.connect('default', host='localhost', port=str(port)) # Connecting
        self.client = MilvusClient(uri=f'http://localhost:{port}', token='root:Milvus')
        self.use_embedding = True
        self.clear()
        self.cols = 'id embedding text file page chunk'.split()
        self._create_collection()
        self._create_index()


    def clear(self):
        if self.client.has_collection(INDEX_NAME):
            self.client.drop_collection(INDEX_NAME)


    def _create_collection(self):
        self.collection = Collection(name=INDEX_NAME, schema=SCHEMA)
    

    def _create_index(self):
        index_params = {'index_type': 'IVF_FLAT', 'metric_type': 'L2', 'params': {'nlist': VECTOR_DIM}}
        self.collection.create_index(field_name='embedding', index_params=index_params)
    
    
    def store_embedding(self, embedding, raw, **kwargs):
        key = f'{"_".join([f"{k}_{v}" for k, v in kwargs.items()])}'
        embedding = list(embedding) + [0] * (VECTOR_DIM - len(embedding))
        self.collection.insert([[key], [embedding], [raw]] + [[kwargs.get(k)] for k in self.cols[3:]])


    def query_embedding(self, embedding, raw, k=3):
        self.collection.load()
        search_params = {'metric_type': 'L2', 'params': {}} # nprobe is how many docs to check
        embedding = list(embedding) + [0] * (VECTOR_DIM - len(embedding))
        results = self.collection.search([embedding], 'embedding', search_params, limit=k, output_fields=self.cols)[0]
        return [{'similarity': results.distances[i]} | {f: results[i].entity.get(f) for f in 'text file page chunk'.split()} for i in range(k)]