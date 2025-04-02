import redis
from redis.commands.search.query import Query
import numpy as np
from .Database import Database


VECTOR_DIM = 768
INDEX_NAME = 'embedding_index'
DOC_PREFIX = 'doc:'
DISTANCE_METRIC = 'COSINE'


# Represents a redis-stack vector database instance
class RedisDB(Database):
    def __init__(self, embedding_model, port=6380):
        self.embedding_model = embedding_model
        self.client = redis.Redis(host='localhost', port=port, db=0)
        self.INDEX_NAME = "embedding_index"
        self.DOC_PREFIX = "doc:"
        self.clear()
        self._create_hsnw_index()


    def _create_hnsw_index(self):
        try: self.client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
        except redis.exceptions.ResponseError: pass
        self.client.execute_command(f'FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX} SCHEMA text \
            TEXT embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}')


    def clear(self):
        self.client.flushdb()


    def store_embedding(self, embedding, **kwargs):
        key = f'{DOC_PREFIX}:{"_".join([f"{k}_{v}" for k, v in kwargs.items()])}'
        embedding = np.array(embedding, dtype=np.float32).tobytes() # byte array
        self.client.hset(key, mapping={'embedding':embedding, **kwargs})

    
    def query_embedding(self, embedding, k=3):
        def format_result(r):
            return {'file': r.file, 'page': r.page, 'chunk': r.chunk, 'similarity': r.vector_distance}
        query_vector = np.array(embedding, dtype=np.float32).tobytes()
        q = Query('*=>[KNN 5 @embedding $vec AS vector_distance]')\
            .sort_by('vector_distance')\
            .return_fields(*'id file page chunk vector_distance'.split())\
            .dialect(2)
        results = self.client.ft(INDEX_NAME).search(q, query_params={'vec': query_vector})
        return list(map(format_result, results.docs[:k]))