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
    def __init__(self, port=6380, **kwargs):
        self.config(**kwargs)
        self.client = redis.Redis(host='localhost', port=port, db=0)
        self.use_embedding = True
        self.clear()
        self._create_hnsw_index()


    def _create_hnsw_index(self):
        try: self.client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
        except redis.exceptions.ResponseError: pass
        self.client.execute_command(f'FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX} SCHEMA text \
            TEXT embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}')


    def clear(self):
        self.client.flushdb()


    def store_embedding(self, embedding, raw, **kwargs):
        key = f'{DOC_PREFIX}:{"_".join([f"{k}_{v}" for k, v in kwargs.items()])}'
        embedding = np.array(embedding, dtype=np.float32).tobytes() # byte array
        self.client.hset(key, mapping={'embedding':embedding, 'text': raw, **kwargs})

    
    def query_embedding(self, embedding, raw, k=3):
        def format_result(r):
            return {'file': r.file, 'page': r.page, 'chunk': r.chunk, 'similarity': r.vector_distance, 'text': r.text}
        query_vector = np.array(embedding, dtype=np.float32).tobytes()
        q = Query('*=>[KNN 5 @embedding $vec AS vector_distance]')\
            .sort_by('vector_distance')\
            .return_fields(*'id file page chunk vector_distance text'.split())\
            .dialect(2)
        results = self.client.ft(INDEX_NAME).search(q, query_params={'vec': query_vector})
        return list(map(format_result, results.docs[:k]))