from .RedisDB import RedisDB
from .ChromaDB import ChromaDB
from .MilvusDB import MilvusDB


def create_database(name, *args, **kwargs):
    if name == 'redis':
        return RedisDB(*args, **kwargs)
    if name == 'chroma':
        return ChromaDB(*args, **kwargs)
    if name == 'milvus':
        return MilvusDB(*args, **kwargs)
    raise NotImplementedError(f'Database {name} not found')