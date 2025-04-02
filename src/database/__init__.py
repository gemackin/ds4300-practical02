from .RedisDB import RedisDB
from .ChromaDB import ChromaDB


def create_database(name, *args, **kwargs):
    if name == 'redis':
        return RedisDB(*args, **kwargs)
    if name == 'chroma':
        return ChromaDB(*args, **kwargs)
    raise NotImplementedError(f'Database {name} not found')