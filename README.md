# Ollama RAG Ingest and Search

## Prerequisites

- Ollama app set up ([Ollama.com](Ollama.com))
- Pull necessary models from Ollama (`ollama pull nomic-embed-test`) (`ollama pull mistral`)
- Python with Ollama, Redis-py, and Numpy installed (`pip install ollama redis numpy`)
- Redis Stack running (Docker container is fine) on port 6379.  If that port is mapped to another port in 
Docker, change the port number in the creation of the Redis client in both python files in `src`.
- Add PDFs you want the model to process into the `./data` folder

## Source Code
- `src/ingest.py` - imports and processes PDF files in `./data` folder. Embeddings and associated information 
stored in Redis-stack
- `src/search.py` - simple question answering using 

## How to Run
- Make sure `redis-stack` container is running in Docker
- Run `python src/ingest.py` to process the files
- Run `python src/search.py` to open the prompt answering. Some relevant chunks of text from the input data will be returned before the response is given (may take up to a couple minutes)
