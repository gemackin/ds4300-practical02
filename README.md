# DS4300 Practical 02

## Prerequisites

- Ollama app set up ([Ollama.com](Ollama.com))
- Python with necessary libraries installed (`pip install ollama redis numpy chromadb pymilvus`)
- Redis Stack running on port 6380 (I used the Docker container)
- ChromaDB running on port 8000 (I used the Docker container)
- Milvus running on port 19530 (I used the docker-compose file in `milvus/`)

## How to Run
First, modify the `indir` and `outdir` fields in `config.json` to reflect the folder where the input PDFs are located and where you want the output CSV files to be saved, respectively.

Then, to run the script, navigate to the repository folder in the command line and input `python src/main.py`.

## Organization of `config.json`
- `arguments`: The list of possible arguments that are used in configuring each run
  - `chunk_size`: The number of words to use in each chunk for preprocessing the index text
  - `chunk_overlap`: The number of words to overlap each chunk for preprocessing the index text
  - `llm`: The name of the large language model from Ollama to be used for generating responses
  - `embedding_model`: The name of the model from Ollama or SentenceTransformers to be used for embedding
  - `db`: The name of the vector database to use
  - `preproc`: The name of the preprocessing function to use for raw text
    - The preprocessing function must already be implemented in the Python code
    - Use argument "dont_preprocess_text" to simply use the text separated by whitespace
- `search_queries`: A list of questions to search for in the database and return responses from the LLM

## Organization of Output
- `runs.csv`: Contains a table of run IDs and the configuration arguments used in each of them, as well as their memory usage
- `responses.csv`: Contains a table of responses to queries for each run and the time it took to generate them
- `timing.csv`: Contains a table of how long it took each database to perform each step during indexing and searching
