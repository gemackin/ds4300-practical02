import json
from itertools import product
from database import create_database
from metrics import track, track_memory
from utils import *


# Performs a single run given a specific configuration
def perform_run(run_no, log_file, response_file, db, llm, embedding_model,
                chunk_size, chunk_overlap, queries, preproc, **kws):
    print('='*50, 'PERFORMING RUN', run_no, '='*50)
    # Initializing and configuring the database
    db = create_database(db)
    db.initialize_embedding_model(embedding_model)
    db.config(chunk_size=chunk_size, chunk_overlap=chunk_overlap, preprocess=globals()[preproc])
    db.initialize_logger(log_file, track, run_no) # Allow for logging metrics
    # Indexing the PDFs into the database
    for fname in list_pdfs(config['indir']):
        db.index(fname)
    # Searching the database and generating responses
    for query in queries:
        context = db.search(query)
        response, metrics = track(generate_rag_response)(llm, query, context)
        response = response.replace('"', '\\"') # Formatting for entry into CSV file
        query = query.replace('"', '\\"') # Formatting for entry into CSV file
        response_file.write(f'\n{run_no},"{query}","{response.strip()}",{",".join(map(str, metrics))}')
    return db # Returning the database; important to delete *after* we check memory usage


# Returns the Cartesian product of a dictionary of lists
# Used for getting all combinations of configuration arguments
def argument_combinations(d):
    d = {k: v or [None] for k, v in d.items()} # Fixing empty lists
    d = [[(k, x) for x in v] for k, v in d.items()]
    return list(map(dict, product(*d)))


# Clears all of the databases to be used in the project
# Should clear up some memory one of the databases has data
def clear_databases():
    for db_name in config['arguments']['db']:
        create_database(db_name).clear()


# Performs all runs and saves logs and metrics to CSV files
def main():
    def open_file(fname, cols=None):
        file = open(os.path.join(config['outdir'], fname), 'w')
        if cols: file.write(cols if isinstance(cols, str) else ','.join(map(str, cols)))
        return file
    if not os.path.isdir(config['outdir']): os.mkdir(config['outdir'])
    run_cols = 'db llm embedding_model chunk_size chunk_overlap'.split()
    run_file = open_file('runs.csv', ['run', *run_cols, *'memory_rss memory_vms memory_peak_wset'.split()])
    log_file = open_file('timing.csv', 'run,process,part,argument,duration')
    response_file = open_file('responses.csv', 'run,query,response,duration')
    for i, kwargs in enumerate(argument_combinations(config['arguments'])):
        db, mem = track_memory(perform_run)(i+1, log_file, response_file,
                                        queries=config['search_queries'], **kwargs)
        db.clear(); del db # Clear the database to save on memory for future runs
        run_vals = [i+1, *map(kwargs.get, run_cols), *mem] # Values to be written
        run_file.write('\n' + ','.join(map(str, run_vals)))
    run_file.close(); log_file.close(); response_file.close()


if __name__ == '__main__':
    with open('config.json') as file:
        config = json.load(file)
    clear_databases() # Clearing memory beforehand
    main()