from ingestion.chunker import chunk_text
from retrieval.retriever import retrieve_neighbors


def process_query(fast: bool,):
    query = 'What are the latest neuroprosthetic news?' if fast else input('Please enter here your query. ')
    query_chunks = chunk_text(query)
    neighbors_info = retrieve_neighbors(query_chunks, fast=fast)


if __name__ == "__main__":
    process_query(
        fast=False,
    )
