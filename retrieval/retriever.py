import numpy as np
from ingestion.embedder import embed_text
from ingestion.chunker import chunk_text
from retrieval.vector_store import MedLitRagIndex


def retrieve_neighbors(query: str, fast: bool = False) -> list[list[dict]]:
    """
    Performs nearest-neighbor search over an existing index to find elements similar to the input query, and returns
    each neighbors metadata.
    :param query: User's input query.
    :param fast: If True, loads small version of index (for faster iteration).
    :return: List of metadata of vectors similar to the input query.
    """
    # Chunk query text and embed them
    query_chunks = chunk_text(query)
    embeddings = np.array([embed_text(chunk).numpy() for chunk in query_chunks])

    # Load index and metadata
    index = MedLitRagIndex()
    idx_fname = 'index_small' if fast else None  # None loads the standard
    metadata_fname = 'metadata_small.json' if fast else None  # None loads the standard
    index.load(idx_fname=idx_fname, metadata_fname=metadata_fname)

    # Find k most similar neighbors of each query's embedding
    _, neigh_idxs = index.search(embeddings, k=20)

    # Retrieve and return neighbors metadata
    neigh_idxs = neigh_idxs.tolist()  # convert to list
    return [[index.metadata[neigh] for neigh in query_chunk_neighbors] for query_chunk_neighbors in neigh_idxs]
