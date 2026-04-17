import numpy as np
from ingestion.embedder import embed_text
from ingestion.chunker import chunk_text
from retrieval.vector_store import MedLitRagIndex


def retrieve_neighbors(query: str, fast: bool = False) -> list[list[dict]]:
    """
    Performs nearest-neighbor search over an existing index to find elements similar to the input query, and returns
    each neighbors information (metadata, similarity score, index position).
    :param query: User's input query.
    :param fast: If True, loads small version of index and less neighbors are returned (for faster iteration).
    :return: Lists of neighbors information per query chunk.
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
    k = 7 if fast else 20
    neigh_scores, neigh_idxs = index.search(embeddings, k=k)
    neigh_idxs = neigh_idxs.tolist()  # convert to list
    neigh_scores = neigh_scores.tolist()  # convert to list

    # Create result list: one list of neighbor information dicts per query chunk
    neigh_info = []
    for query_chunk_neigh_ids, query_chunk_neigh_scores in zip(neigh_idxs, neigh_scores):
        neigh_info.append([])
        for neigh, score in zip(query_chunk_neigh_ids, query_chunk_neigh_scores):
            neigh_info[-1].append(dict(  # the current query chunk list is always the last created
                **index.metadata[neigh],
                score=score,  # add similarity score to neighbor dict
                faiss_idx=neigh,  # add position in FAISS to neighbor dict
            ))
    return neigh_info
