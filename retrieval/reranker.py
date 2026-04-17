from sentence_transformers import CrossEncoder
from copy import deepcopy

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_neighbors(
        neighbors_information: list[list[dict]],
        query_chunks: list[str],
        top_k: int = 5,
) -> list[list[dict]]:
    """
    Re-rank neighbors of a query chunks using a cross-encoder and only return top-k similar neighbors.
    :param neighbors_information: Lists of neighbors information per query chunk.
    :param query_chunks: List of chunks of a query.
    :param top_k: Number of neighbors to return
    :return: Lists of top-k neighbors information (with updated scores) per query chunk.
    """

    # Initialise reranked result - use deep copy to avoid mutating the input internal lists
    reranked = deepcopy(neighbors_information)
    for chunk_i, (chunk, neighbors) in enumerate(zip(query_chunks, neighbors_information)):
        pairs = [[chunk, neighbor['chunk_text']] for neighbor in neighbors]
        new_scores = model.predict(pairs)

        # Update score
        for neigh_i, chunk_dict in enumerate(reranked[chunk_i]):
            chunk_dict['score'] = float(new_scores[neigh_i])  # because cross-encoder returns np.floats

        # Sort neighbors based on new scores and only keep top-k ones
        reranked[chunk_i].sort(key=lambda neighbor: neighbor['score'], reverse=True)
        reranked[chunk_i] = reranked[chunk_i][:top_k]

    return reranked


