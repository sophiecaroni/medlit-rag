import numpy as np
from ingestion.loader import load_articles, fetch_articles
from ingestion.chunker import chunk_text
from ingestion.embedder import embed_text
from retrieval.vector_store import MedLitRagIndex


def _expand_metadata_to_chunks(arts_metadata: list[dict], chunked_abstracts: list[list[str]]) -> list[dict]:
    """
    Repeat each article's metadata once per chunk, so that metadata[i] corresponds to index embeddings[i].

    :param arts_metadata: List of article metadata-dicts (pmid, year, title, text).
    :param chunked_abstracts: List of chunk lists, one per article.
    :return: Flat list of metadata dicts (without 'text'), where length equals total number of chunks.
    """
    metadata_per_chunk = []

    # For each chunk of an abstract, retrieve the metadata of the abstract article
    for art_dict, chunks in zip(arts_metadata, chunked_abstracts):

        # First retrieve id, year, title of the entire article
        base = {k: v for k, v in art_dict.items() if k != 'text'}

        # Then retrieve the abstract text of the current chunk and add (along with article metadata) to chunk metadata
        for chunk in chunks:
            metadata_per_chunk.append({**base, "chunk_text": chunk})

    return metadata_per_chunk


def ingest(fast: bool = False, save: bool = False):
    """
    Run the full ingestion pipeline: fetch articles, chunk abstracts, embed chunks, and build the FAISS index.
    :param fast: If True, fetches a small number of articles for fast iteration.
    :param save: If True, saves the index and metadata to disk.
    """

    # Retrieve articles
    art_ids = load_articles(fast=fast)
    arts_metadata = fetch_articles(art_ids)

    # Chunk articles abstracts
    abstracts = [article_dict['text'] for article_dict in arts_metadata]
    chunked_abstracts = [chunk_text(abstract) for abstract in abstracts]
    embeddings = np.array([
        embed_text(chunk).numpy()  # convert tensors for cleaner handling
        for abst in chunked_abstracts for chunk in abst
    ])  # shape: (n_chunks, 768)  # n_chunks from all abstracts together

    # Initialize index
    index = MedLitRagIndex()

    # Define metadata for each chunked abstract
    embeddings_metadata = _expand_metadata_to_chunks(arts_metadata, chunked_abstracts)

    # Add embeddings and metadata to index
    index.add(embeddings, embeddings_metadata)

    print(
        f"✅ Created index for {index.index.ntotal} embeddings of abstract chunks."
    )
    if save:
        # Export index
        idx_fname = 'index_small' if fast else None  # None for default
        metadata_fname = 'metadata_small.json' if fast else None  # None for default
        index.save_index(verbose=True, idx_fname=idx_fname, metadata_fname=metadata_fname)


if __name__ == "__main__":
    ingest(
        fast=False,
        save=True,
    )
