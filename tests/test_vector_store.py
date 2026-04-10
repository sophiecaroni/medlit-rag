import numpy as np
import pytest
from retrieval.vector_store import MedLitRagIndex


def test_add():
    # Initialize index
    index = MedLitRagIndex(d=10)

    # Initialise database
    nb = 10
    np.random.seed(42)
    xb = np.random.random((nb, index.d)).astype('float32')
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)

    # Create test metadata
    metadata = [
        {"pubmed_id": str(i), "title": f"Paper {i}", "chunk_text": f"text {i}", "chunk_index": i}
        for i in range(nb)
    ]
    
    # Add database and metadata
    index.add(xb, metadata)

    # Assert index and metadata contain nb elements
    assert index.index.ntotal == nb
    assert len(index.metadata) == nb
    
    # Check metadata content
    assert index.metadata[0] == {"pubmed_id": "0", "title": "Paper 0", "chunk_text": "text 0", "chunk_index": 0}
    
    
def test_add_mismatched_metadata_raises():
    # Initialize index
    index = MedLitRagIndex(d=10)

    # Initialise database
    nb = 10
    np.random.seed(42) 
    xb = np.random.random((nb, index.d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)  # normalize

    # metadata length (nb-1) doesn't match number of embeddings (nb)
    mismatched_metadata = [dict() for _ in range(nb-1)]
    with pytest.raises(ValueError):
        index.add(xb, mismatched_metadata)


@pytest.mark.parametrize(
    "inp, expected",
    [
        (1, np.array([[0]])),
        (2, np.array([[1]])),
        (3, np.array([[2]])),
        (4, np.array([[3]])), ]
)
def test_search(inp, expected):
    # Initialize index
    index = MedLitRagIndex(d=10)

    # Initialise database
    nb = 10  # database size
    np.random.seed(42)
    xb = np.random.random((nb, index.d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)  # normalize

    # Add database (and metadata of the same length) to index
    metadata = [dict() for _ in range(nb)]
    index.add(xb, metadata)
    k = 1
    query = xb[inp-1:inp]  # select n-th embedding in the index (e.g. inp 1 selects first embedding)

    # The nearest neighbor should be the vector itself
    _, idxs = index.search(xq=query, k=k)
    np.testing.assert_array_equal(idxs, expected)


