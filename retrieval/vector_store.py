import faiss
import numpy as np
import torch
import os
import json
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # need to prevent OMP crash when running on mac


class MedLitRagIndex:
    def __init__(self, d: int = 768):
        self.d = d
        self.index = faiss.IndexFlatIP(d)
        self.metadata = []

    def add(self, xb: torch.Tensor | np.ndarray,  metadata: list[dict] | None = None) -> faiss.Index:
        """
        Adds database embeddings (along with metadata) to a FAISS index.
        :param xb: database embeddings
        :param metadata: emdeddings metadata
        :return: updated index
        """
        if xb.shape[1] != self.d:
            raise ValueError(f'Dimension 1 of xb should be {self.d}, got {xb.shape[1]} ({xb.shape=}).')
        if metadata and len(metadata) != xb.shape[0]:
            raise ValueError(f"metadata length {len(metadata)} must match number of embeddings {xb.shape[0]}")

        # FAISS excpetcs a float32 numpy array
        if isinstance(xb, torch.Tensor):
            xb = xb.numpy()
        xb = np.ascontiguousarray(xb, dtype='float32')

        self.index.add(x=xb)
        if metadata:
            self.metadata.extend(metadata)

        return self.index

    def save_index(self) -> None:
        """
        Save index and metadata.
        :return: None
        """
        output_path = Path(__file__).parent.parent / 'outputs'
        output_path.mkdir(parents=True, exist_ok=True)

        # Save index
        idx_fname = 'index'
        idx_fpath = output_path / idx_fname
        faiss.write_index(self.index, str(idx_fpath))

        # Define and save metadata
        metadata_fname = 'metadata.json'
        metadata_fpath = output_path / metadata_fname

        with open(metadata_fpath, 'w') as f:
            json.dump(self.metadata, f)
        print(
            f"Index and metadata exported at {output_path} as {idx_fname} and {metadata_fname}"
        )

    def load(self, idx_fpath: Path | None = None, metadata_fpath: Path | None = None) -> tuple[faiss.Index, list]:
        """
        Load index and metadata.
        :param idx_fpath:
        :param metadata_fpath:
        :return:
        """
        output_path = Path(__file__).parent.parent / 'outputs'
        idx_fpath = idx_fpath or output_path / 'index'
        metadata_fpath = metadata_fpath or output_path / 'metadata.json'
        self.index = faiss.read_index(str(idx_fpath))
        with open(metadata_fpath, 'r') as f:
            self.metadata = json.load(self.metadata, f)
        return self.index, self.metadata

    def search(self, xq: torch.Tensor | np.ndarray, k: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Search query neighbors into an index.
        :param xq: query embedding
        :param k: number of vectors most similar to the query
        :return:
        """
        # FAISS expects a float32 numpy array
        if isinstance(xq, torch.Tensor):
            xq = xq.numpy()
        xq = np.ascontiguousarray(xq, dtype='float32')

        neigh_dists, neigh_idxs = self.index.search(x=xq, k=k)
        return neigh_dists, neigh_idxs

