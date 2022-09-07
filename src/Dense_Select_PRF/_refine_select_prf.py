import os
from dataclasses import dataclass
from typing import Dict, List, Union, Tuple

import numpy as np

from transformers.file_utils import is_faiss_available
from pyserini.search import FaissSearcher


if is_faiss_available():
    import faiss


@dataclass
class DenseSearchResult:
    docid: str
    score: float


@dataclass
class PRFDenseSearchResult:
    docid: str
    score: float
    vectors: [float]


def search(self, query: Union[str, np.ndarray], k: int = 10, threads: int = 1, return_vector: bool = False) \
        -> Union[List[DenseSearchResult], Tuple[np.ndarray, List[PRFDenseSearchResult]]]:
    """Search the collection.

    Parameters
    ----------
    query : Union[str, np.ndarray]
        query text or query embeddings
    k : int
        Number of hits to return.
    threads : int
        Maximum number of threads to use for intra-query search.
    return_vector : bool
        Return the results with vectors
    Returns
    -------
    Union[List[DenseSearchResult], Tuple[np.ndarray, List[PRFDenseSearchResult]]]
        Either returns a list of search results.
        Or returns the query vector with the list of PRF dense search results with vectors.
    """
    if isinstance(query, str):
        emb_q = self.query_encoder.encode(query)
        assert len(emb_q) == self.dimension
        emb_q = emb_q.reshape((1, len(emb_q)))
    else:
        emb_q = query
    faiss.omp_set_num_threads(threads)
    if return_vector:
        distances, indexes, vectors = self.index.search_and_reconstruct(emb_q, k)
        vectors = vectors[0]
        distances = distances.flat
        indexes = indexes.flat
        return emb_q, [PRFDenseSearchResult(self.docids[idx], score, vector)
                       for score, idx, vector in zip(distances, indexes, vectors) if idx != -1]
    else:
        distances, indexes = self.index.search(emb_q, k)
        distances = distances.flat
        indexes = indexes.flat
        return [DenseSearchResult(self.docids[idx], score)
                for score, idx in zip(distances, indexes) if idx != -1]


FaissSearcher.search = search


def batch_search(self, queries: Union[List[str], np.ndarray], q_ids: List[str], k: int = 10,
                 threads: int = 1, return_vector: bool = False) \
        -> Union[Dict[str, List[DenseSearchResult]], Tuple[np.ndarray, Dict[str, List[PRFDenseSearchResult]]]]:
    """

    Parameters
    ----------
    queries : Union[List[str], np.ndarray]
        List of query texts or list of query embeddings
    q_ids : List[str]
        List of corresponding query ids.
    k : int
        Number of hits to return.
    threads : int
        Maximum number of threads to use.
    return_vector : bool
        Return the results with vectors

    Returns
    -------
    Union[Dict[str, List[DenseSearchResult]], Tuple[np.ndarray, Dict[str, List[PRFDenseSearchResult]]]]
        Either returns a dictionary holding the search results, with the query ids as keys and the
        corresponding lists of search results as the values.
        Or returns a tuple with ndarray of query vectors and a dictionary of PRF Dense Search Results with vectors
    """
    if isinstance(queries, np.ndarray):
        q_embs = queries
    else:
        q_embs = np.array([self.query_encoder.encode(q) for q in queries])
        n, m = q_embs.shape
        assert m == self.dimension
    faiss.omp_set_num_threads(threads)
    if return_vector:
        D, I, V = self.index.search_and_reconstruct(q_embs, k)
        return q_embs, {key: [PRFDenseSearchResult(self.docids[idx], score, vector)
                              for score, idx, vector in zip(distances, indexes, vectors) if idx != -1]
                        for key, distances, indexes, vectors in zip(q_ids, D, I, V)}
    else:
        D, I = self.index.search(q_embs, k)
        return {key: [DenseSearchResult(self.docids[idx], score)
                      for score, idx in zip(distances, indexes) if idx != -1]
                for key, distances, indexes in zip(q_ids, D, I)}


FaissSearcher.batch_search = batch_search
