import os
from dataclasses import dataclass
from typing import Dict, List, Union, Tuple
from transformers.file_utils import is_faiss_available
import numpy as np
import random
if is_faiss_available():
    import faiss
import faiss
import torch
import logging


from pyserini.search import FaissSearcher




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)




@dataclass
class DenseSearchResult:
    docid: str
    score: float


@dataclass
class PRFDenseSearchResult:
    docid: str
    score: float
    vectors: [float]


class FaissSearcherSplitPRF(FaissSearcher):
    def get_prf_index(self, prf: int = 3, start: int = 0, end: int = 200):
        s = random.sample(range(start, end), prf)
        s.sort()
        return s

    def search(self, query: Union[str, np.ndarray], seed: int = 42, fbn: int = 3, spt: int = 10, nspt: int = 0,
               k: int = 1000, threads: int = 1, return_vector: bool = False, log_name: str = 'dense_position') \
            -> Union[List[DenseSearchResult], Tuple[np.ndarray, List[PRFDenseSearchResult]]]:
        """Search the collection.

        Parameters
        ----------
        query : Union[str, np.ndarray]
            query text or query embeddings.
        fbn  : int
            Number of feedback documents to return.
        spt : int
            Number of splits of the retrieval list.
        nspt : int
            Number of the rank of the split.
        k : int
            Number of hits the index to return.
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
        setup_seed(seed)
        assert nspt < spt, "out of split index!"
        if isinstance(query, str):
            emb_q = self.query_encoder.encode(query)
            assert len(emb_q) == self.dimension
            emb_q = emb_q.reshape((1, len(emb_q)))
        else:
            emb_q = query
        faiss.omp_set_num_threads(threads)
        prf_list = self.get_prf_index(fbn, start=0, end=int(k/spt))
        prf_list = [item+int(k/spt)*nspt for item in prf_list]
        if return_vector:
            distances, indexes, vectors = self.index.search_and_reconstruct(emb_q, k)
            vectors = vectors[0, prf_list]
            distances = distances.flat[prf_list]
            indexes = indexes.flat[prf_list]
            return emb_q, [PRFDenseSearchResult(self.docids[idx], score, vector)
                           for score, idx, vector in zip(distances, indexes, vectors) if idx != -1]
        else:
            distances, indexes = self.index.search(emb_q, k)
            distances = distances.flat
            indexes = indexes.flat
            return [DenseSearchResult(self.docids[idx], score)
                    for score, idx in zip(distances, indexes) if idx != -1]

    def batch_search(self, queries: Union[List[str], np.ndarray], q_ids: List[str], seed: int = 42, fbn: int = 3, spt: int = 10, nspt: int = 0,
                     k: int = 1000, threads: int = 1, return_vector: bool = False, log_name: str = 'dense_position', rand_prf: bool = False) \
            -> Union[Dict[str, List[DenseSearchResult]], Tuple[np.ndarray, Dict[str, List[PRFDenseSearchResult]]]]:
        """

        Parameters
        ----------
        queries : Union[List[str], np.ndarray]
            List of query texts or list of query embeddings
        q_ids : List[str]
            List of corresponding query ids.
        fbn  : int
            Number of feedback documents to return.
        spt : int
            Number of splits of the retrieval list.
        nspt : int
            Number of the rank of the split.
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
        setup_seed(seed)
        assert nspt < spt, "out of split index!"
        if isinstance(queries, np.ndarray):
            q_embs = queries
        else:
            q_embs = np.array([self.query_encoder.encode(q) for q in queries])
            n, m = q_embs.shape
            assert m == self.dimension
        faiss.omp_set_num_threads(threads)
        prf_list = self.get_prf_index(fbn, start=0, end=int(k / spt))
        prf_list = [item + int(k / spt) * nspt for item in prf_list]

        if return_vector:
            D_o, I_o, V_o = self.index.search_and_reconstruct(q_embs, k)
            if rand_prf:
                D,I,V = self.get_random_fb(D_o, I_o, V_o, prf_list)
            else:
                D = D_o[:, prf_list]
                I = I_o[:, prf_list]
                V = V_o[:, prf_list]
            with open(log_name, 'a', encoding='utf-8') as lg:
                lg.write(f"seed:{seed}\n"
                         f"Qids:{q_ids}\n"
                         f"Indexes:{I}\n"
                         f"prf_list:{prf_list}\n"
                         f"rand_prf:{rand_prf}\n")
            return q_embs, {key: [PRFDenseSearchResult(self.docids[idx], score, vector)
                                  for score, idx, vector in zip(distances, indexes, vectors) if idx != -1]
                            for key, distances, indexes, vectors in zip(q_ids, D, I, V)}
        else:
            D, I = self.index.search(q_embs, k)
            return {key: [DenseSearchResult(self.docids[idx], score)
                          for score, idx in zip(distances, indexes) if idx != -1]
                    for key, distances, indexes in zip(q_ids, D, I)}


    def get_random_fb(self, D_o, I_o, V_o, prf_list):
        permutation = np.random.permutation(D_o.shape[0])

        D = D_o[:, prf_list]
        I = I_o[:, prf_list]
        V = V_o[:, prf_list]

        D_pm = D[permutation]
        I_pm = I[permutation]
        V_pm = V[permutation]
        flag = 0
        for line1, line2 in zip(I_o, I_pm):
            if [i for i in line1 if i in line2]:
                flag = 1
                break
        if flag == 1:
            D_pm, I_pm, V_pm = self.get_random_fb(D_o, I_o, V_o, prf_list)

        return D_pm, I_pm, V_pm


