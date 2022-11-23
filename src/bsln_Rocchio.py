'''
use BM25 vector reproduce Rocchio feedback, you can change the feedback documents you like
first you need to have the initial ranking and the query list, you should config
for each query,
'''
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.lucene import querybuilder
from pyserini.index.lucene import IndexReader
from pyserini.query_iterator import get_query_iterator, TopicsFormat
from pyserini.search.lucene import LuceneSearcher
from pyserini.trectools import TrecRun
import json
from tqdm import tqdm
import os
from pyserini import search
from pyserini.analysis import Analyzer, get_lucene_analyzer
import argparse
from collections import Counter
from pyserini.output_writer import get_output_writer, OutputFormat, tie_breaker
import numpy as np
import random
import torch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_topic_iterator(topic_name):
    '''

    :param topic_name: the name of topic set
    :return: topic_iterator:you can use topic_iterator.topics to get a topic dict like
    {id1:{'title':'how long is life cycle of flea'}, id2:...}
    '''
    query_iterator = get_query_iterator(topic_name, TopicsFormat(TopicsFormat.DEFAULT.value))
    return query_iterator


def get_BoW_query(orging_query):
    analyzer = Analyzer(get_lucene_analyzer())
    tokens = analyzer.analyze(orging_query)
    query_count_dict = Counter(tokens)
    # query_dict = dict(query_count_dict)
    return query_count_dict


def get_prf_index(prf: int = 3, start: int = 0, end: int = 200):
    s = random.sample(range(start, end), prf)
    s.sort()
    return s


def get_random_fbs(query_rank, topic_id_list, top_docs, prf_list, run, total_prf_docs):
    qid = topic_id_list[query_rank]
    random_docs = run.get_docs_by_topic(qid, total_prf_docs)
    random_doc_ids = list(random_docs.iloc[prf_list, 2])
    inter = [i for i in random_doc_ids if i in top_docs]
    if inter:
        random_doc_ids = get_random_fbs(query_rank-1, topic_id_list, top_docs, prf_list, run, total_prf_docs)
    return random_doc_ids


def get_fb_doc(trec_path, log_path, depth=3, mode='random_split',total_prf_docs=100, split_num=10, nsplit=0, seed=42):
    '''
    get feedback doc id for each query
    :param trec_path:
    :param depth:
    :param mode:
    :return:
    '''


    query_fb_doc = {}
    run = TrecRun(trec_path)
    topic_id_list = run.topics()
    if mode == 'natural':
        for qid in topic_id_list:
            top_docs = run.get_docs_by_topic(qid, depth)
            top_doc_ids = list(top_docs.iloc[0:depth, 2])
            query_fb_doc[qid] = top_doc_ids
    elif mode == 'random_split':
        setup_seed(seed)
        assert nsplit < split_num, "out of split index!"
        prf_list = get_prf_index(depth, start=0, end=int(total_prf_docs / split_num))
        prf_list = [item + int(total_prf_docs / split_num) * nsplit for item in prf_list]
        for qid in topic_id_list:
            top_docs = run.get_docs_by_topic(qid, total_prf_docs)
            top_doc_ids = list(top_docs.iloc[prf_list, 2])
            query_fb_doc[qid] = top_doc_ids
            with open(log_path, 'a', encoding='utf-8') as lg:
                lg.write(f"seed:{seed}\n"
                         f"Qids:{qid}\n"
                         f"Indexes:{top_doc_ids}\n"
                         f"prf_list:{prf_list}\n")
    elif mode == 'total_random':
        setup_seed(seed)
        assert nsplit < split_num, "out of split index!"
        prf_list = get_prf_index(depth, start=0, end=int(total_prf_docs / split_num))
        prf_list = [item + int(total_prf_docs / split_num) * nsplit for item in prf_list]
        topic_id_list = list(topic_id_list)
        for i, qid in enumerate(topic_id_list):
            top_docs = run.get_docs_by_topic(qid, total_prf_docs)
            top_doc_ids = list(top_docs.iloc[prf_list, 2])
            rand = random.randint(1, len(topic_id_list))
            random_docs = get_random_fbs(i-rand, topic_id_list, top_docs, prf_list, run, total_prf_docs)
            query_fb_doc[qid] = random_docs
            with open(log_path, 'a', encoding='utf-8') as lg:
                lg.write(f"seed:{seed}\n"
                         f"Qids:{qid}\n"
                         f"Indexes:{random_docs}\n"
                         f"prf_list:{prf_list}\n")

    return query_fb_doc


def get_mean_fb_doc_vector(fb_doc_ids, index_reader, mode='tf'):
    num_fb = len(fb_doc_ids)
    doc_vector = Counter()
    for docid in fb_doc_ids:
        tf = index_reader.get_document_vector(docid)
        if mode == 'tf':
            doc_vector += Counter(tf)
        elif mode == 'bm25':
            bm25_vector = {term: index_reader.compute_bm25_term_weight(docid, term, analyzer=None) for term in
                           tf.keys()}
            doc_vector += Counter(bm25_vector)
    # doc_vector = dict(doc_vector)
    for k in doc_vector:
        doc_vector[k] /= num_fb

    return doc_vector


def get_Rocchio_query(orig_query_vector, fb_doc_vector, alpha=0.7, belta=0.3):
    new_query = Counter()
    for k in orig_query_vector:
        orig_query_vector[k] *= alpha
    for k in fb_doc_vector:
        fb_doc_vector[k] *= belta
    new_query = orig_query_vector + fb_doc_vector

    should = querybuilder.JBooleanClauseOccur['should'].value
    boolean_query_builder = querybuilder.get_boolean_query_builder()
    # print(new_query)
    for k in new_query:
        try:
            term = querybuilder.get_term_query(k)
            boost = querybuilder.get_boost_query(term, new_query[k])
            boolean_query_builder.add(boost, should)
        except:
            # print(k)
            pass
    query_now = boolean_query_builder.build()
    return query_now


def search_after_prf(query_now, searcher):
    hits = searcher.search(query_now, k=1000)
    return hits


def run_rocchio(topic_name, initial_run, outpath, log_path, alpha=1, belta=0.75, prf_depth=3,
                total_prf_docs=100, split_num=10, nsplit=0, seed=42, mode="random_split", initial_name="ance"):
    query_fb_doc = get_fb_doc(initial_run, log_path=log_path, depth=prf_depth, mode=mode,
                              total_prf_docs=total_prf_docs, split_num=split_num, nsplit=nsplit, seed=seed)
    query_iterator = get_topic_iterator(topic_name)
    results = []
    index_name = 'msmarco-v1-passage-full'
    index_reader = IndexReader.from_prebuilt_index(index_name)
    searcher = LuceneSearcher.from_prebuilt_index(index_name)
    searcher.set_bm25(0.9, 0.4)
    topics = query_iterator.topics
    for index, (topic_id, text) in enumerate(tqdm(query_iterator, total=len(topics.keys()))):
        topic_text = query_iterator.topics[topic_id]['title']
        query_bow_dict = get_BoW_query(topic_text)
        fb_ids = query_fb_doc[topic_id]
        doc_bow_dict = get_mean_fb_doc_vector(fb_ids, index_reader, mode='tf')
        new_query = get_Rocchio_query(query_bow_dict, doc_bow_dict, alpha, belta)
        hits = search_after_prf(new_query, searcher)
        results.append((topic_id, hits))
    output_writer = get_output_writer(outpath, OutputFormat.TREC, max_hits=1000,
                                      tag=f'{initial_name}+rocchio')
    with output_writer:
        for topic, hits in results:
            output_writer.write(topic, hits)


def sparse_rocchio_main():
    parser = argparse.ArgumentParser(description='Conduct a BM25+Rocchio search on sparse indexes.')
    parser.add_argument("--topic_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The topic set name.")
    parser.add_argument("--initial_run",
                        default=None,
                        type=str,
                        required=True,
                        help="The initial run path.")
    parser.add_argument("--output_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The output file path.")
    parser.add_argument('--alpha',
                        type=float,
                        default=1,
                        help="alpha")
    parser.add_argument('--belta',
                        type=float,
                        default=0.75,
                        help="belta")
    parser.add_argument('--prf-depth', type=int, metavar='num of passages used for PRF', required=False, default=0,
                        help="Specify how many passages are used for PRF, 0: Simple retrieval with no PRF, > 0: perform PRF")
    parser.add_argument('--total_prf_docs', type=int, metavar='total num of passages used for PRF test', required=False,
                        default=1000,
                        help="Specify how many passages are used for PRF test.")
    parser.add_argument('--split_num', type=int, metavar='num of passages split for PRF', required=False, default=10,
                        help="Specify how many passages groups are used for PRF effectiveness.")
    parser.add_argument('--nsplit', type=int, metavar='rank num of passages split for PRF', required=False, default=0,
                        help="Specify the number of group to return prf docs.")
    parser.add_argument('--seed', type=int, metavar='random seed.', required=False, default=42,
                        help="Specify the random seed to return prf docs.")
    parser.add_argument('--log_path', type=str, metavar='log path.', required=False,
                        help='The path for logs.')
    parser.add_argument('--mode', type=str, metavar='prf select mode.', required=False,
                        help='natural, random_split, total_random')
    parser.add_argument('--initial_name', type=str, metavar='initial ranking name.', required=False,
                        help='initial ranking name')
    args = parser.parse_args()
    print('The args: {}'.format(args))

    run_rocchio(args.topic_name, args.initial_run, args.output_path, args.log_path, args.alpha, args.belta,
                prf_depth=args.prf_depth, total_prf_docs=args.total_prf_docs, split_num=args.split_num,
                nsplit=args.nsplit, seed=args.seed, mode=args.mode, initial_name=args.initial_name)


if __name__ == '__main__':
    sparse_rocchio_main()
