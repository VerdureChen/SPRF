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
import os
from pyserini import search
from pyserini.analysis import Analyzer, get_lucene_analyzer
import logging
import argparse
from collections import Counter
from pyserini.output_writer import get_output_writer, OutputFormat, tie_breaker


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

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
    #query_dict = dict(query_count_dict)
    return query_count_dict

def get_fb_doc(trec_path, depth=3, mode='natural'):
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

    return query_fb_doc

def get_mean_fb_doc_vector(fb_doc_ids, index_reader, mode='bm25'):
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
    #doc_vector = dict(doc_vector)
    for k in doc_vector:
        doc_vector[k] /= 3

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
    #print(new_query)
    for k in new_query:
        try:
            term = querybuilder.get_term_query(k)
            boost = querybuilder.get_boost_query(term, new_query[k])
            boolean_query_builder.add(boost, should)
        except:
            print(k)
            pass
    query_now = boolean_query_builder.build()
    return query_now

def search_after_prf(query_now, searcher):
    hits = searcher.search(query_now, k=1000)
    return hits

def run_rocchio(topic_name, initial_run, outpath, alpha=0.7, belta=0.3):
    query_fb_doc = get_fb_doc(initial_run, depth=3, mode='natural')
    query_iterator = get_topic_iterator(topic_name)
    results = []
    index_name = 'msmarco-v1-passage-full'
    index_reader = IndexReader.from_prebuilt_index(index_name)
    searcher = LuceneSearcher.from_prebuilt_index(index_name)
    searcher.set_bm25(0.9, 0.4)
    for topic_id in query_iterator.topics:
        topic_text = query_iterator.topics[topic_id]['title']
        query_bow_dict = get_BoW_query(topic_text)
        fb_ids = query_fb_doc[topic_id]
        doc_bow_dict = get_mean_fb_doc_vector(fb_ids, index_reader, mode='tf')
        new_query = get_Rocchio_query(query_bow_dict, doc_bow_dict, alpha, belta)
        hits = search_after_prf(new_query, searcher)
        results.append((topic_id, hits))
    output_writer = get_output_writer(outpath, OutputFormat.TREC, max_hits=1000,
                                      tag='bm25+rocchio')
    with output_writer:
        for topic, hits in results:
            output_writer.write(topic, hits)

if __name__ == '__main__':
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
    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    run_rocchio(args.topic_name, args.initial_run, args.output_path, args.alpha, args.belta)