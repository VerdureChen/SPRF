import itertools
import json
import math
import os
import shlex
import sys
import logging
from pathlib import Path
from typing import Literal, Optional, Union
import subprocess
from jsonargparse import CLI
from ray import air, tune
import argparse


def upload_bsln_result(trec_path, orig_topics, total_prf_docs, prf_method):
    log_name = f"{Path.cwd().parents[0]}/logs/dense_position_{prf_method}/{orig_topics}" \
               f"/run.{Path(trec_path).name}.log"

    topics = orig_topics
    if orig_topics == "dl20":
        topics = "dl20-passage"
    sys.argv = list(
        itertools.chain(
            ["python", "-m", "pyserini.eval.trec_eval"],
            ["-c"],
            ["-m", "map"],
            ["-m", "ndcg_cut.10"],
            ["-m", "recall.1000"],
            ["-l", f"2"],
            [topics],
            [trec_path],

        )
    )
    stdoutput, _ = subprocess.Popen(sys.argv, stdout=subprocess.PIPE).communicate()
    items = str(stdoutput, encoding='utf-8').rstrip().replace('\t', '').replace(' all', '').split('\n')[-3:]
    output = {k: float(v) for k, v in [pair for pair in [item.split() for item in items]]}
    print(f"value in result: {output}")

    config = {"nsplit": 999, "topics": orig_topics, }
    import wandb
    wandb.init(
        project=f"vector_position_prf_{orig_topics}_{total_prf_docs}",
        name=f"wandb/{Path(trec_path).name}",
        dir="/home1/cxy/A-SPRF/logs/sparse_position/wandb",
        config=config,
    )
    metrics = output
    wandb.log(metrics)
    wandb.finish()
    with open(log_name, 'a', encoding='utf-8') as lg:
        lg.write('##########{results}###########\n')
        for metric in sorted(metrics):
            lg.write('{}: {}\n'.format(metric, metrics[metric]))
        lg.write('##############################\n')


def upload_bsln_main():
    parser = argparse.ArgumentParser(description='Conduct a BM25+Rocchio search on sparse indexes.')
    parser.add_argument("--topic_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The topic set name.(dl19-passage & dl20)")
    parser.add_argument("--initial_run",
                        default=None,
                        type=str,
                        required=True,
                        help="The initial run path.")
    parser.add_argument("--prf_method",
                        default=None,
                        type=str,
                        required=True,
                        help="The prf method to probe.")
    parser.add_argument('--total_prf_docs', type=int, metavar='total num of passages used for PRF test', required=False,
                        default=1000,
                        help="Specify how many passages are used for PRF test.")

    args = parser.parse_args()
    print('The args: {}'.format(args))
    upload_bsln_result(trec_path=args.initial_run, orig_topics=args.topic_name, total_prf_docs=args.total_prf_docs,
                       prf_method=args.prf_method)

if __name__ == '__main__':
    upload_bsln_main()
