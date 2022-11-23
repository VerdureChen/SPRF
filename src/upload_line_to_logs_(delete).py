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
import wandb


def upload_bsln_result(trec_path, orig_topics):
    result_paths = sorted(Path(trec_path).glob('*.trec'))
    topics = orig_topics
    if orig_topics == "dl20":
        topics = "dl20-passage"

    for run in result_paths:
        run_path = str(run)
        _, total_prf_num, split_num, prf_depth, nsplit, seed, _ = run.name.split('.')
        sys.argv = list(
            itertools.chain(
                ["python", "-m", "pyserini.eval.trec_eval"],
                ["-c"],
                ["-m", "map"],
                ["-m", "ndcg_cut.10"],
                ["-m", "recall.1000"],
                ["-l", f"2"],
                [topics],
                [run_path],

            )
        )
        stdoutput, _ = subprocess.Popen(sys.argv, stdout=subprocess.PIPE).communicate()
        items = str(stdoutput, encoding='utf-8').rstrip().replace('\t', '').replace(' all', '').split('\n')[-3:]
        output = {k: float(v) for k, v in [pair for pair in [item.split() for item in items]]}
        print(f"value in result {run.name}: {output}")

        run_config = {"nsplit": int(nsplit),
                  "total_prf_num": int(total_prf_num),
                  "split_num": int(split_num),
                  "prf_depth": int(prf_depth),
                  "seed": int(seed),
                  }


        metrics = output
        metrics.update(run_config)
        wandb.log(metrics)




def upload_bsln_main():
    parser = argparse.ArgumentParser(description='Conduct a BM25+Rocchio search on sparse indexes.')
    parser.add_argument("--topic_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The topic set name.(dl19-passage & dl20)")
    parser.add_argument("--run_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The initial run path.")
    parser.add_argument("--prf_method",
                        default=None,
                        type=str,
                        required=True,
                        help="The prf method to probe.")

    args = parser.parse_args()
    print('The args: {}'.format(args))
    config = {
              "topics": args.topic_name,
              "prf_method": args.prf_method
              }
    wandb.init(
        project=f"compare_position_prf_{args.topic_name}_100",
        name=f"wandb/dense",
        dir="/home1/cxy/A-SPRF/logs/dense_position/wandb",
        config=config,
    )
    upload_bsln_result(trec_path=args.run_dir, orig_topics=args.topic_name,)
    wandb.finish()

if __name__ == '__main__':
    upload_bsln_main()
