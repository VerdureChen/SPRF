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
print(str(Path.cwd()))
sys.path.append(str(Path.cwd()))

from bsln_Rocchio import sparse_rocchio_main

# assert Path(".git").exists()
os.environ["PL_DISABLE_FORK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7"
def run_cli(config, initial_name, initial_run, prf_method):
    os.chdir(os.environ["TUNE_ORIG_WORKING_DIR"])

    output_name = f"{Path.cwd().parents[0]}/output/sparse_position_{prf_method}/{config['topics']}" \
                  f"/run.{config['total_prf_docs']}.{config['split_num']}.{config['prf_depth']}.{config['nsplit']}.{config['seed']}.trec"
    log_name = f"{Path.cwd().parents[0]}/logs/sparse_position_{prf_method}/{config['topics']}" \
                  f"/run.{config['total_prf_docs']}.{config['split_num']}.{config['prf_depth']}.{config['nsplit']}.{config['seed']}.log"
    Path(log_name).parents[0].mkdir(parents=True, exist_ok=True)
    sys.argv = list(
        itertools.chain(
            ["bsln_Rocchio.py"],
            ["--topic_name", f"{config['topics']}"],
            ["--initial_run", f"{initial_run}"],
            ["--initial_name", f"{initial_name}"],
            ["--output_path", f"{output_name}"],
            ["--total_prf_docs", f"{config['total_prf_docs']}"],
            ["--split_num", f"{config['split_num']}"],
            ["--nsplit", f"{config['nsplit']}"],
            ["--prf-depth", f"{config['prf_depth']}"],
            ["--seed", f"{config['seed']}"],
            ["--log_path", f"{log_name}"],
            ["--mode", f"{config['mode']}"],
            ["--alpha", f"{config['alpha']}"],
            ["--belta", f"{config['belta']}"]
        )
    )
    print(shlex.join(sys.argv))
    sparse_rocchio_main()

    topics = config['topics']
    if config['topics'] == "dl20":
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
            [output_name],

        )
    )
    stdoutput, _ = subprocess.Popen(sys.argv, stdout=subprocess.PIPE).communicate()
    items = str(stdoutput, encoding='utf-8').rstrip().replace('\t', '').replace(' all', '').split('\n')[-3:]
    output = {k: float(v) for k, v in [pair for pair in [item.split() for item in items]]}
    print(f"value in result: {output}")

    import wandb
    wandb.init(
        project=f"sparse_position_prf_{config['topics']}_{config['total_prf_docs']}",
        name=f"wandb/{Path(output_name).name}",
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


def sweep_sparse_position(
        initial_run: str = "/home1/cxy/A-SPRF/Baselines/run.msmarco-v1-passage.ance-otf.dl20.txt",
        mode: Literal["random_split", "natural"] = "random_split",
        initial_name: str = "ance",
        prf_method: Literal["rocchio"] = "rocchio",
        alpha=None,
        belta=None,
        seed=None,
        split_num=None,
        nsplit=None,
        total_prf_docs=None,
        prf_depth=None,
        topics=None,
):

    if topics is None:
        # topics = ["dl19-passage"]
        topics = ["dl20"]
        # topics = ["dl19-passage", "dl20"]
    if prf_depth is None:
        prf_depth = [3]
    if total_prf_docs is None:
        # total_prf_docs = [1000]
        total_prf_docs = [100, 1000]
    if nsplit is None:
        # nsplit = [1]
        nsplit = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    if split_num is None:
        split_num = [10]
    if seed is None:
        # seed = [42]
        seed = [42, 51, 10, 23, 34, 65, 78, 86, 97, 9]
    if alpha is None:
        alpha = [1]
    if belta is None:
        belta = [0.75]

    param_space = {
        "seed": tune.grid_search(seed),
        "split_num": tune.grid_search(split_num),
        "nsplit": tune.grid_search(nsplit),
        "total_prf_docs": tune.grid_search(total_prf_docs),
        "prf_depth": tune.grid_search(prf_depth),
        "topics": tune.grid_search(topics),
        "alpha": tune.grid_search(alpha),
        "belta": tune.grid_search(belta),
        "mode": tune.grid_search([mode])
    }

    tune_config = tune.TuneConfig()
    run_config = air.RunConfig(
        name="sparse_position",
        local_dir="../logs/tune/sparse_position/ance_prf",
        log_to_file=True,
        verbose=0,
    )
    trainable = tune.with_parameters(
        run_cli,
        initial_name=initial_name,
        initial_run=initial_run,
        prf_method=prf_method
    )
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={}),
        param_space=param_space,
        tune_config=tune_config,
        run_config=run_config,
    )
    tuner.fit()




if __name__ == "__main__":
    CLI(sweep_sparse_position)