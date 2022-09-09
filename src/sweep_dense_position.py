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

from Dense_Select_PRF.__main__ import dsprf_main

# assert Path(".git").exists()
os.environ["PL_DISABLE_FORK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7"
def run_cli(config, index, encoder, threads, sparse_index,
            ance_prf_encoder, prf_method, devices: int = 1):
    os.chdir(os.environ["TUNE_ORIG_WORKING_DIR"])

    output_name = f"{Path.cwd().parents[0]}/output/dense_position_{prf_method}/{config['topics']}" \
                  f"/run.{config['total_prf_docs']}.{config['split_num']}.{config['prf_depth']}.{config['nsplit']}.{config['seed']}.trec"
    log_name = f"{Path.cwd().parents[0]}/logs/dense_position_{prf_method}/{config['topics']}" \
                  f"/run.{config['total_prf_docs']}.{config['split_num']}.{config['prf_depth']}.{config['nsplit']}.{config['seed']}.log"
    Path(log_name).parents[0].mkdir(parents=True, exist_ok=True)
    sys.argv = list(
        itertools.chain(
            ["Dense_Select_PRF"],
            ["--topics", f"{config['topics']}"],
            ["--index", f"{index}"],
            ["--encoder", f"{encoder}"],
            ["--batch-size", f"{config['batch_size']}"],
            ["--output", f"{output_name}"],
            ["--total_prf_docs", f"{config['total_prf_docs']}"],
            ["--split_num", f"{config['split_num']}"],
            ["--nsplit", f"{config['nsplit']}"],
            ["--prf-depth", f"{config['prf_depth']}"],
            ["--prf-method", f"{prf_method}"],
            ["--threads", f"{threads}"],
            ["--sparse-index", f"{sparse_index}"],
            ["--ance-prf-encoder", f"{ance_prf_encoder}"],
            ["--device", f"cuda:0"],
            ["--seed", f"{config['seed']}"],
            ["--log_path", f"{log_name}"]
        )
    )
    print(shlex.join(sys.argv))
    # dsprf_main()

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
        project=f"dense_position_prf_{config['topics']}_{config['total_prf_docs']}",
        name=f"wandb/{Path(output_name).name}",
        dir="/home1/cxy/A-SPRF/logs/dense_position/wandb",
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


def sweep_dense_position(
        index: Literal["msmarco-passage-ance-bf"] = "msmarco-passage-ance-bf",
        encoder: Literal["castorini/ance-msmarco-passage"] = "castorini/ance-msmarco-passage",
        prf_method: Literal["ance-prf"] = "ance-prf",
        sparse_index: Literal["msmarco-passage"] = "msmarco-passage",
        ance_prf_encoder: str = "/home1/cxy/.cache/pyserini/ckpt/k3_checkpoint",
        output_dir: str = "/home1/cxy/A-SPRF/split_prf/ance_prf/",
        gpus_per_trial: Union[int, float] = 1,
        seed=None,
        split_num=None,
        nsplit=None,
        total_prf_docs=None,
        prf_depth=None,
        topics=None,
        batch_size=None,
):

    if topics is None:
        # topics = ["dl19-passage"]
        topics = ["dl20"]
        # topics = ["dl19-passage", "dl20"]
    if prf_depth is None:
        prf_depth = [3]
    if total_prf_docs is None:
        # total_prf_docs = [100]
        total_prf_docs = [100, 1000]
    if nsplit is None:
        # nsplit = [1]
        nsplit = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    if split_num is None:
        split_num = [10]
    if seed is None:
        # seed = [42]
        seed = [42, 51, 10, 23, 34, 65, 78, 86, 97, 9]
    if batch_size is None:
        batch_size = [32]

    param_space = {
        "seed": tune.grid_search(seed),
        "split_num": tune.grid_search(split_num),
        "nsplit": tune.grid_search(nsplit),
        "total_prf_docs": tune.grid_search(total_prf_docs),
        "prf_depth": tune.grid_search(prf_depth),
        "topics": tune.grid_search(topics),
        "batch_size": tune.grid_search(batch_size)
    }

    tune_config = tune.TuneConfig()
    run_config = air.RunConfig(
        name="dense_position",
        local_dir="../logs/tune/dense_position/ance_prf",
        log_to_file=True,
        verbose=1,
    )
    trainable = tune.with_parameters(
        run_cli,
        index=index,
        encoder=encoder,
        threads=12,
        sparse_index=sparse_index,
        ance_prf_encoder=ance_prf_encoder,
        prf_method=prf_method,
        devices=math.ceil(gpus_per_trial),
    )
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"gpu": gpus_per_trial}),
        param_space=param_space,
        tune_config=tune_config,
        run_config=run_config,
    )
    tuner.fit()




if __name__ == "__main__":
    CLI(sweep_dense_position)