from cir.utils.limit_threads import *

import torch
import torch.nn as nn
import wandb
import pickle
import os
import hydra
from hydra.utils import get_original_cwd

from avalanche.evaluation.metrics import loss_metrics, accuracy_metrics, \
    forgetting_metrics
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin

from cir.utils.generic import init_paths, get_exp_name, get_loggable_args
from cir.benchmarks import get_cir_benchmark
from cir.models import get_model
from cir.strategies import get_strategy
from cir.metrics import (
    ExperiencePerClassAccuracy,
    ExperienceSeenClassesAccuracy
)
from cir.plugins import CheckpointSaver


@hydra.main(config_path="conf", config_name="class_incremental")
def main(args):
    # Initialization
    exp_name = get_exp_name(args)
    args.dataset_root = os.path.join(get_original_cwd(), args.dataset_root)
    args.outputs_dir = os.path.join(get_original_cwd(), args.outputs_dir)

    paths = init_paths(args, exp_name)

    # Set device
    device = torch.device(args.device)
    print("Device: ", device)

    # Benchmark and evaluation plugins
    benchmark = get_cir_benchmark(args)

    # Update number of classes
    if args.n_pretraining_classes > 0:
        args.n_classes = args.n_classes - args.n_pretraining_classes

    # Loggers
    loggers = [InteractiveLogger()]
    if args.wandb_proj != "":
        wandb_logger = WandBLogger(project_name=args.wandb_proj,
                                   run_name=exp_name,
                                   config=get_loggable_args(args))
        loggers += [wandb_logger]

    # Per-class and see-classes accuracy
    class_based_accuracies = [
        ExperiencePerClassAccuracy(
            args.n_classes,
            present_classes_in_each_exp=benchmark.details["classes_in_each_exp"]
        ),
        ExperienceSeenClassesAccuracy(
            args.n_classes,
            device=device,
            seen_classes_in_each_exp=benchmark.details["seen_classes_up_to_exp"]
        )
    ]

    eval_plugin = EvaluationPlugin(
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        accuracy_metrics(minibatch=True, epoch=True,
                         experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        class_based_accuracies,
        loggers=loggers
    )

    # Strategy
    model = get_model(args)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    plugins = []
    if args.save_results:
        plugins.append(CheckpointSaver(
            checkpoint_saving_steps=args.checkpoint_saving_steps,
            checkpoint_dir_path=paths["checkpoints"]))

    strategy = get_strategy(args, model, optimizer, criterion,
                            eval_plugin, plugins, device)

    # ==========> Main training and evaluation loop
    results = []
    for i, exp in enumerate(benchmark.train_stream):
        print(f"Starting training on experience {i} ...")
        strategy.train(exp, num_workers=args.num_workers)
        res = strategy.eval(benchmark.test_stream[0],
                            num_workers=args.num_workers)

        results.append(res)

    # Finalize logs and finish results
    if args.wandb_proj != "":
        wandb.finish()

    if args.save_results:
        save_path = os.path.join(paths["results"], "results.pkl")
        with open(save_path, "wb") as res_file:
            pickle.dump(results, res_file)
        ckpt_path = os.path.join(paths["results"], "checkpoints/ckpt_final.pt")
        torch.save(strategy.model.state_dict(), ckpt_path)


if __name__ == "__main__":
    main()
