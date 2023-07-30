import yaml
import random
import numpy as np
import torch
import time
import os
from omegaconf import OmegaConf
import yaml
import collections


def init_paths(args, exp_name):
    results_path = os.path.join(args.outputs_dir, exp_name)
    checkpoints_path = os.path.join(results_path, "checkpoints")
    paths = {"results": results_path,
             "checkpoints": checkpoints_path
             }

    if args.save_results:
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(checkpoints_path, exist_ok=True)

        # Save a copy of params to the results folder
        output_params_yml_path = os.path.join(results_path, "params.yml")
        print(dict(args))
        with open(output_params_yml_path, 'w') as outfile:
            yaml.dump(OmegaConf.to_container(args), outfile,
                      default_flow_style=False)

    return paths


def flatten_omegaconf_for_wandb(config):
    config = OmegaConf.to_container(config)

    def flatten(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flattened_config = flatten(config)

    return flattened_config


def get_loggable_args(args):
    args = OmegaConf.to_container(args)
    args_to_log = {}
    for k, v in args.items():
        if type(v) != dict:
            args_to_log[k] = v
        else:
            for a, b in args[k].items():
                args_to_log[k + "_" + a] = b
    return args_to_log


def set_random_seed(seed):
    print("Setting random seed: ", seed)
    random.seed(seed)
    torch.cuda.cudnn_enabled = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False


def get_exp_name(args):
    exp_name = f'{args.strategy}'
    exp_name += f'_{args.dataset}'
    exp_name += f'_s{args.seed}'
    t_suff = time.strftime("%m%d%H%M%S")
    exp_name += f'_{t_suff}'

    return exp_name


def load_params(yml_file_path):
    """ Loads param file and returns a dictionary.  """
    with open(yml_file_path, "r") as yaml_file:
        params = yaml.safe_load(yaml_file)

    return params
