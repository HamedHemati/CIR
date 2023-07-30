import os
import torch
from omegaconf import OmegaConf
from torch_cka import CKA

from avalanche.benchmarks.utils.avalanche_dataset import AvalancheSubset

from cir.models import get_model


def get_exp_config(exp_name, outputs_path="./out/outputs/"):
    exp_path = os.path.join(outputs_path, exp_name)

    # OmegaConf
    config = OmegaConf.load(os.path.join(exp_path, "params.yml"))
    config.exp_path = exp_path

    device = torch.device(f"cuda:{config.cuda}" if torch.cuda.is_available()
                          and config.cuda >= 0 else "cpu")
    print("device: ",  device)

    return config, device


def get_exp_model(config, ckpt_path):
    # Model
    model = get_model(config)

    # Checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt)

    return model


def get_ckpt_path(config, ckpt_id):
    ckpt_path = os.path.join(config.exp_path, f"checkpoints/ckpt_{ckpt_id}.pt")
    return ckpt_path


def get_dataset_subset(dataset, classes, n_samples_per_class):
    targets = torch.LongTensor(dataset.targets)
    all_indices = []
    for c in classes:
        ind_c = torch.where(targets == c)[0][:n_samples_per_class]
        all_indices.append(ind_c)
    selected_indices = torch.cat(all_indices, dim=0)
    dataset_subset = AvalancheSubset(dataset, indices=selected_indices)

    return dataset_subset


def compute_CKA(config, ckpt_id_1, ckpt_id_2, dataloader):
    # Models
    ckpt_path_1 = get_ckpt_path(config, ckpt_id_1)
    model_1 = get_exp_model(config, ckpt_path_1)

    ckpt_path_2 = get_ckpt_path(config, ckpt_id_2)
    model_2 = get_exp_model(config, ckpt_path_2)

    cka = CKA(model_1, model_2,
              model1_name=f"Experience-{ckpt_id_1}",
              model2_name=f"Experience-{ckpt_id_2}",
              device='cuda')

    cka.compare(dataloader)
    results = cka.export()

    return results
