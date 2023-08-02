from functools import partial
from torch.optim.lr_scheduler import OneCycleLR

from avalanche.training.supervised import (
    Naive,
    EWC,
    SynapticIntelligence,
    LwF,
    AGEM
)
from avalanche.training.storage_policy import (
    ClassBalancedBuffer,
    ReservoirSamplingBuffer
)
from avalanche.training.plugins import ReplayPlugin, LRSchedulerPlugin
from avalanche.training.plugins import GSS_greedyPlugin
from avalanche.training.supervised import ER_ACE

from .storage_policies import AdaptiveBuffer
from .er_ace_adaptive import ER_ACE_Adaptive
from .er_ace_rs import ER_ACE_RS
from .der import DER
from .der_adaptive import DER_Adaptive
from .der_rs import DER_RS


def get_strategy(
        args,
        model,
        optimizer,
        criterion,
        eval_plugin,
        plugins,
        device
):
    if args.strategy == "Naive":
        STRATEGY = Naive

    elif args.strategy == "EWC":
        STRATEGY = EWC

    elif args.strategy == "SynapticIntelligence":
        STRATEGY = SynapticIntelligence

    elif args.strategy == "LwF":
        STRATEGY = LwF

    elif args.strategy == "AGEM":
        STRATEGY = AGEM

    elif args.strategy == "ER-GSS":
        input_size = [3, args.input_size, args.input_size]
        gss_plugin = GSS_greedyPlugin(mem_size=args.buffer_size,
                                      mem_strength=args.mem_strength,
                                      input_size=input_size)

        plugins.append(gss_plugin)
        STRATEGY = Naive

    elif args.strategy == "ER-ACE":
        STRATEGY = partial(
            ER_ACE,
            mem_size=args.buffer_size)

    elif args.strategy == "ER-Adaptive":
        storage_policy = AdaptiveBuffer(max_size=args.buffer_size,
                                        total_num_classes=args.n_classes)
        replay_plugin = ReplayPlugin(mem_size=args.buffer_size,
                                     storage_policy=storage_policy)
        plugins.append(replay_plugin)

        STRATEGY = Naive

    elif args.strategy == "ER-ACE-Adaptive":
        STRATEGY = partial(
            ER_ACE_Adaptive,
            mem_size=args.buffer_size,
            n_classes=args.n_classes
        )

    elif args.strategy == "ER-DER":
        STRATEGY = partial(
            DER,
            mem_size=args.buffer_size)

    elif args.strategy == "ER-DER-RS":
        STRATEGY = partial(
            DER_RS,
            mem_size=args.buffer_size)

    elif args.strategy == "ER-DER-Adaptive":
        STRATEGY = partial(
            DER_Adaptive,
            mem_size=args.buffer_size,
            total_num_classes=args.n_classes
        )

    elif args.strategy == "ER-CB":
        storage_policy = ClassBalancedBuffer(max_size=args.buffer_size)
        replay_plugin = ReplayPlugin(mem_size=args.buffer_size,
                                     storage_policy=storage_policy)
        plugins.append(replay_plugin)
        STRATEGY = Naive

    elif args.strategy == "ER-RS":
        storage_policy = ReservoirSamplingBuffer(max_size=args.buffer_size)
        replay_plugin = ReplayPlugin(mem_size=args.buffer_size,
                                     storage_policy=storage_policy)
        plugins.append(replay_plugin)
        STRATEGY = Naive

    elif args.strategy == "ER-ACE-RS":
        STRATEGY = partial(
            ER_ACE_RS,
            mem_size=args.buffer_size,
            n_classes=args.n_classes
        )

    else:
        raise NotImplementedError()

    strategy = STRATEGY(model, optimizer, criterion,
                        evaluator=eval_plugin, device=device,
                        plugins=plugins,
                        **args.strategy_params)

    return strategy
