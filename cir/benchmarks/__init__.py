from .cir_sampling_based import (
    cir_cifar100,
    cir_tinyimagenet
)

from .cir_slot_based import (
    cir_slot_mnist,
    cir_slot_cifar100,
    cir_slot_tinyimagenet
)


def get_cir_benchmark(args):
    if args.generator == "sampling_based":
        if args.dataset == "cifar-100":
            benchmark_helper = cir_cifar100
        elif args.dataset == "tinyimagenet":
            benchmark_helper = cir_tinyimagenet
        elif args.dataset == "mnist":
            benchmark_helper = None
        else:
            raise NotImplementedError()

        benchmark = benchmark_helper(
            dataset_root=args.dataset_root,
            n_e=args.n_e,
            s_e=args.s_e,
            p_a=args.p_a,
            sampler_type=args.sampler_type,
            use_all_samples=args.use_all_samples,
            dist_first_occurrence=args.dist_first_occurrence,
            dist_recurrence=args.dist_recurrence,
            seed=args.seed,
            classes_to_use=args.classes_to_use,
        )

    elif args.generator == "slot_based":
        if args.dataset == "cifar-100":
            benchmark_helper = cir_slot_cifar100
        elif args.dataset == "tinyimagenet":
            benchmark_helper = cir_slot_tinyimagenet
        elif args.dataset == "mnist":
            benchmark_helper = cir_slot_mnist
        else:
            raise NotImplementedError()

        benchmark = benchmark_helper(
            dataset_root=args.dataset_root,
            N=args.N,
            K=args.K,
            seed=args.seed
        )

    else:
        raise NotImplementedError()

    return benchmark
