from avalanche.models import SimpleCNN as AVLSimpleCNN

from .resnet18 import resnet18


def get_model(args):
    if args.model == "avl_simple_cnn":
        model = AVLSimpleCNN(num_classes=args.n_classes)
    elif args.model == "resnet18":
        model = resnet18(nclasses=args.n_classes)
    else:
        raise NotImplementedError()

    return model
