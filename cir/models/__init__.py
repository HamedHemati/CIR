from avalanche.models import SimpleCNN as AVLSimpleCNN
from avalanche.models import SimpleMLP
from .resnet18 import resnet18


def get_model(args):
    if args.model == "avl_simple_cnn":
        model = AVLSimpleCNN(num_classes=args.n_classes)
    elif args.model == "simple_mlp":
        model = SimpleMLP(args.n_classes, hidden_size=256, hidden_layers=2)
    elif args.model == "resnet18":
        model = resnet18(nclasses=args.n_classes)
    else:
        raise NotImplementedError()

    return model
