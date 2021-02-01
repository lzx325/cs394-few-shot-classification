from utils.data import get_datasets
from torchvision.datasets import Omniglot
from models.maml import MAMLModel
from models.pure_layers import PureSequential,PureConv2d,PureLinear,PureProxy
import torch
import torch.nn as nn
import torch.nn.functional as F
if __name__=="__main__":
    def PureReLU():
        return PureProxy(F.relu)
    def PureAvgPool2d(pool_size):
        return PureProxy(lambda x: F.avg_pool2d(x,pool_size))
    def PureFlatten():
        return PureProxy(lambda x: torch.flatten(x,start_dim=1))
    def PureAdaptiveAvgPool2d(pool_size):
        return PureProxy(lambda x: F.adaptive_avg_pool2d(x,pool_size))
    target_model=PureSequential(
        PureConv2d(1, 6, 5),
        PureReLU(),
        PureAvgPool2d(2),
        PureConv2d(6, 16, 5),
        PureReLU(),
        PureAvgPool2d(2),
        PureConv2d(16, 120, 5),
        PureReLU(),
        # Neck
        PureAdaptiveAvgPool2d((1, 1)),
        PureFlatten(),
        # Head
        PureLinear(120, 84)
    )
    model=MAMLModel(dict(),target_model)
    x_batch=torch.randn(10,1,64,64)
    model(x_batch)