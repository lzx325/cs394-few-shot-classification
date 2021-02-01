from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

HIT_TIMES=0
class ProtoNet(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()

        self.config = config

        # TODO(protonet): your code here
        # Use the same embedder as in LeNet
        self.embedder = nn.Sequential(
            # Body
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 120, 5),
            nn.ReLU(),
            # Neck
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # Head
            nn.Linear(120, 84),
        )
    @staticmethod
    def square_distance(mat1,mat2):
        assert mat1.shape[1]==mat2.shape[1]
        res_list=list()
        for i in range(mat1.shape[0]):
            dist=torch.sum((mat1[i,:].view(1,-1)-mat2)**2,1)
            res_list.append(dist)
        return torch.stack(res_list)

    def forward(self, x_ep_train: Tensor, y_ep_train: Tensor, x_ep_test: Tensor) -> Tensor:
        """
        Arguments:
            - x: images of shape [num_classes * batch_size, c, h, w]
        """
        # Aggregating across batch-size
        num_classes = self.config['training']['num_classes_per_task']
        batch_size = len(x_ep_train) // num_classes # batch_size or shots
        c, h, w = x_ep_train.shape[1:]

        embeddings_ep_train = self.embedder(x_ep_train) # [num_classes * batch_size, dim]
        assert torch.all(y_ep_train>=0) and torch.all(y_ep_train<num_classes)
        prototype_list=list()
        for c in range(num_classes):
            idx=y_ep_train==c
            proto=torch.mean(embeddings_ep_train[idx],0)
            prototype_list.append(proto)
        # TODO(protonet): compute prototypes given the embedding
        prototypes = torch.stack(prototype_list)

        # TODO(protonet): copmute the logits based on embeddings and prototypes similarity
        # You can use either L2-distance or cosine similarity

        embeddings_ep_test=self.embedder(x_ep_test)

        # logits=-ProtoNet.square_distance(embeddings_ep_test,prototypes)
        logits=torch.mm(embeddings_ep_test,prototypes.T)/(torch.norm(embeddings_ep_test,dim=1).view(-1,1)+1e-6)/(torch.norm(prototypes,dim=1).view(1,-1)+1e-6)

        return logits
