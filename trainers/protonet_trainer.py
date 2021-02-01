from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from .trainer import Trainer
from models import init_model

class ProtoNetTrainer(Trainer):
    def sample_batch(self, dataset):
        """
        In ProtoNet we require that the batch contains equal number of examples
        per each class. We do this so it is simpler to compute prototypes
        """
        k = self.config['training']['num_classes_per_task']
        num_shots = len(dataset) // k
        batch_size = min(self.config['training']['batch_size'], num_shots)

        idx = [(c * num_shots + i) for c in range(k) for i in self.rnd.permutation(num_shots)[:batch_size]] # will sample batch_size*way
        x = torch.stack([dataset[i][0] for i in idx])
        y = torch.stack([dataset[i][1] for i in idx])
        return x, y
    
    def train_on_episode(self, model, optim, ds_train, ds_test):
        losses = []
        accs = []

        model.train()

        for it in range(self.config['training']['num_train_steps_per_episode']):
            x_ep_train, y_ep_train = self.sample_batch(ds_train)
            x_ep_test, y_ep_test = self.sample_batch(ds_test)


            x_ep_train = x_ep_train.to(self.config['device'])
            y_ep_train = y_ep_train.to(self.config['device'])
            x_ep_test = x_ep_test.to(self.config['device'])
            y_ep_test = y_ep_test.to(self.config['device'])

            # import ipdb; ipdb.set_trace()
            logits = model(x_ep_train,y_ep_train,x_ep_test) # [batch_size, num_classes_per_task]
            loss = F.cross_entropy(logits, y_ep_test)
            acc = (logits.argmax(dim=1) == y_ep_test).float().mean()

            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
            accs.append(acc.item())

        return losses[-1], accs[-1]   

    def fine_tune(self, ds_train, ds_test) -> Tuple[float, float]:
        # TODO(protonet): your code goes here
        # How does ProtoNet operate in the inference stage?
        curr_model = init_model(self.config).to(self.config['device'])
        curr_model.load_state_dict(self.model.state_dict())
        curr_optim = torch.optim.Adam(curr_model.parameters(), **self.config['training']['optim_kwargs'])
        # curr_optim = torch.optim.SGD(curr_model.parameters(), **self.config['training']['optim_kwargs'])
        train_scores = self.train_on_episode(curr_model, curr_optim, ds_train, ds_test)
        test_scores = self.compute_scores(curr_model, ds_train, ds_test)
        return train_scores, test_scores

    @torch.no_grad()
    def compute_scores(self, model, ds_ep_train, ds_ep_test) -> Tuple[np.float, np.float]:
        """
        Computes loss/acc for the dataloader
        """
        model.eval()

        x_ep_train = torch.stack([s[0] for s in ds_ep_train]).to(self.config['device'])
        y_ep_train = torch.stack([s[1] for s in ds_ep_train]).to(self.config['device'])
        x_ep_test = torch.stack([s[0] for s in ds_ep_test]).to(self.config['device'])
        y_ep_test = torch.stack([s[1] for s in ds_ep_test]).to(self.config['device'])
        logits = model(x_ep_train,y_ep_train,x_ep_test)
        loss = F.cross_entropy(logits, y_ep_test).item()
        acc = (logits.argmax(dim=1) == y_ep_test).float().mean().item()

        return loss, acc