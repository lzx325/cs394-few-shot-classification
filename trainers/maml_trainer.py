from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .trainer import Trainer
from models import init_model
import torch
import torch.autograd
from tqdm import tqdm

class MAMLTrainer(Trainer):
    def train(self):
        episodes = tqdm(range(self.config['training']['num_train_episodes']))
        for ep in episodes:
            ds_train, ds_test = self.source_dataloader.sample_random_task()
            ep_train_loss, ep_train_acc, ep_test_loss, ep_test_acc = self.train_on_episode(self.model, self.optim, ds_train, ds_test, return_test_metrics=True) 
            # print("repeat %d: "%(i+1), "ep_test_loss: ", ep_test_loss, "ep_test_acc: ", ep_test_acc)
            episodes.set_description(f'[Episode {ep: 03d}] Loss: {ep_train_loss: .3f}. Acc: {ep_train_acc: .03f}')
    def train_on_episode(self, model, optim, ds_train, ds_test,mode="train",return_test_metrics=False):
        losses = []
        accs = []

        model.train()
        initial_params=model.params
        fast_w=model.params
        assert mode in ("train","test")
        
        for it in range(self.config['model']['num_inner_steps']):
            x, y = self.sample_batch(ds_train)

            x = x.to(self.config['device'])
            y = y.to(self.config['device'])

            # TODO(maml): perform forward pass, compute logits, loss and accuracy
            logits = model(x)
            loss = F.cross_entropy(logits,y)
            acc = (y==logits.argmax(1)).float().mean()

            # TODO(maml): compute the gradient and update the fast weights
            # Hint: think hard about it. This is maybe the hardest part of the assignment
            # You will likely need to check open-source implementations to get the idea of how things work
            grad = torch.autograd.grad(loss,fast_w,create_graph=True,retain_graph=True)
            assert type(grad)==tuple
            grad=grad[0]
            fast_w = fast_w-self.config['training']['inner_lr']*grad
            model.params=fast_w
            # print("inner loop loss: ", loss.item(),"inner loop acc: ", acc.item())
            losses.append(loss.item())
            accs.append(acc.item())

        x = torch.stack([s[0] for s in ds_test]).to(self.config['device'])
        y = torch.stack([s[1] for s in ds_test]).to(self.config['device'])

        # TODO(maml): compute the logits, outer-loop loss and outer-loop accuracy
        logits = model(x)
        outer_loss = F.cross_entropy(logits,y)
        outer_acc = (y==logits.argmax(1)).float().mean()

        if mode=="train":
            optim.zero_grad()
            outer_loss.backward()
            # TODO(maml): you may like to add gradient clipping here
            # print(initial_params.grad)
            optim.step()
            model.params=initial_params
            # print("initial_params after update:",initial_params[:10])
        
        if return_test_metrics:
            return losses[-1],accs[-1],outer_loss.item(),outer_acc.item()
        else:
            return losses[-1],accs[-1]

    def fine_tune(self, ds_train, ds_test) -> Tuple[float, float]:
        curr_model = init_model(self.config).to(self.config['device'])
        curr_model.params.data.copy_(self.model.params.data)
        curr_optim = torch.optim.Adam(curr_model.parameters(), **self.config['training']['optim_kwargs'])

        train_loss, train_acc, test_loss, test_acc = self.train_on_episode(curr_model, curr_optim, ds_train, ds_test,"test",return_test_metrics=True)

        return (train_loss, train_acc), (test_loss, test_acc)
