import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config

from .lr_scheduler import cosine_annealing


def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()
    return model


def freeze_noMWE(model):
    for module in model.modules():
        if not "Custom" in module.__class__.__name__:
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.requires_grad_(False)
            module.eval()
    return model


class MaxWEnt:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:

        self.net = net
        self.train_loader = train_loader
        self.config = config
        
        self.optimizer = torch.optim.Adam(
            net.parameters(),
            config.optimizer.lr,
        )
        

    def train_epoch(self, epoch_idx):
        self.net.train()
        self.net = freeze_bn(self.net)
        self.net = freeze_noMWE(self.net)
        
        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1, len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['label'].cuda()

            # forward
            logits_classifier = self.net(data)
            loss = F.cross_entropy(logits_classifier, target)
            
            weights_dist = 0.
            count = 0.
            for w in filter(lambda p: p.requires_grad, self.net.parameters()):
                weights_dist += torch.sum(torch.abs(w))
                count += torch.sum(torch.ones_like(w))
                    
            weights_dist /= count
            weights_dist *= 10.
            
            final_loss = loss - weights_dist

            # backward
            self.optimizer.zero_grad()
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.)
            self.optimizer.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2
                weights_avg = float(weights_dist)

        # comm.synchronize()
        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)
        metrics['weights'] = weights_avg

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])
        return total_losses_reduced
    
    
    def setup(self):
        print("Setup")
