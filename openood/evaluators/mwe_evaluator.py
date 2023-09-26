import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.postprocessors import BasePostprocessor
from openood.utils import Config

from sklearn.metrics import auc, roc_curve


class MWEEvaluator():
    def __init__(self, config: Config):
        self.config = config

    def eval_ood(self,
                 net,
                 id_data_loader,
                 ood_data_loaders,
                 postprocessor,
                 epoch_idx: int = -1):
        with torch.no_grad():
            if type(net) is dict:
                for subnet in net.values():
                    subnet.eval()
            else:
                net.eval()
            auroc = self.get_auroc(net, id_data_loader['val'],
                                   ood_data_loaders['val'], postprocessor)
            metrics = {
                'epoch_idx': epoch_idx,
                'image_auroc': auroc,
            }
            return metrics

    def get_auroc(self, net, id_data_loader, ood_data_loader, postprocessor):
        _, id_conf, id_gt = postprocessor.inference(net, id_data_loader)
        _, ood_conf, ood_gt = postprocessor.inference(net, ood_data_loader)
        ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood

        conf = np.concatenate([id_conf, ood_conf])
        label = np.concatenate([id_gt, ood_gt])

        ind_indicator = np.zeros_like(label)
        ind_indicator[label != -1] = 1

        fpr, tpr, _ = roc_curve(ind_indicator, conf)

        auroc = auc(fpr, tpr)

        return auroc
    
    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1):
        net.eval()

        loss_avg = 0.0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(data_loader,
                              desc='Eval: ',
                              position=0,
                              leave=True):
                # prepare data
                data = batch['data'].cuda()
                target = batch['label'].cuda()

                # forward
                output = net(data)
                loss = F.cross_entropy(output, target)

                # accuracy
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

                # test loss average
                loss_avg += float(loss.data)

        loss = loss_avg / len(data_loader)
        acc = correct / len(data_loader.dataset)

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss)
        metrics['acc'] = self.save_metrics(acc)
        return metrics
    
    def save_metrics(self, value):
        all_values = comm.gather(value)
        temp = 0
        for i in all_values:
            temp = temp + i
        return temp
