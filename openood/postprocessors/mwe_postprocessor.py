from typing import Any
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class MWEPostprocessor:
    
    def __init__(self, config):
        self.config = config
        self.postprocessor_args = config.postprocessor.postprocessor_args
        print(self.postprocessor_args)
        self.n_preds = self.postprocessor_args.n_preds

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        net.eval()

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        outputs = []
        for _ in range(self.n_preds):
            output = net(data)
            outputs.append(torch.softmax(output, dim=1))
        outputs = torch.stack(outputs, dim=-1)
        
        output = torch.mean(outputs, dim=-1)
        conf = torch.sum(output * torch.log(output + 1e-8), dim=1)

        _, pred = torch.max(torch.mean(outputs, dim=-1), dim=1)
        return pred, conf

    def inference(self, net: nn.Module, data_loader: DataLoader):
        pred_list, conf_list, label_list = [], [], []
        data_loader = iter(data_loader)
        for test_step in tqdm(range(1, len(data_loader) + 1),
                               desc='Test: ',
                               position=0,
                               leave=True):
            batch = next(data_loader)
            data = batch['data'].cuda()
            label = batch['label'].cuda()
            pred, conf = self.postprocess(net, data)
            for idx in range(len(data)):
                pred_list.append(pred[idx].cpu().tolist())
                conf_list.append(conf[idx].cpu().tolist())
                label_list.append(label[idx].cpu().tolist())

        # convert values into numpy array
        pred_list = np.array(pred_list, dtype=int)
        conf_list = np.array(conf_list)
        label_list = np.array(label_list, dtype=int)

        return pred_list, conf_list, label_list
