import os
import time
from pathlib import Path

import torch


class MWERecorder:

    def __init__(self, config) -> None:
        self.config = config

        self.best_acc = 0.0
        self.best_epoch_idx = 0

        self.begin_time = time.time()
        self.output_dir = config.output_dir


    def report(self, train_metrics, val_metrics):
        print('\nEpoch {:03d} | Time {:5d}s | Train Loss {:.4f} | '
              'Weights {:.7f} | Val Loss {:.3f} | Val Acc {:.2f}'
              ' | AUROC {:.4f}'.format(
                  (train_metrics['epoch_idx']),
                  int(time.time() - self.begin_time), train_metrics['loss'],
                  train_metrics["weights"], val_metrics['loss'],
                  100.0 * val_metrics['acc'], 100.0 * val_metrics['image_auroc']),
              flush=True)
        # print('Time {:5d}s'.format(int(time.time() - self.begin_time)), flush=True)

    def save_model(self, net, val_metrics):
        # update the best model
        torch.save(net.state_dict(),
        os.path.join(self.output_dir, 'best.ckpt'))


    def summary(self):
        print('Training Completed! '
              'Best accuracy: {:.2f} '
              'at epoch {:d}'.format(100 * self.best_acc, self.best_epoch_idx),
              flush=True)
