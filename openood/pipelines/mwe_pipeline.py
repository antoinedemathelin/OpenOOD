from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.recorders import get_recorder
from openood.trainers import get_trainer
from openood.utils import setup_logger
from openood.postprocessors import get_postprocessor
import torch
import os


class MWEPipeline:
    
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        loader_dict = get_dataloader(self.config)
        ood_data_loaders = get_ood_dataloader(self.config)
        train_loader, val_loader = loader_dict['train'], loader_dict['val']
        test_loader = loader_dict['test']
        
        # init postprocessor
        postprocessor = get_postprocessor(self.config)

        # init network
        net = get_network(self.config.network)

        # init trainer and evaluator
        trainer = get_trainer(net, train_loader, self.config)
        evaluator = get_evaluator(self.config)

        # init recorder
        recorder = get_recorder(self.config)

        # trainer setup
        trainer.setup()
        print('\n' + '#' * 70, flush=True)

        print('Start training...', flush=True)
        for epoch_idx in range(1, self.config.optimizer.num_epochs + 1):
            # train and eval the model
            net, train_metrics = trainer.train_epoch(epoch_idx)
            val_metrics = evaluator.eval_acc(net, val_loader, postprocessor, epoch_idx)
            ood_val_metrics = evaluator.eval_ood(net, loader_dict, ood_data_loaders,
                                                 postprocessor)
            val_metrics.update(ood_val_metrics)
            # save model and report the result
            recorder.save_model(net, val_metrics)
            recorder.report(train_metrics, val_metrics)
            
            # print(train_metrics)
            # recorder.save_model(net, train_metrics)
        recorder.summary()
        print('#' * 70, flush=True)

        # evaluate on test set
        print('Start testing...', flush=True)
        test_metrics = evaluator.eval_acc(net, test_loader)
        print('\nComplete Evaluation, accuracy {:.2f}'.format(
            100.0 * test_metrics['acc']),
              flush=True)
        print('Completed!', flush=True)
