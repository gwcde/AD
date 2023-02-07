import os
import math
import torch
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

from ..model import *
from .model import *
from .dataset import IEMOCAPDataset, collate_fn
import yaml

import csv

import pandas as pd




def parse_results(s):
    s = s.replace('[', '')
    s = s.replace(']', '')
    l = s.split()
    if len(l) == 2:
        n = l[1]
        return float(n)

    return 0



class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    Args:
                upstream_dim: int
                    Different upstream will give different representation dimension
                    You might want to first project them to the same dimension
                downstream_expert: dict
                    The 'downstream_expert' field specified in your downstream config file
                    eg. downstream/example/config.yaml
                expdir: string
                    The expdir from command-line argument, you should save all results into
                    this directory, like some logging files.
                **kwargs: dict
                    All the arguments specified by the argparser in run_downstream.py
                    and all the other fields in config.yaml, in case you need it.

                    Note1. Feel free to add new argument for __init__ as long as it is
                    a command-line argument or a config field. You can check the constructor
                    code in downstream/runner.py      
    """
    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim

        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']

        TRAIN_DATA_ROOT = self.datarc['train_root']
        TEST_DATA_ROOT = self.datarc['test_root']
        meta_data = self.datarc["meta_data"]


        self.fold = self.datarc.get('test_fold') or kwargs.get("downstream_variant")
        if self.fold is None:
            self.fold = "fold1"

        print(f"[Expert] - using the testing fold: \"{self.fold}\". Ps. Use -o config.downstream_expert.datarc.test_fold=fold2 to change test_fold in config.")

        train_path = self.datarc['train_path']
        print(f'[Expert] - Training path: {train_path}')

        test_path = self.datarc['test_path']
        print(f'[Expert] - Testing path: {test_path}')
        
        dataset = IEMOCAPDataset(TRAIN_DATA_ROOT, train_path, self.datarc['pre_load'])
        trainlen = int((1 - self.datarc['valid_ratio']) * len(dataset))
        lengths = [trainlen, len(dataset) - trainlen]
        
        torch.manual_seed(0)
        self.train_dataset, self.dev_dataset = random_split(dataset, lengths)

        self.test_dataset = IEMOCAPDataset(TEST_DATA_ROOT, test_path, self.datarc['pre_load'])

        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        print(model_conf)
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim = self.modelrc['projector_dim'],
            output_dim = 1,
            **model_conf,
        )
        self.objective = nn.MSELoss()
        self.expdir = expdir
        self.register_buffer('best_score', torch.zeros(1))

    def get_downstream_name(self):
        return self.fold.replace('fold', 'emotion')


    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'],
            shuffle=(sampler is None), sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=collate_fn
        )

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    # Interface
    def get_dataloader(self, mode):
        return eval(f'self.get_{mode}_dataloader')()

    # Interface
    def forward(self, mode, features, labels, filenames, records, **kwargs):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)

        features = pad_sequence(features, batch_first=True)
        features = self.projector(features)
        predicted, _ = self.model(features, features_len)

        labels = torch.FloatTensor(labels).to(features.device)
        loss = self.objective(predicted, labels)

        records['mse'] += sum([i**2 for i in (predicted - labels)])
        records['loss'].append(loss.item())

        records["filename"] += filenames
        records["predict"] += predicted.cpu().tolist()
        records["truth"] += labels.cpu().tolist()

        return loss


    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
        def get_metrics(predict_path, truth_path):
            with open(predict_path) as prediction_file, open(truth_path) as label_file:
                predictions = prediction_file.readlines()
                labels = label_file.readlines()

                predicted = []
                truth = []
                for p,t in zip(predictions, labels):
                    predicted.append(parse_results(p))
                    truth.append(parse_results(t))

            mae = sum([abs(p-t) for p,t in zip(predicted, truth)]) / len(predictions)
            mse = sum([(p-t)*(p-t) for p,t in zip(predicted, truth)]) / len(predictions)

            return mae, mse

        save_names = []
        for key in ["acc", "loss"]:
            values = records[key]
            average = torch.FloatTensor(values).mean().item()
            logger.add_scalar(
                f'emotion-{self.fold}/{mode}-{key}',
                average,
                global_step=global_step
            )
            with open(Path(self.expdir) / "log.log", 'a') as f:
                if key == 'acc':
                    print(f"{mode} {key}: {average}")
                    f.write(f'{mode} at step {global_step}: {average}\n')
                    if mode == 'dev' and average > self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(f'New best on {mode} at step {global_step}: {average}\n')
                        save_names.append(f'{mode}-best.ckpt')

                if key == 'loss':
                    print(f"{mode} loss: {average}")
                    f.write(f'{mode} loss at step {global_step}: {average}\n')
                    #if mode == 'dev' and average > self.best_score:
                        #self.best_score = torch.ones(1) * average
                        #f.write(f'New best on {mode} at step {global_step}: {average}\n')
                        #save_names.append(f'{mode}-best.ckpt')


        if mode in ["dev", "test"]:
            with open(Path(self.expdir) / f"{mode}_{self.fold}_predict.txt", "w") as file:
                line = [f"{f} {e}\n" for f, e in zip(records["filename"], records["predict"])]
                file.writelines(line)

            with open(Path(self.expdir) / f"{mode}_{self.fold}_truth.txt", "w") as file:
                line = [f"{f} {e}\n" for f, e in zip(records["filename"], records["truth"])]
                file.writelines(line)
            if mode == "test":
                # F1
                predict_path = Path(self.expdir) / f"{mode}_{self.fold}_predict.txt"
                truth_path = Path(self.expdir) / f"{mode}_{self.fold}_truth.txt"

                mae, mse = get_metrics(predict_path, truth_path)

                with open(Path(self.expdir) / "log.log", 'a') as f:
                    f.write(f'MAE: {mae}, MSE: {mse}, \n\n')

        return save_names
