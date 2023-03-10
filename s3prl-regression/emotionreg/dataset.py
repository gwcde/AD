# -*- coding: utf-8 -*- #
"""
    FileName     [ dataset.py ]
    Synopsis     [ the emotion classifier dataset ]
    Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""

import json
import torch
from pathlib import Path
from os.path import join as path_join
import numpy as np
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Resample

SAMPLE_RATE = 16000
MAX_WAV_LENTH = 1120000

class IEMOCAPDataset(Dataset):
    def __init__(self, data_dir, meta_path, pre_load=True):
        self.data_dir = data_dir
        self.pre_load = pre_load
        import os
        # print(os.getcwd(), meta_path)
        with open(meta_path, 'r') as f:
            self.data = json.load(f)

        self.meta_data = self.data['meta_data']
        self.meta_data = [{'path': item['path'], 'label': item['label'] / 30.0} for item in self.meta_data]
        _, origin_sr = torchaudio.load(
            path_join(self.data_dir, self.meta_data[0]['path']))
        self.resampler = Resample(origin_sr, SAMPLE_RATE)
        if self.pre_load:
            self.wavs = self._load_all()

    def _load_wav(self, path):
        wav, _ = torchaudio.load(path_join(self.data_dir, path))

        l = wav.shape[1]
        if l > MAX_WAV_LENTH:
            wav = wav[:, :MAX_WAV_LENTH]
        else:
            ll = MAX_WAV_LENTH - l
            wav = np.pad(wav, [(0, 0), (0, ll)], mode='constant', constant_values=0)
        wav = self.resampler(wav).squeeze(0)
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        # print(wav.shape, type(wav))
        return wav

    def _load_all(self):
        wavforms = []
        for info in self.meta_data:
            wav = self._load_wav(info['path'])
            wavforms.append(wav)
        return wavforms

    def __getitem__(self, idx):
        label = self.meta_data[idx]['label']
        if self.pre_load:
            wav = self.wavs[idx]
        else:
            wav = self._load_wav(self.meta_data[idx]['path'])
        return wav.numpy(), label, Path(self.meta_data[idx]['path']).stem

    def __len__(self):
        return len(self.meta_data)

def collate_fn(samples):
    return zip(*samples)
