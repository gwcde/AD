import os.path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data.dataset import Dataset

SAMPLE_RATE = 16000
EXAMPLE_WAV_MIN_SEC = 5
EXAMPLE_WAV_MAX_SEC = 20
EXAMPLE_DATASET_SIZE = 200

torchaudio.set_audio_backend("soundfile")

SAMPLE_RATE = 16000
EXAMPLE_WAV_MAX_SEC = 60
MAX_WAV_LENTH = 960000


def get_label(label_str, classes):
    return classes.index(label_str)


class MadressDataset(Dataset):
    def __init__(self, **kwargs):
        self.class_num = 2
        self.df = pd.read_csv('/home/shaoz/projects/collaborate/spgc/AD/MADReSS-23-train/training-groundtruth.csv')
        self.data_path = '/home/shaoz/projects/collaborate/spgc/AD/data/train/'
        self.classes = list(self.df['dx'].unique())
        self.classes.sort()


    def __getitem__(self, idx):
        item = self.df.loc[idx, :]
        wav, sample_rate = torchaudio.load(os.path.join(self.data_path, item['adressfname']+'.wav'))
        l = wav.shape[1]
        if l > MAX_WAV_LENTH:
            wav = wav[:, :MAX_WAV_LENTH]
        else:
            ll = MAX_WAV_LENTH - l
            wav = np.pad(wav, [(0, 0), (0, ll)], mode='constant', constant_values=0)
        print('audio length', l/SAMPLE_RATE, l, wav.shape)
        wav = wav[0, :]
        label = get_label(item['dx'], self.classes)
        return wav, label


    def __len__(self):
        return self.df.shape[0]


    def collate_fn(self, samples):
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels
