import torch
from torch.utils.data import Dataset as TorchDataset
import numpy as np

class DNdataset(TorchDataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

def generate_data(num_samples=1000, num_classes=10, img_size=(256, 256)):
    data = np.random.rand(num_samples, 1, *img_size).astype(np.float32)
    labels = np.random.randint(0, num_classes, num_samples)
    return data, labels
