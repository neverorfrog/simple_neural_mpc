from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset


class TensorSample:
    sample: torch.Tensor
    label: torch.Tensor

    def __init__(self, sample: torch.Tensor, label: torch.Tensor):
        self.sample = sample
        self.label = label

    def __repr__(self):
        return f"Sample: {self.sample}, Label: {self.label}"


class TensorDataset(Dataset):
    def __init__(self, data=None, labels=None):
        self.data: torch.Tensor = data
        self.labels: torch.Tensor = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        return TensorSample(sample, label)

    def reshape(self, new_shape):
        data = self.data.reshape(new_shape)
        return data

    def get_range(self) -> torch.Tensor:
        x_max, x_min = self.labels[:, 0].max(), self.labels[:, 0].min()
        return torch.tensor([x_min, x_max])

    def collate(self, batch: List[TensorSample]):
        elem: TensorSample = batch[0]
        assert isinstance(elem, TensorSample), "batch must contain TensorSample objects"

        data = []
        labels = []
        for elem in batch:
            data.append(elem.sample)
            labels.append(elem.label)
        return TensorDataset(torch.stack(data), torch.stack(labels))
