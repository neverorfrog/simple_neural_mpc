import torch
from torch.utils.data import Dataset


class TensorDataset(Dataset):
    def __init__(self, data=None, labels=None):
        self.data: torch.Tensor = data
        self.labels: torch.Tensor = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        return sample, label

    def reshape(self, new_shape):
        data = self.data.reshape(new_shape)
        return data

    def get_range(self) -> torch.Tensor:
        x_max, x_min = self.labels[:, 0].max(), self.labels[:, 0].min()
        return torch.tensor([x_min, x_max])
