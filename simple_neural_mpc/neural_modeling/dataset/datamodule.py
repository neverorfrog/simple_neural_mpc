import os
from abc import ABC
from typing import List, Optional

import lightning as L
import torch
from torch.utils.data import DataLoader

from simple_neural_mpc.config import DatasetConfig as config
from simple_neural_mpc.neural_modeling.dataset.tensor_dataset import (
    TensorDataset,
)
from simple_neural_mpc.utils import project_root

projroot = project_root()
root = f"{projroot}/data"


class Datamodule(ABC, L.LightningDataModule):
    """The abstract class for handling datasets"""

    def __init__(
        self,
        dataset: Optional[TensorDataset],
        savedpath=None,
    ):
        self.dataset = dataset
        self.name = config.name
        if dataset is None:
            self.load(savedpath)
        else:
            self.train_data, self.val_data, self.test_data = self.random_split(
                dataset, config.ratios
            )
            self.save(savedpath)

    def train_dataloader(self) -> DataLoader:
        """
        Returns a training dataloader with a specified batch size.

        Args:
            batch_size (int): The number of samples in each batch.

        Returns:
            torch.utils.data.DataLoader: The training dataloader.
        """
        return self._get_dataloader(self.train_data, config.batch_size, False)

    def val_dataloader(self) -> DataLoader:
        """
        Creates a validation data loader with the given batch size.

        Args:
            batch_size (int): The size of each batch.

        Returns:
            torch.utils.data.DataLoader: The validation data loader.
        """
        return self._get_dataloader(self.val_data, config.batch_size, False)

    def test_dataloader(self) -> DataLoader:
        """
        A function to create a test data loader with the given batch size.

        Args:
            self: The object instance
            batch_size (int): The size of the batch for the data loader

        Returns:
            DataLoader: The test data loader
        """
        return self._get_dataloader(self.test_data, config.batch_size, False)

    def _get_dataloader(
        self, dataset: TensorDataset, batch_size: int, use_weighting: bool
    ):
        """
        A function to get a DataLoader with optional weighted sampling.

        Parameters:
            dataset (Dataset): The dataset to load.
            batch_size (int): The batch size for the DataLoader.
            use_weighting (bool): Flag to enable weighted sampling.

        Returns:
            DataLoader: A PyTorch DataLoader object.
        """
        # Dataloader stuff
        g = torch.Generator()
        g.manual_seed(2000)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda batch: dataset.collate(batch),
            shuffle=False,
            num_workers=12,
            generator=g,
        )

    def random_split(
        self, data: TensorDataset, ratios: List[float]
    ) -> List[TensorDataset]:
        """
        Randomly splits the dataset into the given ratios
        """
        assert sum(ratios) == 1.0

        n_samples = len(data)
        indices = list(range(n_samples))

        split_indices = [int(ratio * n_samples) for ratio in ratios]
        # Making sure there are no leftovers due to fp32 precision
        split_indices[-1] = n_samples - sum(split_indices[:-1])

        start_idx = 0
        splitted_datasets = []

        for size in split_indices:
            end_idx = start_idx + size
            subset_indices = indices[start_idx:end_idx]
            splitted_datasets.append(
                TensorDataset(data.data[subset_indices], data.labels[subset_indices])
            )
            start_idx = end_idx

        return splitted_datasets

    def __len__(self):
        return len(self.train_data)

    def save(self, path=None):
        if path is None:
            return
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.train_data, open(os.path.join(path, "train_data.dat"), "wb"))
        torch.save(self.val_data, open(os.path.join(path, "val_data.dat"), "wb"))
        torch.save(self.test_data, open(os.path.join(path, "test_data.dat"), "wb"))
        print("DATA SAVED!")

    def load(self, path=None):
        self.train_data = torch.load(open(os.path.join(path, "train_data.dat"), "rb"))
        self.val_data = torch.load(open(os.path.join(path, "val_data.dat"), "rb"))
        self.test_data = torch.load(open(os.path.join(path, "test_data.dat"), "rb"))
        print("DATA LOADED!\n")
