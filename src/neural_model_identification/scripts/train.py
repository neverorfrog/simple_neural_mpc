import os

import numpy as np
import torch
from tqdm import tqdm

from neural_model_identification.data_generation.preprocess import (
    DataPreProcess,
)
from neural_model_identification.learner.mlp_learner import Learner
from neural_model_identification.parameters.train_params import TrainParams

torch.manual_seed(0xDEADBEEF)
np.random.seed(0xDEADBEEF)

try:
    dataset_tensor = torch.load(os.path.join(TrainParams.data_path, "dataset.pt"))
    print("Loaded dataset from file")
except FileNotFoundError:
    dataset_tensor, features = DataPreProcess().run()
    torch.save(dataset_tensor, os.path.join(TrainParams.data_path, "dataset.pt"))

# The batch will be of shape [batch_size, n_data_points, [state_dim + action_dim]]
loader = torch.utils.data.DataLoader(
    dataset_tensor, batch_size=TrainParams.batch_size, shuffle=True
)

learner = Learner()

with tqdm(total=len(loader)) as pbar:
    for i, batch in enumerate(loader):
        loss = learner.train_step(batch)
        if i % 100 == 0:
            pbar.set_description(f"LOSS: {loss:.5f}")
        pbar.update(1)

print("Training ended .............")
learner.save()
