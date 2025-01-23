import os
from typing import Optional

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from simple_neural_mpc.config.neural_config import TrainerConfig as config


class Trainer(L.Trainer):  # type: ignore[misc]
    def __init__(self, group: Optional[str] = None) -> None:

        wandb.login(key=config.wandb_api_key)  # type: ignore
        wandb.init(project=config.wandb_project, group=group)  # type: ignore

        self._checkpoint_callback = SaveBestModel(
            monitor="val/loss",
            dirpath=config.ckpt_path,
            mode="min",
            filename="unicycle.pth"
        )

        early_stopping_callback = EarlyStopping(
            monitor="val/loss",
            patience=config.patience,
            min_delta=config.min_delta,
            verbose=True,
        )

        super().__init__(
            max_epochs=config.max_epochs,
            callbacks=[self._checkpoint_callback, early_stopping_callback],
            logger=WandbLogger(),
        )

    def fit(self, model: L.LightningModule, datamodule: L.LightningDataModule) -> None:
        """Train the model, resuming a previous checkpoint if needed."""

        checkpoint_path = f"{config.ckpt_path}/last.ckpt"

        if config.resume_training and os.path.exists(checkpoint_path):
            ckpt_path = checkpoint_path
        else:
            ckpt_path = None

        super().fit(
            model=model,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader(),
            ckpt_path=ckpt_path,
        )

    def test(self, model: L.LightningModule, datamodule: L.LightningDataModule) -> None:
        """Test the model on the test set."""
        best_ckpt_path = self._checkpoint_callback.best_model_path

        if not best_ckpt_path:
            raise ValueError(
                "Best model path not found. \
                Make sure the model was trained and saved."
            )

        super().test(
            model=model,
            dataloaders=datamodule.test_dataloader(),
            ckpt_path=best_ckpt_path,
        )
        
        
class SaveBestModel(Callback):
    def __init__(self, monitor: str, dirpath: str,  mode: str, filename: str):
        super().__init__()
        self.monitor = monitor
        self.dirpath = dirpath
        self.mode = mode
        self.filename = filename
        self.best_score = None

        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        current_score = logs.get(self.monitor)

        if current_score is None:
            return

        if self.best_score is None:
            self.best_score = current_score

        if (self.mode == "min" and current_score < self.best_score) or (self.mode == "max" and current_score > self.best_score):
            self.best_score = current_score
            filepath = os.path.join(self.dirpath, self.filename)
            torch.save(pl_module.state_dict(), filepath)
            print(f"Model saved to {filepath}")
