import torch
import pandas as pd
from tqdm import tqdm
import ast
import mlflow

from torch.utils.data import DataLoader
from ecglib.data import EcgDataset
from ecglib.models.model_builder import create_model
from utils.losses import get_loss

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl

class EcgLightningModule(pl.LightningModule):
    def __init__(self, cfg, model_name='resnet1d18', pathology='AFIB'):
        super().__init__()
        self.cfg = cfg
        self.model = create_model(model_name, pathology)
        self.criterion = get_loss(device=self.cfg.device, df=None)

    def forward(self, input):
        return self.model(input)

    def training_step(self, batch, batch_idx):
        index, (input, targets) = batch
        inp = input[0].to(self.device)
        targets = targets.to(self.device)
        outputs = self.model(inp)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        index, (input, targets) = batch
        inp = input[0].to(self.device)
        targets = targets.to(self.device)
        outputs = self.model(inp)
        loss = self.criterion(outputs, targets)
        self.log('val_loss', loss)

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
        return [optimizer], [scheduler]

class EcgLightningDataModule(pl.LightningDataModule):
    def __init__(self, cfg, train_df=None, valid_df=None):
        super().__init__()
        self.cfg = cfg
        self.train_df = train_df
        self.valid_df = valid_df

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = EcgDataset(self.train_df, self.train_df.target.values)
            self.valid_dataset = EcgDataset(self.valid_df, self.valid_df.target.values)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            drop_last=True,
        )

@hydra.main(version_base=None, config_path=".", config_name="config")
def train_model(cfg: DictConfig):
    train_df = pd.read_csv(cfg.train_path)
    valid_df = pd.read_csv(cfg.valid_path)

    datamodule = EcgLightningDataModule(cfg, train_df, valid_df)
    model = EcgLightningModule(cfg)

    trainer = pl.Trainer(
        max_epochs=10,
        gpus=cfg.gpus,
        progress_bar_refresh_rate=20,
        checkpoint_callback=True,
        checkpoint_dir=cfg.checkpoint_path,
    )
    
    mlflow_callback = MlflowLoggingCallback(
        tracking_uri="http://localhost:5000",
        experiment_name="my_experiment"
    )
    
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    train_model()

class MlflowLoggingCallback(pl.Callback):
    def __init__(self, tracking_uri, experiment_name):
        super().__init__()
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run = None

    def on_fit_start(self, trainer, pl_module):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run()

    def on_train_epoch_end(self, trainer, pl_module):
        mlflow.log_metric("train_loss", pl_module.trainer.callback_metrics["train_loss"])

    def on_validation_epoch_end(self, trainer, pl_module):
        mlflow.log_metric("val_loss", pl_module.trainer.callback_metrics["val_loss"])
        mlflow.log_metric("avg_val_loss", pl_module.trainer.callback_metrics["avg_val_loss"])

    def on_fit_end(self, trainer, pl_module):
        mlflow.end_run()