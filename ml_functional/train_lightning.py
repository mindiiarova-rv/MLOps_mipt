import torch
import pandas as pd
from tqdm import tqdm
import ast
import mlflow

from torch.utils.data import DataLoader
from ecglib.data import EcgDataset
from ecglib.models.model_builder import create_model

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl


class EcgLightningModule(pl.LightningModule):
    def __init__(self, cfg, model_name="resnet1d18", pathology="AFIB"):
        super().__init__()
        self.cfg = cfg
        self.model = create_model(model_name=model_name, pathology=pathology)
        self.criterion = torch.nn.BCEWithLogitsLoss(
            size_average=True,
            reduce=True,
        )

    def forward(self, input):
        return self.model(input)

    def training_step(self, batch, batch_idx):
        index, (input, targets) = batch
        inp = input[0].to(self.device)
        targets = targets.to(self.device)
        outputs = self.model(inp)
        loss = self.criterion(outputs, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        mlflow.log_metric("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        index, (input, targets) = batch
        inp = input[0].to(self.device)
        targets = targets.to(self.device)
        outputs = self.model(inp)
        loss = self.criterion(outputs, targets)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        mlflow.log_metric("val_loss", loss)
        return {"val_loss": loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss)
        self.log("val_loss", avg_loss)  # Explicitly log the val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
        return {"optimizer": optimizer, "scheduler": scheduler, "monitor": "val_loss"}


class EcgLightningDataModule(pl.LightningDataModule):
    def __init__(self, cfg, train_df=None, valid_df=None):
        super().__init__()
        self.cfg = cfg
        self.train_df = train_df
        self.valid_df = valid_df

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = EcgDataset(
                self.train_df, self.train_df.target.values, data_type="npz"
            )
            self.valid_dataset = EcgDataset(
                self.valid_df, self.valid_df.target.values, data_type="npz"
            )

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
    loggers = [
        pl.loggers.CSVLogger("./.logs/my-csv-logs", name=cfg.experiment_name),
        pl.loggers.MLFlowLogger(
            experiment_name=cfg.experiment_name,
            # tracking_uri="file:./.logs/my-mlflow-logs",
            tracking_uri="http://localhost:5000",  # до этого в терминале mlflow server --host 127.0.0.1 --port 5000
        ),
    ]

    train_df = pd.read_csv(cfg.train_path)
    valid_df = pd.read_csv(cfg.valid_path)
    targets_train = [
        [0.0] if "AFIB" in ast.literal_eval(train_df.iloc[i]["scp_codes"]) else [1.0]
        for i in range(train_df.shape[0])
    ]

    train_df["target"] = targets_train

    targets_valid = [
        [0.0] if "AFIB" in ast.literal_eval(valid_df.iloc[i]["scp_codes"]) else [1.0]
        for i in range(valid_df.shape[0])
    ]

    valid_df["target"] = targets_valid

    # ... преобразование данных DataFrame ...

    datamodule = EcgLightningDataModule(cfg, train_df, valid_df)
    model = EcgLightningModule(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        progress_bar_refresh_rate=20,
        checkpoint_callback=True,
        logger=loggers,
    )
    with mlflow.start_run():
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train_model()
