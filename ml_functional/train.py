import torch
import pandas as pd
from tqdm import tqdm
import ast

from torch.utils.data import DataLoader
from ecglib.data import EcgDataset
from ecglib.models.model_builder import create_model
from utils.losses import get_loss

import hydra

from omegaconf import DictConfig


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_loader = None
        self.valid_loader = None
        self.model = create_model(model_name="resnet1d18", pathology="AFIB")
        self.criterion = None
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer
        )
        self.epochs = cfg.epochs
        self.device = cfg.device
        self.augmentation = None

    def _init_dataloaders(self, cfg, train_df, valid_df):
        train_loader = self.get_dataset_loader(train_df, cfg)
        valid_loader = self.get_dataset_loader(valid_df, cfg)
        return train_loader, valid_loader

    def get_dataset_loader(
        self,
        df: pd.DataFrame,
        cfg,
    ):
        ecg_dataset = EcgDataset(df, df.target.values, data_type="npz")
        ecg_loader = DataLoader(
            ecg_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=True,
        )
        return ecg_loader

    def train(self):
        for ep in range(self.epochs):
            train_loss, val_loss = self.train_epoch()
            print(train_loss, val_loss)
            self.save_checkpoint()
        print("Training completed!")

    def train_epoch(self):
        train_loss = self.train_fn()
        val_loss = self.eval_fn()

        self.scheduler.step(val_loss)

        return train_loss, val_loss

    def train_fn(self):
        sum_loss = 0
        self.model.train()

        for bi, batch in tqdm(
            enumerate(self.train_loader), total=len(self.train_loader)
        ):
            index, (input, targets) = batch

            inp = input[0].to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inp)

            loss = self.criterion(outputs, targets)

            loss.backward()
            sum_loss += loss.detach().item()

            self.optimizer.step()

        return sum_loss / len(self.train_loader)

    def eval_fn(self):
        self.model.eval()
        sum_loss = 0

        with torch.no_grad():
            for bi, batch in tqdm(
                enumerate(self.valid_loader), total=len(self.valid_loader)
            ):
                index, (input, targets) = batch

                inp = input[0].to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inp)

                loss = self.criterion(outputs, targets)

                sum_loss += loss.detach().item()

        return sum_loss / len(self.valid_loader)

    def save_checkpoint(self):
        checkpoint_dir = self.cfg.checkpoint_path
        checkpoint_path = f"{checkpoint_dir}/12_leads_resnet1d18_AFIB.pt"

        torch.save(self.model.state_dict(), checkpoint_path)


@hydra.main(version_base=None, config_path=".", config_name="config")
def train_model(cfg: DictConfig):
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

    trainer = Trainer(cfg)

    trainer.train_loader, trainer.valid_loader = trainer._init_dataloaders(
        cfg, train_df, valid_df
    )
    trainer.criterion = get_loss(
        device=trainer.device,
        df=train_df,
    )
    trainer.train()


if __name__ == "__main__":
    train_model()
