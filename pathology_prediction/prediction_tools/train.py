import torch
import pandas as pd
from tqdm import tqdm
import sys

from torch.utils.data import DataLoader
from ecglib.data import EcgDataset
from hydra.utils import instantiate
from utils.model_utils import get_model
from utils.metrics_utils import (
    metrics_report,
    select_best_validation_threshold,
    stopping_criterion,
)
import hydra

# import wandb
from omegaconf import DictConfig

import sys
from hydra.core.hydra_config import HydraConfig



class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.leads_num = 12
        self.pathology_names = "AFIB"
        self.train_loader = None
        self.valid_loader = None
        self.model = get_model(cfg, self.leads_num)
        self.criterion = None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.training_params.learning_rate, weight_decay=self.cfg.training_params.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=0.8,
            patience=3,
            verbose=True,
        )
        self.best_metrics = {metric: 1000 * (metric == "loss") for metric in ["loss"]}
        self.epochs = 10
        self.epochs_no_improve = 0
        self.prediction_threshold = 0.5
        self.early_stopping_patience = 10
        self.tensor_logger = None
        self.device = "cpu"
        self.augmentation = None

    def _init_dataloaders(self, cfg, train_df, valid_df):
        train_loader = self.get_dataset_loader(
            train_df, cfg, self.augmentation, cfg.task_params.classes
        )
        valid_loader = self.get_dataset_loader(
            valid_df, cfg, self.augmentation, cfg.task_params.classes
        )
        return train_loader, valid_loader


    def get_dataset_loader(
        self,
        ecg_info: pd.DataFrame,
        cfg,
        classes_num,
        augmentation=None
    ):
        ecg_target = ecg_info.target.values
        ecg_dataset = EcgDataset.for_train_from_config(
            ecg_info, ecg_target, augmentation, cfg, classes_num
        )
        ecg_loader = DataLoader(
            ecg_dataset,
            batch_size=cfg.training_params.batch_size,
            shuffle=True,
            num_workers=cfg.training_params.num_workers,
            drop_last=True,
        )
        return ecg_loader

    def train(self):

        # ====================================== Train the model
        for ep in range(self.epochs):
            train_loss, val_loss, metrics = self.train_epoch()

            self.save_checkpoint(val_loss / len(self.valid_loader), metrics)

            if self.epochs_no_improve >= self.early_stopping_patience:
                print("Early stopping")
                break

    def train_epoch(self):
        train_loss = self.train_fn()
        val_loss, fin_targets, fin_outputs = self.eval_fn()
        metrics = self.calculate_metrics(fin_targets, fin_outputs)
        self.scheduler.step(val_loss)

        return train_loss, val_loss, metrics

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

            loss = self.get_loss_value(self.criterion, outputs, targets)

            loss.backward()
            sum_loss += loss.detach().item()

            self.optimizer.step()

        return sum_loss

    def eval_fn(self):
        self.model.eval()
        sum_loss = 0
        fin_targets = []
        fin_outputs = []

        with torch.no_grad():
            for bi, batch in tqdm(
                enumerate(self.valid_loader), total=len(self.valid_loader)
            ):
                index, (input, targets) = batch

                inp = input[0].to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inp)

                loss = self.get_loss_value(self.criterion, outputs, targets)

                sum_loss += loss.detach().item()

                fin_targets.extend(targets.tolist())
                fin_outputs.extend(outputs.tolist())

        return sum_loss, fin_targets, fin_outputs

    def get_loss_value(self, criterion, outputs, targets):
        return criterion(outputs, targets)

    def calculate_metrics(self, fin_targets, fin_outputs):
        sigmoid = torch.nn.Sigmoid()

        fin_outputs = sigmoid(torch.as_tensor(fin_outputs))
        self.prediction_threshold = select_best_validation_threshold(
            fin_targets, fin_outputs, self.cfg.training_params.metrics_threshold
        )
        results = (fin_outputs > self.prediction_threshold).float()
        metrics, _ = metrics_report(
            fin_targets, results.tolist(), self.pathology_names, fin_outputs
        )
        return metrics

    def save_checkpoint(self, val_loss, metrics):
        self.epochs_no_improve, self.best_metrics = stopping_criterion(
            val_loss, metrics, self.best_metrics, self.epochs_no_improve
        )
        if self.epochs_no_improve == 0:
            print("Model saved!")
            # Create run command for experiment report
            model_info = {
                "model": self.model.state_dict(),
                "config_file": self.cfg,
                "threshold": self.prediction_threshold,
            }
            checkpoint_dir = self.cfg.single_run_dir

            pathology_name_to_save = self.pathology_names

            checkpoint_path = f"{checkpoint_dir}/12_leads_resnet1d18_{pathology_name_to_save}.pt"

            torch.save(model_info, checkpoint_path)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train_model(cfg: DictConfig):


if __name__ == "__main__":
