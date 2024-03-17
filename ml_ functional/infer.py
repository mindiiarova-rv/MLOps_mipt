import pandas as pd
import torch
import hydra
import tqdm
from omegaconf import DictConfig

from ecglib.models.model_builder import create_model

from ecglib.data.datasets import EcgDataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report


@hydra.main(version_base=None, config_path="../ml_function", config_name="config")
def inference(cfg: DictConfig):
   
    state_dict = torch.load(cfg.checkpoint_path, map_location=torch.device("cpu"))

    data_sources = cfg["dataset"]["data_sources"]
    leads_num = 12

    model_output_path = cfg.predict.save_model_output if cfg.predict.save_model_output else ''
   
    df_test = pd.read_csv(cfg.test_path)
    test_target = df_test.target.values

    test_dataset = EcgDataset(df_test, test_target, data_type="npz")

    model = create_model(model_name="resnet1d18", config=cfg)
    model.load_state_dict(state_dict)
    model.to(cfg.device)
    model.eval()

    data_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    raw_preds, true_labels = ([], [])

    for batch in tqdm(data_loader, total=len(data_loader)):
        index, (input, targets) = batch

        outputs =  torch.nn.Sigmoid()(model(input))
        
        targets = targets.to(cfg.device) 

        raw_preds.extend(outputs.tolist())
        true_labels.extend(targets.tolist())

    print(classification_report(true_labels, raw_preds, zero_division=False))
    
if __name__ == "__main__":
    inference()
