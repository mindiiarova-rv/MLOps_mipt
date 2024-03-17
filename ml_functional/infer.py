import pandas as pd
import ast
import torch
import hydra
from tqdm import tqdm
from omegaconf import DictConfig

from ecglib.models.model_builder import create_model

from ecglib.data.datasets import EcgDataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report


@hydra.main(version_base=None, config_path=".", config_name="config")
def inference(cfg: DictConfig):
   
    path = f"{cfg.checkpoint_path}/12_leads_resnet1d18_AFIB.pt"
    state_dict = torch.load(path, map_location=torch.device("cpu"))

    df_test = pd.read_csv(cfg.test_path)
    targets_train = [ [0.0] if 'AFIB' in ast.literal_eval(df_test.iloc[i]['scp_codes']) else [1.0] 
           for i in range(df_test.shape[0])]
    
    df_test["target"] = targets_train

    test_dataset = EcgDataset(df_test, df_test.target.values)

    model = create_model(model_name="resnet1d18", pathology='AFIB')
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
