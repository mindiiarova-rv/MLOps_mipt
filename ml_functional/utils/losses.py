import torch
import numpy as np


def get_loss(df, device):
    pos_weight = calculate_pos_weight(df)
    pos_weight = torch.tensor([pos_weight]).to(device)

    return torch.nn.BCEWithLogitsLoss(
        size_average=True, reduce=True, pos_weight=pos_weight
    )


def calculate_pos_weight(df):
    target_array = np.array(df["target"].tolist())

    zeros_count = np.sum(target_array == 0, axis=0)
    ones_count = np.sum(target_array == 1, axis=0)

    ratios = zeros_count / ones_count
    return ratios.tolist()
