import json
import torch

def read_json(fpath):
    with open(fpath, 'r') as f:
        data = json.load(f)
    return data

OPTIMIZER_DICT = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW
}