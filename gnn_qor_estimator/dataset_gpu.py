
import torch
from torch_geometric.data import Dataset, Data
import os.path as osp
import glob

class CustomData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "cfg_edge_index" or key == "cfg_select":
            return self.num_bbs
        if key == "bb_batch":
            return 1
        if 'index' in key:
            return self.num_nodes
        else:
            return 0
    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'index' in key:
            return 1
        else:
            return 0
        

class CustomDataset(Dataset):
    def __init__(self, config_name):
        self.saveDir = "datasets/" + config_name + "/"
        super().__init__()

    @property
    def processed_file_names(self):
        return glob.glob(f"{self.saveDir}*.pt")
    
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.saveDir, f'data_{idx}.pt'))
        return data