import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset): 
    def __init__(self, input_process_path, vehID_process_path, output_process_path):
        self.x_data = np.load(input_process_path)
        self.x_data2 = np.load(vehID_process_path)
        self.y_data = np.load(output_process_path)

    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        x2 = torch.FloatTensor(self.x_data2[idx])
        y = torch.FloatTensor(self.y_data[idx])

        return x, x2, y
        
def expand_dims_n(x, n):
    if x.shape[1] % n != 0:
        x12_shape = list(x.shape)
        dx = n - x12_shape[1] % n
        x12_shape[1] = dx
        x12 = torch.zeros(x12_shape)
        x = torch.cat([x, x12], dim=1)
    if x.shape[2] % n != 0:
        x22_shape = list(x.shape)
        dy = n - x22_shape[2] % n
        x22_shape[2] = dy
        x22 = torch.zeros(x22_shape)        
        x = torch.cat([x, x22], dim=2)
    return x

def make_multiple_of_8(size):
    return ((size + 7) // 8) * 8  

def normalize_data(x, y, max):
    x_norm = x / max
    y_norm = y / max
    return x_norm, y_norm

def save_normalization_params(max, save_path):
    torch.save({'max': max}, save_path)
    print(f"Normalization parameters saved to {save_path}")

def load_normalization_params(load_path):
    params = torch.load(load_path)
    return params['max']

def calculate_params(output_process_path):
    output_process = np.load(output_process_path)

    max = np.max(output_process)
    return max