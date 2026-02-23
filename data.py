import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset): 
    def __init__(self, input_process_path, vehID_process_path, output_process_path, clip):
        self.x_data = np.load(input_process_path)[int(clip/2):-int(clip/2),:,:]
        self.x_data2 = np.load(vehID_process_path)[int(clip/2):-int(clip/2),:,:]
        self.y_data = np.load(output_process_path)[int(clip/2):-int(clip/2),:,:]

    # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.x_data)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        x2 = torch.FloatTensor(self.x_data2[idx])
        y = torch.FloatTensor(self.y_data[idx])

        return x, x2, y

class CustomDataset_woclip(Dataset): 
    def __init__(self, input_process_path, vehID_process_path, output_process_path):
        self.x_data = np.load(input_process_path)
        self.x_data2 = np.load(vehID_process_path)
        self.y_data = np.load(output_process_path)

    # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.x_data)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        x2 = torch.FloatTensor(self.x_data2[idx])
        y = torch.FloatTensor(self.y_data[idx])

        return x, x2, y
    
class CustomDataset_woID(Dataset): 
    def __init__(self, input_mode_path, output_mode_path):
        self.x_data = np.load(input_mode_path)
        self.y_data = np.load(output_mode_path)

    # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.x_data)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx): 
        x1 = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])

        return x1, y


class CustomDataset_full(Dataset): 
    def __init__(self, input_mode_path, input_adj_path, output_mode_path):
        self.x_data = np.load(input_mode_path)[50:-50]
        self.x_adj1 = np.load(input_adj_path)[0][50:-50]
        self.x_adj2 = np.load(input_adj_path)[1][50:-50]
        self.y_data = np.load(output_mode_path)[50:-50]
        self.p1_1 = np.load(input_adj_path)[0][25:-75]
        self.p1_2 = np.load(input_adj_path)[0][:-100]
        self.p2_1 = np.load(input_adj_path)[1][75:-25]
        self.p2_2 = np.load(input_adj_path)[1][100:]

    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        x_adj1 = torch.FloatTensor(self.x_adj1[idx])
        x_adj2 = torch.FloatTensor(self.x_adj2[idx])
        y = torch.FloatTensor(self.y_data[idx])
        p1_1 = torch.FloatTensor(self.p1_1[idx])
        p1_2 = torch.FloatTensor(self.p1_2[idx])
        p2_1 = torch.FloatTensor(self.p2_1[idx])
        p2_2 = torch.FloatTensor(self.p2_2[idx])

        return x, x_adj1, x_adj2, y, p1_1, p1_2, p2_1, p2_2

    
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


# 입력 사이즈를 8의 배수로 조정
def make_multiple_of_8(size):
    return ((size + 7) // 8) * 8  # 올림하여 8의 배수로 만듦

def normalize_data(x, y, max):
    # Normalize x and y using their respective means and stds
    x_norm = x / max
    y_norm = y / max
    return x_norm, y_norm

def save_normalization_params(max, save_path):
    # Save the normalization parameters (mean, std)
    torch.save({'max': max}, save_path)
    print(f"Normalization parameters saved to {save_path}")

def load_normalization_params(load_path):
    # Load the normalization parameters (mean, std)
    params = torch.load(load_path)
    return params['max']

def calculate_params(output_process_path):
    output_process = np.load(output_process_path)

    max = np.max(output_process)
    return max