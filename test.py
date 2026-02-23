import argparse
import os
import numpy as np
import time
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from data import *
from pathlib import Path

DATA_DIR = ''

parser = argparse.ArgumentParser()
parser.add_argument('--section', type=str, help='Section name')
parser.add_argument('--time', type=str, help='Time name')
parser.add_argument('--config', type=str, help='Sensor configuration')
parser.add_argument('--model_site', type=str, help='Site identifier')
parser.add_argument('--model_name', type=str, help='Name identifier for saving model')
parser.add_argument('--mode', type=str, default='speed', help='traffic state mode')
parser.add_argument('--window_size', type=int, default=16, help='Window size')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--max_speed', type=int, default=80, help='Maximum speed')  
parser.add_argument('--clip', type=int, default=90, help='Clip second') 
parser.add_argument('--process_path', type=Path, default='/home/transmatics/HC/TSE_double/Data/US101/Processed', help='Process path')
parser.add_argument('--model_base_path', type=Path, default='/home/transmatics/HC/TSE_double/Data/Model', help='Model base path')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_model_path(opt):    
    prefix = 'DASwinTSE'
    model_path = opt.model_base_path / opt.model_site / opt.model_name / f'{prefix}.pt'
    return model_path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt = parser.parse_args()

    processed_path = opt.process_path / opt.section / opt.time

    input_mode_path = processed_path / f"input_{opt.mode}_{opt.config}.npy"
    input_occupancy_path = processed_path / f"input_occupancy_{opt.config}.npy"
    output_mode_path = processed_path / f"output_{opt.mode}.npy"
        
    dataset = CustomDataset(input_mode_path, input_occupancy_path, output_mode_path, opt.clip)
    model_path = get_model_path(opt)

    generator =  torch.load(model_path, weights_only=False)

    generator.to(device)

    testloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    criterion = nn.MSELoss()
    criterion2 = nn.L1Loss()
    
    total_steps = 0
    running_loss = 0
    running_loss2 = 0

    for i, (x1, x2, y) in enumerate(testloader):
        mask = torch.unsqueeze(x2, dim=1).float()
        if opt.mode == 'speed':
            x1 = x1/opt.max_speed
        x1 = expand_dims_n(x1, opt.window_size)
        x2 = expand_dims_n(x2, opt.window_size)

        x1 = torch.unsqueeze(x1, dim=1)
        x2 = torch.unsqueeze(x2, dim=1)
        y = torch.unsqueeze(y, dim=1)

        X_test = x1
        X_mask = x2
        Y_test = y

        X_test = X_test.to(device)
        X_mask = X_mask.to(device)
        Y_test = Y_test.to(device)
        mask   = mask.to(device)

        outputs = generator(X_test, X_mask)
        if opt.mode == 'speed':
            outputs = outputs*opt.max_speed

        y0, y1 = y.shape[2], y.shape[3]
        outputs = outputs[:, :, :y0, :y1]
        outputs = torch.clip(outputs, 0, opt.max_speed)

        loss = criterion(outputs, Y_test)
        loss2 = criterion2(outputs, Y_test)
        total_steps += X_test.shape[0]

        running_loss += loss.item() * X_test.shape[0]
        running_loss2 += loss2.item() * X_test.shape[0]
        total_steps += X_test.shape[0]

    test_loss = np.sqrt(running_loss / total_steps)
    test_loss2 = running_loss2 / total_steps

    print(f'Test Loss (RMSE): {test_loss:.3f} km/h')
    print(f'Test Loss (L1): {test_loss2:.3f} km/h')


if __name__ == '__main__':
    main()
