import argparse
import os
from models.generator import DASwinTSE
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import DataLoader
from data import *
from pathlib import Path
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
import torch.nn.functional as F

DATA_DIR = ''

parser = argparse.ArgumentParser()
parser.add_argument('--site', type=str, help='Site identifier')
parser.add_argument('--name', type=str, help='Name identifier for saving model')
parser.add_argument('--mode', type=str, default='speed', help='traffic state mode')
parser.add_argument('--embed_dim', type=int, default=24, help='Embedding dimension')
parser.add_argument('--K', type=int, default=3, help='Number of CSTB')
parser.add_argument('--L', type=int, default=4, help='Number of STL')
parser.add_argument('--window_size', type=int, default=16, help='Window size')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--max_speed', type=int, default=80, help='Maximum speed') 
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--num_epoch', type=int, default=20, help='Number of epochs for training')
parser.add_argument('--process_path', type=Path, default='/home/transmatics/HC/TSE_double/Data/SUMO/Processed', help='Process path')
parser.add_argument('--model_base_path', type=Path, default='/home/transmatics/HC/TSE_double/Data/Model', help='Model base path')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def make_depths(K, L):
    return [L] * K

def heads_for(K):
    return [6] * K

def get_model_and_path(opt):
    prefix = 'DASwinTSE'

    model_dir = opt.model_base_path / opt.site / opt.name
    os.makedirs(model_dir, exist_ok=True)
    model_path = model_dir / f'{prefix}.pt'
        
    return DASwinTSE(
        in_chans=1,
        out_chans=1,
        window_size=opt.window_size,
        depths=make_depths(opt.K, opt.L),
        depths_t=make_depths(opt.K, opt.L),
        embed_dim=opt.embed_dim,
        num_heads=heads_for(opt.K),
        num_heads_t=heads_for(opt.K),
        mlp_ratio=2,
        s_size=20,
        t_size=1
    ), model_path
    

def train_model():
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_path = opt.process_path / opt.site / opt.name

    input_mode_path = processed_path / f"input_{opt.mode}.npy"
    input_occupancy_path = processed_path / f"input_occupancy.npy"
    output_mode_path = processed_path / f"output_{opt.mode}.npy"
        
    dataset = CustomDataset_woclip(input_mode_path, input_occupancy_path, output_mode_path)

    generator, model_path = get_model_and_path(opt)
                 
    generator.to(device)
        
    train_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(generator.parameters(), lr=opt.learning_rate)

    for epoch in range(opt.num_epoch):
        generator.train()
        running_loss = 0
        total_steps = 0

        for i, data in enumerate(train_loader):
            x1, x2, y = data
            if opt.mode == 'speed':
                x1, y = x1/opt.max_speed, y/opt.max_speed
            x1 = expand_dims_n(x1, opt.window_size)
            x2 = expand_dims_n(x2, opt.window_size)

            x1 = torch.unsqueeze(x1, dim=1)
            x2 = torch.unsqueeze(x2, dim=1)

            X_train = x1
            X_mask = x2

            y = torch.unsqueeze(y, dim=1)
            Y_train = y

            X_train = X_train.to(device)
            X_mask = X_mask.to(device)
            Y_train = Y_train.to(device)

            outputs = generator(X_train, X_mask)  
            y0, y1 = y.shape[2], y.shape[3]
            
            outputs = outputs[:, :, :y0, :y1]
            Y_train = Y_train[:, :, :, :]

            loss = criterion(outputs, Y_train) 
            running_loss += loss.item()*x1.shape[0]
            total_steps += x1.shape[0]

            generator.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_loss = running_loss / total_steps
        print('Epoch {}: '.format(epoch), train_loss)

    torch.save(generator, model_path)
        

if __name__ == '__main__':
    train_model()
