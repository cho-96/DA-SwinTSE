import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from models.data import CustomDataset, expand_dims_n

def get_args():
    parser = argparse.ArgumentParser(description="Test DA-SwinTSE Model")
    parser.add_argument('--folder_path', type=Path, required=True, help='Path to processed test data folder')
    parser.add_argument('--data', type=str, required=True, help='Sub-folder/Site name for testing')
    parser.add_argument('--config', type=str, required=True, help='Sensor configuration (e.g., config1)')
    parser.add_argument('--model_path', type=Path, required=True, help='Base path where models are saved')
    parser.add_argument('--model_name', type=str, required=True, help='Experiment name to load')
    parser.add_argument('--mode', type=str, default='speed')
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--max_speed', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--clip', type=int, default=90, help='Clip value for evaluation')
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proc_path = args.folder_path / args.data
    model_file_path = args.model_path / args.model_name / 'DASwinTSE.pt'
    
    dataset = CustomDataset(
        proc_path / f"input_{args.mode}_{args.config}.npy",
        proc_path / f"input_occupancy_{args.config}.npy",
        proc_path / f"output_{args.mode}.npy"
    )
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    if not model_file_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_file_path}")
        
    model = torch.load(model_file_path, map_location=device, weights_only=False)
    model.eval()

    mse_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()
    
    total_mse, total_mae, total_samples = 0.0, 0.0, 0

    print(f"Testing model: {args.model_name} on data: {args.data} ({args.config})")

    with torch.no_grad():
        for x_speed, x_occ, y in test_loader:
            x_speed_norm = x_speed / args.max_speed if args.mode == 'speed' else x_speed
            
            x_in = torch.unsqueeze(expand_dims_n(x_speed_norm, args.window_size), 1).to(device)
            mask_in = torch.unsqueeze(expand_dims_n(x_occ, args.window_size), 1).to(device)
            y = torch.unsqueeze(y, 1).to(device)

            outputs = model(x_in, mask_in)
            
            if args.mode == 'speed':
                outputs = outputs * args.max_speed
            
            outputs = outputs[:, :, :y.shape[2], :y.shape[3]]
            outputs = torch.clamp(outputs, 0, args.max_speed)

            batch_size = x_in.size(0)
            total_mse += mse_fn(outputs, y).item() * batch_size
            total_mae += mae_fn(outputs, y).item() * batch_size
            total_samples += batch_size

    rmse = np.sqrt(total_mse / total_samples)
    mae = total_mae / total_samples

    print("-" * 50)
    print(f"Results for {args.data} | Config: {args.config}")
    print(f"RMSE: {rmse:.3f} km/h")
    print(f"MAE : {mae:.3f} km/h")
    print("-" * 50)

if __name__ == '__main__':
    main()