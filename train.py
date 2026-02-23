import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from models.generator import DASwinTSE
from models.data import CustomDataset, expand_dims_n

def get_args():
    parser = argparse.ArgumentParser(description="Train DA-SwinTSE Model")
    parser.add_argument('--folder_path', type=Path, required=True, help='Path to processed data folder')
    parser.add_argument('--data', type=str, required=True, help='Experiment or Site name')
    parser.add_argument('--model_path', type=Path, required=True, help='Base path to save models')
    parser.add_argument('--mode', type=str, default='speed', help='Traffic state mode (e.g., speed)')
    parser.add_argument('--embed_dim', type=int, default=24)
    parser.add_argument('--K', type=int, default=1, help='Number of CSTB') # 3
    parser.add_argument('--L', type=int, default=1, help='Number of STL') # 4
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=1) # 32
    parser.add_argument('--max_speed', type=int, default=80) 
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1) # 20
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    proc_path = args.folder_path / args.name
    save_dir = args.model_path / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = CustomDataset(
        proc_path / f"input_{args.mode}.npy",
        proc_path / "input_occupancy.npy",
        proc_path / f"output_{args.mode}.npy"
    )
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = DASwinTSE(
        in_chans=1, 
        out_chans=1, 
        window_size=args.window_size,
        depths=[args.L] * args.K, 
        depths_t=[args.L] * args.K,
        embed_dim=args.embed_dim, 
        num_heads=[6] * args.K, 
        num_heads_t=[6] * args.K,
        mlp_ratio=2, 
        s_size=20, 
        t_size=1
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Starting training for experiment: {args.name}")
    print(f"Device: {device}")

    for epoch in range(args.epochs):
        model.train()
        running_loss, total_samples = 0.0, 0

        for x_speed, x_occ, y in train_loader:
            if args.mode == 'speed':
                x_speed = x_speed / args.max_speed
                y = y / args.max_speed
            
            x_speed = torch.unsqueeze(expand_dims_n(x_speed, args.window_size), 1).to(device)
            x_occ = torch.unsqueeze(expand_dims_n(x_occ, args.window_size), 1).to(device)
            y = torch.unsqueeze(y, 1).to(device)

            outputs = model(x_speed, x_occ)
            outputs = outputs[:, :, :y.shape[2], :y.shape[3]]

            loss = criterion(outputs, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_speed.size(0)
            total_samples += x_speed.size(0)

        epoch_loss = running_loss / total_samples
        print(f"Epoch [{epoch+1:02d}/{args.epochs}] - Loss: {epoch_loss:.6f}")

    save_path = save_dir / 'DASwinTSE.pt'
    torch.save(model, save_path)
    print(f"Training completed. Model saved at: {save_path}")

if __name__ == '__main__':
    main()