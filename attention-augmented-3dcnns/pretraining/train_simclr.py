import os
import torch
from torch.utils.data import DataLoader
from pretraining.dataset import SimCLRDataset
from pretraining.model import SimCLR3D
from pretraining.losses import nt_xent_loss
from tqdm import tqdm


def train_simclr(
    patches_dir='data/patches_ssl',
    save_path='pretrained_models/ssl_encoder.pth',
    batch_size=32,
    epochs=20, 
    lr=1e-4,
    device=None
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SimCLRDataset(patches_dir)
    print(f"Found {len(dataset)} patch files in {patches_dir}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    model = SimCLR3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for view1, view2 in progress_bar:
            view1 = view1.to(device)
            view2 = view2.to(device)
            z1 = model(view1)
            z2 = model(view2)
            loss = nt_xent_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * view1.size(0)
            progress_bar.set_postfix({'loss': loss.item()})
        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        if (epoch + 1) % 10 == 0:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.backbone.state_dict(), save_path)
    # Final save
    torch.save(model.backbone.state_dict(), save_path)

if __name__ == '__main__':
    train_simclr() 