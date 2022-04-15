import torch
from torch.utils.data import DataLoader
from torch import optim
from expert_dataset import ExpertDataset
from models.cilrs import CILRS
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm 


def validate(model, dataloader, device='cpu'):
    """Validate model performance on the validation dataset"""
    # Your code here
    model.eval()
    total_loss = 0
    for batch in dataloader:
        pred = model(batch['image'].to(device), batch['measure'].to(device), batch['command'].to(device))
        loss = F.mse_loss(pred, torch.cat([batch['measure'].to(device), batch['action'].to(device)], dim=-1))
        total_loss += loss.item()
    
    return total_loss / len(batch)

def train(model, dataloader, optimizer, lmbd = 0.1, device='cpu'):
    """Train model on the training dataset for one epoch"""
    # Your code here
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        pred = model(batch['image'].to(device), batch['measure'].to(device), batch['command'].to(device))
        loss = F.mse_loss(pred, torch.cat([batch['measure'].to(device), batch['action'].to(device)], dim=-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(batch)



def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    # Your code here
    fig, ax = plt.subplots()
    ax.plot(train_loss, label='train')
    ax.plot(val_loss, label='test')
    ax.set_title('CILRS loss graph')
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE loss')
    ax.set_ylim(top=10.)
    fig.legend()
    fig.savefig('cilrs_fig.png')

def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = '/userfiles/eozsuer16/expert_data/train'
    val_root = '/userfiles/eozsuer16/expert_data/val'
    model = CILRS()
    train_dataset = ExpertDataset(train_root)
    val_dataset = ExpertDataset(val_root)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 25
    batch_size = 256
    lr = 0.0002
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path = "cilrs_model.ckpt"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=10, shuffle=True,
                            drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=10, shuffle=False)


    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []

    model.to(device)
    for i in tqdm(range(num_epochs)):
        train_losses.append(train(model, train_loader, optimizer, device=device))
        val_losses.append(validate(model, val_loader, device=device))
        print("Train loss:", train_losses[-1])
        print("Val loss:", val_losses[-1])

    torch.save(model, save_path)
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
