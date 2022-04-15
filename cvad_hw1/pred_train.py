import torch
from torch.utils.data import DataLoader
from expert_dataset import ExpertDataset
from models.affordance_predictor import AffordancePredictor
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm 


def validate(model, dataloader, device='cpu'):
    """Validate model performance on the validation dataset"""
    # Your code here
    model.eval()
    ce_total_loss, mse_total_loss = 0, 0
    for batch in tqdm(dataloader):
        pred = model(batch['image'].to(device), batch['command'].to(device))
        cls_loss = F.binary_cross_entropy(pred[:, 0:1], batch['affordances'].to(device)[:, 0:1]) # tl status
        mse_loss = F.mse_loss(pred[:, 1:], batch['affordances'].to(device)[:, 1:])
        mse_total_loss += mse_loss.item()
        ce_total_loss += cls_loss.item()
    
    return ce_total_loss / len(batch), mse_total_loss / len(batch)

def train(model, dataloader, optimizer, lmbd=0.5, device='cpu'):
    """Train model on the training dataset for one epoch"""
    # Your code here
    model.train()
    ce_total_loss, mse_total_loss = 0, 0
    for batch in tqdm(dataloader):
        pred = model(batch['image'].to(device), batch['command'].to(device))
        cls_loss = F.binary_cross_entropy(pred[:, 0:1], batch['affordances'].to(device)[:, 0:1]) # tl status
        mse_loss = F.mse_loss(pred[:, 1:], batch['affordances'].to(device)[:, 1:])
        loss = (1-lmbd) * mse_loss + lmbd * cls_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mse_total_loss += mse_loss.item()
        ce_total_loss += cls_loss.item()
    
    return ce_total_loss / len(batch), mse_total_loss / len(batch)

def plot_losses(ce_train_losses, mse_train_losses, ce_val_losses, mse_val_losses):
    """Visualize your plots and save them for your report."""
    # Your code here
    fig, ax = plt.subplots()
    ax.plot(ce_train_losses, label='train CE loss')
    ax.plot(mse_train_losses, label='train MSE loss')
    ax.plot(np.array(ce_train_losses) + np.array(mse_train_losses), label='train total loss')
    
    ax.plot(ce_val_losses, label='val CE loss')
    ax.plot(mse_val_losses, label='val MSE loss')
    ax.plot(np.array(ce_val_losses) + np.array(mse_val_losses), label='val total loss')
    
    ax.set_title('Affordances prediction loss graph')
    ax.set_xlabel('epoch')
    ax.set_ylabel('bce + mse loss')
    ax.set_ylim(top=10.)
    fig.legend()
    fig.savefig('affordances_fig.png')

def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = '/userfiles/eozsuer16/expert_data/train'
    val_root = '/userfiles/eozsuer16/expert_data/val'
    model = AffordancePredictor()
    train_dataset = ExpertDataset(train_root)
    val_dataset = ExpertDataset(val_root)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 20
    batch_size = 64
    lr = 0.0002
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path = "pred_model.ckpt"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=10, shuffle=True,
                            drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=10, shuffle=False)


    optimizer = optim.Adam(model.parameters(), lr=lr)
    ce_train_losses, mse_train_losses = [], []
    ce_val_losses, mse_val_losses = [], []

    model.to(device)
    for i in tqdm(range(num_epochs)):
        ce, mse = train(model, train_loader, optimizer, device=device)
        ce_train_losses.append(ce)
        mse_train_losses.append(mse)
        
        ce, mse = validate(model, val_loader, device=device)
        ce_val_losses.append(ce)
        mse_val_losses.append(mse)
        
    torch.save(model, save_path)
    plot_losses(ce_train_losses, mse_train_losses, ce_val_losses, mse_val_losses)


if __name__ == "__main__":
    main()
