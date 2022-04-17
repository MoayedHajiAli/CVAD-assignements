import torch
from torch.utils.data import DataLoader
from torch import optim
from expert_dataset import ExpertDataset
from models.cilrs import CILRS
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm 
import numpy as np
import argparse
import os
import wandb

global_step = 0

def validate(model, dataloader, lmbd=0.2, device='cpu'):
    """Validate model performance on the validation dataset"""
    # Your code here
    global global_step

    model.eval()
    s_total_loss, ac_total_loss = 0, 0
    for batch in tqdm(dataloader):
        pred = model(batch['image'].to(device), batch['measure'].to(device), batch['command'].to(device))
        s_loss = F.l1_loss(pred[:, 0:1], batch['measure'].to(device))
        throttle_loss = F.l1_loss(pred[:, 1:2], batch['action'][:, 0:1].to(device))
        brake_loss = F.l1_loss(pred[:, 2:3], batch['action'][:, 1:2].to(device))
        steer_loss = F.l1_loss(pred[:, 3:4], batch['action'][:, 2:3].to(device))
        ac_loss = throttle_loss + brake_loss + steer_loss
        loss = (1 - lmbd) * ac_loss + lmbd * s_loss
        s_total_loss += s_loss.item()
        ac_total_loss += ac_loss.item()

        wandb.log({'val/throttle_loss': throttle_loss.mean().item(),
                   'val/brake_loss': brake_loss.mean().item(),
                   'val/steer_loss': steer_loss.mean().item(),
                   'val/action_loss': ac_loss.mean().item(),
                   'val/speed_loss': s_loss.mean().item(),
                   'val/total_loss': loss.mean().item(),
                   'global_step': global_step})
    
    return s_total_loss / len(batch), ac_total_loss / len(batch)

def train(model, dataloader, optimizer, lmbd = 0.2, device='cpu'):
    """Train model on the training dataset for one epoch"""
    # Your code here
    global global_step
    model.train()
    s_total_loss, ac_total_loss = 0, 0
    for batch in tqdm(dataloader):
        pred = model(batch['image'].to(device), batch['measure'].to(device), batch['command'].to(device))
        s_loss = F.l1_loss(pred[:, 0:1], batch['measure'].to(device))
        throttle_loss = F.l1_loss(pred[:, 1:2], batch['action'][:, 0:1].to(device))
        brake_loss = F.l1_loss(pred[:, 2:3], batch['action'][:, 1:2].to(device))
        steer_loss = F.l1_loss(pred[:, 3:4], batch['action'][:, 2:3].to(device))
        ac_loss = throttle_loss + brake_loss + steer_loss
        loss = (1 - lmbd) * ac_loss + lmbd * s_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        s_total_loss += s_loss.item()
        ac_total_loss += ac_loss.item()

        wandb.log({'train/throttle_loss': throttle_loss.mean().item(),
                   'train/brake_loss': brake_loss.mean().item(),
                   'train/steer_loss': steer_loss.mean().item(),
                   'train/action_loss': ac_loss.mean().item(),
                   'train/speed_loss': s_loss.mean().item(),
                   'train/total_loss': loss.mean().item(),
                   'global_step': global_step})
        
        global_step += 1
    return s_total_loss / len(batch), ac_total_loss / len(batch)



def plot_losses(ac_train_losses, s_train_losses, ac_val_losses, s_val_losses):
    """Visualize your plots and save them for your report."""
    # Your code here
    fig, ax = plt.subplots()
    ax.plot(ac_train_losses, label='train action')
    ax.plot(s_train_losses, label='train speed')
    ax.plot(np.array(ac_train_losses) + np.array(s_train_losses), label='train total')
    
    ax.plot(ac_val_losses, label='val action')
    ax.plot(s_val_losses, label='val speed')
    ax.plot(np.array(ac_val_losses) + np.array(s_val_losses), label='val total')
    
    ax.set_title('CILRS loss graph')
    ax.set_xlabel('epoch')
    ax.set_ylabel('mse loss')
    ax.set_ylim(top=2.)
    fig.legend()
    fig.savefig('cilrs_fig.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default="F")
    parser.add_argument('--name', type=str, default='cilrs_run')
    args =  parser.parse_args()


    wandb.init(project='cvad-hw1', name=args.name)

    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = '/userfiles/eozsuer16/expert_data/train'
    val_root = '/userfiles/eozsuer16/expert_data/val'
    if args.resume == 'F':
        model = CILRS()
    else:
        print("Loading model from cilrs_model.ckpt")
        model = torch.load(args.resume)
    train_dataset = ExpertDataset(train_root)
    val_dataset = ExpertDataset(val_root)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 40
    batch_size = 512
    lr = 0.0002
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = "checkpoints/cilrs/"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=10, shuffle=True,
                            drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=10, shuffle=False)


    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    ac_train_losses, s_train_losses = [], []
    ac_val_losses, s_val_losses = [], []

    model.to(device)
    for i in tqdm(range(num_epochs)):
        s_loss, ac_loss = train(model, train_loader, optimizer, device=device)
        ac_train_losses.append(ac_loss)
        s_train_losses.append(s_loss)
        print("Train loss:", s_loss, ac_loss)

        s_loss, ac_loss = validate(model, val_loader, device=device)
        ac_val_losses.append(ac_loss)
        s_val_losses.append(s_loss)
        
        print("Val loss:", s_loss, ac_loss)


        torch.save(model, os.path.join(save_dir, f'cilrs_{i}.ckpt'))

    plot_losses(ac_train_losses, s_train_losses, ac_val_losses, s_val_losses)


if __name__ == "__main__":
    main()
