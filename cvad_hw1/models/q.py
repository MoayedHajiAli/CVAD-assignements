import torch.nn as nn
import torch


class MultiLayerQ(nn.Module):
    """Q network consisting of an MLP."""
    def __init__(self, config, inp_dim=13, out_dim=1, hidden_dims=[64, 128]):
        super().__init__()
        hidden_dims = [inp_dim] + hidden_dims + [out_dim]
        lst = []
        for h1, h2 in zip(hidden_dims[:-1], hidden_dims[1:]):
            lst.append(nn.Linear(h1, h2))
            lst.append(nn.ReLU())
        
        lst[-1] = nn.Tanh()
        self.fc = nn.Sequential(*lst)

    def forward(self, features, actions):
        z = torch.cat([features, actions], dim=-1)
        return self.fc(z)
