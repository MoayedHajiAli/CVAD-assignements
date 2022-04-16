import torch.nn as nn


class MultiLayerPolicy(nn.Module):
    """An MLP based policy network"""
    def __init__(self, inp_dim=11, out_dim=2, hidden_dims=[64, 128]):
        super().__init__()
        hidden_dims = [inp_dim] + hidden_dims + [out_dim]
        lst = []
        for h1, h2 in zip(hidden_dims[:-1], hidden_dims[1:]):
            lst.append(nn.Linear(h1, h2))
            lst.append(nn.ReLU())
        
        lst[-1] = nn.Tanh()
        self.fc = nn.Sequential(*lst)

    def forward(self, features):
        return self.fc(features)
