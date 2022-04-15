import torch.nn as nn

import torchvision.models as models
import torch
import numpy as np

class AffordancePredictor(nn.Module):
    
    def __init__(self, n_commands=4):
        super().__init__()
        self.n_commands = n_commands
        
        self.backbone = models.resnet18(pretrained=True)
        
        # discard the last two layers
        self.backbone = nn.Sequential(*(list(self.backbone.children())[:-2]))
        
        # add one addtional conv and global average pool
        self.backbone.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        self.backbone.append(nn.ReLU())
        self.backbone.append(nn.AvgPool2d(kernel_size=7,  stride=1,  padding=0))
        self.backbone.append(nn.Dropout(0.5))
        

        # uncond predictions (1 afford class)
        self.uncond_head = nn.Sequential(nn.Linear(512, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softmax())
        
        # cond: the afford
        self.con_head = nn.ModuleList([ nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 3)) 
                                for i in range(n_commands)])

    def forward(self, imgs, commands):
        img_out = self.backbone(imgs).squeeze(-1).squeeze(-1)
        
        ret = torch.zeros(commands.shape[0], 4).to(imgs.device)
        
        # speed pred
        ret[:, 0:1] = self.uncond_head(img_out)
        
        for c in range(self.n_commands):
            idxs = torch.Tensor(np.where(commands.cpu() == c)[0]).long().to(imgs.device)
            if len(idxs) > 0:
                ret[idxs, 1:] = self.con_head[c](img_out[idxs])
        
        return ret
            



