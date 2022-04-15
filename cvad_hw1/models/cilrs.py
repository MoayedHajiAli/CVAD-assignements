import torch.nn as nn
import torchvision.models as models
import torch
import numpy as np

class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""
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
        self.backbone.append(nn.Dropout(0.2))
        

        # speed
        self.speed_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))
        
        # measurement fc (speed -> 512)
        self.measure_fc = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 128))
        
        # measurement fc (speed -> 512)
        self.post_fc = nn.Sequential(nn.Linear(512+128, 512), nn.ReLU(), nn.Linear(512, 512))
        
        self.command_fc = nn.ModuleList([ nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 3)) 
                                for i in range(n_commands)])

    def forward(self, imgs, measures, commands):
        img_out = self.backbone(imgs).squeeze(-1).squeeze(-1)
        measure_out = self.measure_fc(measures)
        
        out = torch.cat([img_out, measure_out], dim=-1)
        out = self.post_fc(out)
        
        ret = torch.zeros(commands.shape[0], 4).to(imgs.device)
        
        # speed pred
        ret[:, 0:1] = self.speed_head(img_out)
        
        for c in range(self.n_commands):
            idxs = torch.Tensor(np.where(commands.cpu() == c)[0]).long().to(imgs.device)
            if len(idxs) > 0:
                ret[idxs, 1:] = self.command_fc[c](out[idxs])
        
        return ret
            

