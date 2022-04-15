from torch.utils.data import Dataset
import json
from PIL import Image
import numpy as np
from torchvision import transforms
import os
import torch

class ExpertDataset(Dataset):
    """Dataset of RGB images, driving affordances and expert actions"""
    def __init__(self, data_root):
        self.data_dir = data_root
        self.ids = os.listdir(os.path.join(data_root, 'rgb'))
        self.ids = [id[:-4] for id in self.ids]
        
        self.img_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def __getitem__(self, index):
        """Return RGB images and measurements"""
        # Your code here
        ret = {}
        
        # load measurements
        m_path = os.path.join(self.data_dir, 'measurements', f"{self.ids[index]}.json")
        with open(m_path, 'r') as f:
            measurements = json.load(f)
        
        ret['measure'] = torch.Tensor([measurements['speed']])
        ret['command'] = torch.Tensor([measurements['command']])
        ret['action'] = torch.cat([torch.Tensor([measurements['throttle']]), 
                                   torch.Tensor([measurements['brake']]),
                                   torch.Tensor([measurements['steer']])])
        
        ret['affordances'] = torch.cat([torch.Tensor([measurements['tl_state']]),
                                        torch.Tensor([measurements['lane_dist']]), 
                                        torch.Tensor([measurements['route_angle']]),
                                        torch.Tensor([measurements['tl_dist']])]) 
        
        
        # load img
        img_path = os.path.join(self.data_dir, 'rgb', f"{self.ids[index]}.png")
        img = Image.open(img_path)
        img = np.array(img)
        
        # bgr 2 rgb
        tmp = img[:, :, 0].copy()
        img[:, :, 0] = img[:, :, 2]
        img[:, :, 2] = tmp
        
        ret['image'] = self.img_preprocess(Image.fromarray(img))
        
        return ret
    
    def __len__(self,):
        return len(self.ids)