import os

import yaml

from carla_env.env import Env
from models.cilrs import CILRS
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import numpy as np

class Evaluator():
    def __init__(self, env, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.env = env
        self.config = config
        self.agent = self.load_agent()
        
        self.img_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        

    def load_agent(self,):
        model = torch.load('checkpoints/cilrs/cilrs_35.ckpt')
        return model.to(self.device).eval()
        
    def generate_action(self, rgb, command, speed):
        # predict actions 
        rgb = self.img_preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        command = torch.Tensor([command]).unsqueeze(0).to(self.device)
        speed = torch.Tensor([speed]).unsqueeze(0).to(self.device)
        actions = self.agent(rgb, command, speed)
        return torch.clip(actions[0, 1:].cpu().detach().float(), -1. , 1.).numpy().astype(np.float32)

    def take_step(self, state):
        rgb = state["rgb"]
        command = state["command"]
        speed = state["speed"]
        throttle, brake, steer = self.generate_action(rgb, speed, command)

        brake = int(brake * 10)/ 10
        
        print(throttle, steer, brake)
        action = {
            "throttle": float(throttle),
            "brake": float(brake),
            "steer": float(steer)
        }
        state, reward_dict, is_terminal = self.env.step(action)
        return state, is_terminal

    def evaluate(self, num_trials=100):
        terminal_histogram = {}
        for i in tqdm(range(num_trials)):
            state, _, is_terminal = self.env.reset()
            for i in tqdm(range(5000)):
                if is_terminal:
                    break
                state, is_terminal = self.take_step(state)
            if not is_terminal:
                is_terminal = ["timeout"]
                
            print(is_terminal)
            terminal_histogram[is_terminal[0]] = (terminal_histogram.get(is_terminal[0], 0)+1)
        print("Evaluation over. Listing termination causes:")
        for key, val in terminal_histogram.items():
            print(f"{key}: {val}/100")


def main():
    with open(os.path.join("configs", "cilrs.yaml"), "r") as f:
        config = yaml.full_load(f)

    with Env(config) as env:
        evaluator = Evaluator(env, config)
        evaluator.evaluate()


if __name__ == "__main__":
    main()
