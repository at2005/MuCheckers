import torch
import torch.nn as nn
import torch.nn.functional as F

class RepresentationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 128, kernel_size=3)
        
    def forward(self, s):
        return s
        
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_actions = 10
    def forward(self, s):
        policy = F.softmax(torch.randn(self.num_actions))
        some_val = 0.78
        return (policy, some_val)

class DynamicsNet(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, s, a):
        some_reward = 0.3
        return (s, some_reward) 