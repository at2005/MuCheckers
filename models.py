import torch
import torch.nn as nn
import torch.nn.functional as F

# see the previous 20 states
history_dim = 20
unroll_steps = 5
num_actions = 10**2 * 10**2
device = "cpu"


# this appends a homogeneous plane to the board state indicating the player
def create_input(board_state: torch.Tensor, player : str):
    # board state is (B, C x History, H, W)
    b,c,h,w = board_state.shape
    player_colour_plane = torch.zeros((b,1,h,w), device=board_state.device) if player == "black" else torch.ones((b,1,h,w), device=board_state.device)
    input_state = torch.cat([board_state, player_colour_plane], -3)
    return input_state

class RepresentationNet(nn.Module):
    def __init__(self):
        super().__init__()
        # initially we take two 
        self.num_blocks = 16
        self.conv_block = nn.Conv2d(history_dim * 2 + 1, 256, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(256)
        self.residual_blocks = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
        ) for _ in range(self.num_blocks)]) 

    def forward(self, x):
        # x is of shape (B, (history_dim * 2 + 1), 10, 10)
        x = F.relu(self.batch_norm(self.conv_block(x)))
        for block in self.residual_blocks:
            x = F.relu(x + block(x))
        x_norm = (x - x.min()) / (x.max() - x.min())
        return x_norm
        
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_blocks = 29
        self.conv_block = nn.Conv2d(256, 256, 3, padding=1)
        self.batch_norm = nn.BatchNorm2d(256)
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
            ) for _ in range(self.num_blocks)])
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 2, 1, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            # 10,000 possible moves
            nn.Linear(288, 10_000),
            nn.Softmax(-1)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(144, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, input_features : torch.Tensor, compute_value=True):
        x = F.relu(self.batch_norm(self.conv_block(input_features)))
        for block in self.residual_blocks:
            x = F.relu(x + block(x))
        policy = self.policy_head(x)
        value = self.value_head(x) if compute_value else 0
        return policy, value 
        

class DynamicsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_blocks = 16
        self.conv_block = nn.Conv2d(256 + 2, 256, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(256)
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
            ) for _ in range(self.num_blocks)])

    def forward(self, hidden_state, action):
        state = torch.cat([hidden_state, action], dim=-3)
        x = F.relu(self.batch_norm(self.conv_block(state)))
        for block in self.blocks:
            x = F.relu(x + block(x))

        # scale gradient by 1/2, that's what muzero does
        # to ensure the total gradient applied here
        # stays constant
        x.register_hook(lambda grad: grad*0.5)
        return x 
