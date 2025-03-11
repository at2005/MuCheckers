from models import PolicyNet, RepresentationNet, DynamicsNet, unroll_steps, device
import torch
from infra import ExperienceStore, Experience
import torch.nn as nn
from collections import deque


class Trainer:
    def __init__(self, experience_buffer):
        self.experience_buffer: ExperienceStore = experience_buffer
        self.policy = PolicyNet().to(device)
        self.repr = RepresentationNet().to(device)
        self.dynamics = DynamicsNet().to(device)
        self.num_steps = 100_000_000
        self.forward_refresh = 1000
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.policy.parameters(), "lr": 3e-4},
                {"params": self.dynamics.parameters(), "lr": 3e-4},
                {"params": self.repr.parameters(), "lr": 3e-4},
            ],
            weight_decay=1e-2,
        )

        class LossBuffer:
            def __init__(self, capacity):
                self.buffer = deque()
                self.capacity = capacity
                self._mean = 0.0

            def append(self, element):
                old_sum = self._mean * len(self.buffer)
                if len(self.buffer) >= self.capacity:
                    old_sum -= self.buffer.popleft()
                self.buffer.append(element)
                self._mean = (old_sum + element) / len(self.buffer)

            @property
            def mean(self):
                return self._mean

        self.loss_window = LossBuffer(20)
        self.batch_size = 128

    def train(self):
        for i in range(self.num_steps):
            self.update_step()

            # update model parameters
            if i % self.forward_refresh == 0:
                print(f"Avg Loss: {self.loss_window.mean()}")
                # write parameters to a file, which the batcher will then read from
                torch.save(self.policy.state_dict(), "policy.pth")
                torch.save(self.dynamics.state_dict(), "dynamics.pth")
                torch.save(self.repr.state_dict(), "repr.pth")

    def update_step(self):
        self.optimizer.zero_grad()
        # training loop:
        # sample trajectory
        # consists of actions and states, and associated mcts priors
        # compute hidden state and feed into policy net
        # loss = 1/K * sum(loss_v + loss_p)
        # loss_p = D_KL(policy_net.priors, mcts_priors)
        # loss_v = (end_reward - policy_net.value)**2
        # rollout + train, compute loss
        batch: list[list[Experience]] = self.experience_buffer.sample_experiences(
            self.batch_size
        )

        kl_loss_fn = nn.KLDivLoss()
        value_loss_fn = nn.MSELoss()

        loss = torch.tensor(0.0, device=device)

        def player_to_reward(player, winner):
            if winner == "draw":
                return 0
            winner_bit = 1 if winner == "white" else 0
            player_bit = 1 if player == "white" else 0
            return -(1 ** (player_bit ^ winner_bit))

        batch_state = torch.stack(
            [torch.from_numpy(trajectory[0].state) for trajectory in batch]
        ).to(device)
        current_state = self.repr(batch_state)

        for t in range(unroll_steps):
            # for each element in the unroll loop we iterate over each batch and compute its state
            batch_policy_predicted, batch_value_predicted = self.policy(current_state)
            batch_policy_targets = torch.stack(
                [torch.from_numpy(trajectory[t].mcts_policy) for trajectory in batch]
            ).to(device)
            batch_value_targets = (
                torch.Tensor(
                    [
                        player_to_reward(
                            trajectory[t].player,
                            self.experience_buffer.game_store[trajectory[t].game_id],
                        )
                        for trajectory in batch
                    ]
                )
                .view(-1, 1)
                .to(device)
            )
            policy_term = kl_loss_fn(
                torch.log(batch_policy_predicted), batch_policy_targets
            )
            value_term = value_loss_fn(batch_value_predicted, batch_value_targets)
            loss += policy_term + value_term

            actions_tensor = torch.stack(
                [torch.from_numpy(trajectory[t].action) for trajectory in batch]
            ).to(device)
            current_state = self.dynamics(current_state, actions_tensor)

        # scale loss by 1/K to ensure gradient has magnitude invariant to unroll-steps
        loss = loss / unroll_steps

        self.loss_window.append(loss.item())
        loss.backward()
        self.optimizer.step()
