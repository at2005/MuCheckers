# multithreaded mcts search
import asyncio
import numpy as np
import torch
from simulator import fetch_legal_actions

# alphazero values
c1 = 1.25
c2 = 19652
# the number of iterations to run mcts
num_iters = 800

def run_mcts(board_state, repr_fn):
    # compute hidden state
    init_state = repr_fn(board_state)
    root = MCTSNode(init_state, is_root=True)
    for i in range(num_iters):
        root.traverse()
    
    # select child node with greatest visit count
    action,_count = max(root.children.items(), key=lambda x: x[1].count)
    return action

class MCTSNode:
    def __init__(self, state, parent, repr_net, policy_net, dynamics_net, legal_actions=None):
        # hidden state
        self.state = state
        # map of action -> node
        self.children = {}
        self.parent = parent
        self.count = 0
        self.q_value = 0
        self.repr_net = repr_net
        self.policy_net = policy_net
        self.dynamics_net = dynamics_net
        self.is_root = legal_actions is not None 
        self.num_actions = len(legal_actions) if self.is_root else policy_net.num_actions 
        # P(s,a) for all actions
        self.child_priors = policy_net(self.state).cpu().numpy()
    
    def __repr__(self):
        print(f"Num Children: {len(self.children)}")
        print(f"Visit Count: {self.count}")
        print(f"Q-value: {self.q_value}")
        print(f"Root? : {"yes" if self.is_root else "no"}")

    # selects which path to go down
    def select(self):
        sum_counts = sum([self.children[action].count for action in self.children])
        sum_counts_sq = np.sqrt(sum_counts)
        scaling_factor = c1 + np.log((sum_counts + c2 + 1) / c2)
        score_arr = []
        for action in range(self.num_actions):
            score = self.child_priors[action] * sum_counts_sq * scaling_factor
            if action in self.children:
                child = self.children[action]
                score /= (1 + child.count)
                score += child.q
            score_arr.append(score)
        action = np.argmax(np.array(score_arr))
        return action

    def backprop(self, terminal_reward):
        # value update rule
        self.q_value = (self.q_value * self.count + terminal_reward) / (self.count + 1)
        self.count += 1
        if self.parent:
            self.backprop(self.parent, terminal_reward) 
    
    def is_leaf(self):
        return bool(len(self.children))
    
    def expand(self):
        # TODO: lazily create state
        self.children = {action : MCTSNode(state=self.dynamics_net(self.state, action), 
                                           parent=self, 
                                           repr_net=self.repr_net, 
                                           policy_net=self.policy_net, 
                                           dynamics_net=self.dynamics_net) for action in range(self.num_actions)}

    def traverse(self):
        if self.is_leaf():
            self.backprop()
            self.expand()
    