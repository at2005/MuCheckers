# multithreaded mcts search
import multiprocessing as mp
import numpy as np
import torch
from simulator import CheckerBoard
from infra import master_store
from models import create_input,history_dim
import os
import random

# alphazero values
c1 = 1.25
c2 = 19652
# the number of iterations to run mcts
num_iters = 800

def play_game(board : CheckerBoard):
    game_id = os.getpid() 
    player_bit = 1 
    max_num_iters = 100
    counter = 0
    history = []
    while not board.game_over() and counter < max_num_iters:
        player = "white" if player_bit else "black"
        board_tensor = board.as_tensor()
        # list of indices corresponding to valid actions, out of the total 10_000
        legal_action_ids, legal_actions = board.get_valid_actions()

        # what to do if history not yet of history_dim? take random actions
        if len(history) < history_dim:
            action = random.choice(legal_action_ids) 
        else:
            # we create a state that has the player as the last plane
            state_with_player = create_input(board_tensor, player)
            # compute the initial hidden state
            init_hidden_state = master_store["batch_store"].repr_fn(state_with_player).result()
            # TODO handle actions properly
            action_idx, mcts_policy = run_mcts(init_hidden_state, legal_action_ids, game_id)
            action = legal_actions[action_idx]
            # we don't store the hidden state, but the actual board state
            master_store["experience_store"].add_experience(game_id, state_with_player, mcts_policy)

        # actually execute that action, ie materialises it on the board
        board.execute(action, player)

        # switch player
        player_bit ^= 1

        history.append(board_tensor)
        history = history[-history_dim:]
        counter += 1

def run_mcts(root_hidden_state : torch.Tensor, legal_actions, game_id):
    root = MCTSNode(root_hidden_state, None)
    for _ in range(num_iters):
        root.traverse()
    
    # normalised counts == mcts policy 
    masked_policy = [root.children[action].count if action in legal_actions else 0 for action in range(root.num_actions)]
    denominator = sum(masked_policy)
    norm_masked_policy = [val / denominator for val in masked_policy]

    # select valid action with highest visit count 
    action_idx = np.argmax(np.array(norm_masked_policy)) 
    return action_idx, norm_masked_policy 


class MCTSNode:
    def __init__(self, state, parent):
        self.is_root = parent is None
        # hidden state
        self.state = state 
        # map of action -> child_node
        self.children = {}
        self.parent : MCTSNode = parent
        self.count = 0
        self.q_value = 0

        self.num_actions = 10_000
        # P(s,a) for all actions
        self.child_priors, self.value = master_store["batch_store"].policy_fn(self.state).result()
    
    def __repr__(self):
        print(f"Num Children: {len(self.children)}")
        print(f"Visit Count: {self.count}")
        print(f"Q-value: {self.q_value}")
        print(f"Root? : {self.is_root}")
        print(f"Policy : {self.child_priors}")
        print(f"V(s): {self.value}")

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

    def backprop(self, value_to_propagate=None):
        actual_value = self.value if not value_to_propagate else value_to_propagate
        self.q_value = (self.q_value * self.count + actual_value) / (self.count + 1)
        self.count += 1
        if self.parent:
            self.parent.backprop(actual_value)
    
    def is_leaf(self):
        return not bool(len(self.children))
    
    def expand(self):
        # submit all, they are all independent so we dont need to await them
        futures = [master_store["batch_store"].dynamics_fn(self.state, action) for action in range(self.num_actions)]
        for action, future in enumerate(futures):
            new_hidden_state = future.result()
            self.children[action] = MCTSNode(new_hidden_state, parent=self)

    def traverse(self):
        if self.is_leaf():
            self.backprop()
            self.expand()
            return
        
        # if we are not a leaf node, we wanna fetch 
        # the next node to go down
        next_action = self.select()
        next_node = self.children[next_action]
        next_node.traverse()