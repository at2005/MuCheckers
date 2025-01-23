# multithreaded mcts search
import multiprocessing as mp
import numpy as np
import torch
from simulator import CheckerBoard
from infra import master_store

# alphazero values
c1 = 1.25
c2 = 19652
# the number of iterations to run mcts
num_iters = 800

def play_game(board : CheckerBoard):
    player_bit = 1 
    max_num_iters = 1000
    counter = 0
    while not board.game_over() and counter < max_num_iters:
        player = "white" if player_bit else "black"
        action = run_mcts(board, player) 
        board.execute(action, player)
        player_bit ^= 1
        counter += 1

def run_mcts(board : CheckerBoard, player):
    # compute hidden state
    init_state = master_store["batch_store"].repr_fn(board.as_tensor()).result()
    legal_actions_init = board.get_valid_actions()
    root = MCTSNode(init_state, None, len(legal_actions_init))
    for _ in range(num_iters):
        root.traverse()
    
    # select child node with greatest visit count
    action, _ = max(root.children.items(), key=lambda x: x[1].count)
    return legal_actions_init[action]


class MCTSNode:
    def __init__(self, state, parent, num_legal_actions=None):
        self.is_root = num_legal_actions is not None 
        # hidden state
        self.state = master_store["batch_store"].repr_fn(state).result() if self.is_root else state
        # map of action -> node
        self.children = {}
        self.parent : MCTSNode = parent
        self.count = 0
        self.q_value = 0

        self.num_actions = num_legal_actions if self.is_root else 10_000
        # P(s,a) for all actions
        self.child_priors, self.value = master_store["batch_store"].policy_fn(self.state).result()
    
    def __repr__(self):
        print(f"Num Children: {len(self.children)}")
        print(f"Visit Count: {self.count}")
        print(f"Q-value: {self.q_value}")
        print(f"Root? : {self.is_root}")

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