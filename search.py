# multithreaded mcts search
import multiprocessing as mp
import numpy as np
import torch
from simulator import CheckerBoard
from infra import ExperienceStore, BatchStore, DistributedQueues
from models import create_input,history_dim
import random
import logging
import asyncio

# alphazero values
c1 = 1.25
c2 = 19652
# the number of iterations to run mcts
num_iters = 800
num_games = 1000


# this is a single process basically
def play_n_games(buffer : DistributedQueues):
    for _ in range(num_games):
        game_id = buffer.new_game()
        logging.debug(f"Starting game {game_id}")
        board = CheckerBoard()
        # play game to completion
        winner = play_game(board, game_id, buffer)
        buffer.add_game_outcome(game_id, winner)
        logging.debug(f"Game over for {buffer.pid} with winner {winner}")


def play_game(board : CheckerBoard, game_id, store : DistributedQueues, max_timesteps=100) -> str:
    player_bit = 1 
    timestep = 0
    history : list[torch.Tensor] = []
    draw_possibility = False

    while not board.game_over() and timestep < max_timesteps:
        player = "white" if player_bit else "black"
        board_tensor = board.as_tensor()
        # list of indices corresponding to valid actions, out of the total 10_000
        legal_action_map = board.get_valid_actions(player)
        legal_action_ids = list(legal_action_map.keys())

        # if there are no legal moves i can make, return the previous player
        if not len(legal_action_map):
            if draw_possibility:
                return "draw"
            draw_possibility = True
            continue            
        if draw_possibility:
            return player
            
        # what to do if history not yet of history_dim? take random actions
        if len(history) < history_dim:
            action_idx = random.choice(legal_action_ids) 
            action = legal_action_map[action_idx]
        else:
            # we create a state that has the player as the last plane
            state_with_player = create_input(board_tensor, player)
            # compute the initial hidden state
            store.repr_fn(state_with_player)
            logging.debug("Polling repr function")
            init_hidden_state = store.poll_repr()

            # fetch action and policy
            logging.debug("Running MCTS")
            action, mcts_policy = run_mcts(init_hidden_state, legal_action_map, store)
            # we don't store the hidden state, but the actual board state
            store.add_experience(game_id, timestep, state_with_player, mcts_policy, player)

        # actually execute that action, ie materialises it on the board
        # action is a src -> dest tensor
        board.execute(action, player)

        # switch player
        player_bit ^= 1

        history.append(board_tensor)
        history = history[-history_dim:]
        timestep += 1
        logging.debug("Timestep {timestep} completed")
    
    winner = board.who_won()
    logging.debug(f"Winner {winner}")
    return winner

def run_mcts(root_hidden_state : torch.Tensor, legal_action_map, store: DistributedQueues):
    root = MCTSNode(state=root_hidden_state, parent=None, store=store)
    for _ in range(num_iters):
        root.traverse()
    
    # normalised counts == mcts policy 
    masked_policy = np.array([root.children[action].count if action in legal_action_map else 0 for action in range(root.num_actions)])
    denominator = np.sum(masked_policy)
    norm_masked_policy = masked_policy / denominator

    # select valid action with highest visit count 
    action_idx = np.argmax(norm_masked_policy) 
    action = legal_action_map[action_idx]
    return action, norm_masked_policy 


class MCTSNode:
    def __init__(self, state, parent, store: DistributedQueues):
        self.store = store
        self.is_root = parent is None
        # hidden state
        self.state = state 
        # map of action -> child_node
        self.children : dict[str, MCTSNode] = {}
        self.parent : MCTSNode = parent
        self.count = 0
        self.q_value = 0

        # max possible actions, we then mask this at the root
        # to allow only legal ones
        self.num_actions = 10_000
        # P(s,a) for all actions
        self.store.policy_fn(self.state)
        self.action_policy, self.value = self.store.poll_policy()
    
    def __repr__(self):
        print(f"Num Children: {len(self.children)}")
        print(f"Visit Count: {self.count}")
        print(f"Q-value: {self.q_value}")
        print(f"Root? : {self.is_root}")
        print(f"Policy : {self.action_policy}")
        print(f"V(s): {self.value}")

    # selects which path to go down
    def select(self):
        # total sum of counts for each child, ie the total number of traversals
        sum_counts = sum([self.children[action].count for action in self.children])

        sum_counts_sqrt = np.sqrt(sum_counts)

        # UCB stuff
        scaling_factor = c1 + np.log((sum_counts + c2 + 1) / c2)
        score_arr = []

        for action in range(self.num_actions):
            # we scale out "prior" the number of times we have encountered the node, ie 
            # weight heavily if we have encountered it before
            score = self.action_policy[action] * sum_counts_sqrt * scaling_factor
            
            # this check exists bc at the root node we mask out all invalid actions
            # so not all actions have a corresponding entry in children
            if action in self.children:
                child = self.children[action]
                score /= (1 + child.count)
                score += child.q_value
            score_arr.append(score)
        action = np.argmax(np.array(score_arr))
        return action

    def backprop(self, value_to_propagate=None):
        actual_value = self.value if not value_to_propagate else value_to_propagate
        self.q_value = (self.q_value * self.count + actual_value) / (self.count + 1)
        self.count += 1
        if self.parent:
            self.parent.backprop(value_to_propagate=actual_value)
    
    def is_leaf(self):
        return not bool(len(self.children))
    
    def add_child(self, action, state):
        self.children[action] = MCTSNode(state, parent=self, store=self.store)
        
    
    def expand(self):
        # submit all, they are all independent so we dont need to await them
        for action in range(self.num_actions):
            self.store.dynamics_fn(self.state, CheckerBoard.action_idx_to_board(action))
        
        for action in range(self.num_actions):
            new_hidden_state = self.store.poll_dynamics()
            self.add_child(action, new_hidden_state)

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

def parallel_search():
    num_processes = 1#mp.cpu_count()
    experience_store = ExperienceStore()
    batch_store = BatchStore()
    xp_processes = [mp.Process(target=play_n_games, args=(DistributedQueues(
                                                            pid=i,
                                                            experience_queue=experience_store.store[i],
                                                            game_queue=experience_store.game_store,
                                                            policy_queue=batch_store.policy_queue,
                                                            dynamics_queue=batch_store.dynamics_queue, 
                                                            repr_queue=batch_store.repr_queue,
                                                            policyq_sz=batch_store.policy_queue_sz,
                                                            dynamicsq_sz=batch_store.dynamics_queue_sz,
                                                            reprq_sz=batch_store.repr_queue_sz,
                                                            policy_result=batch_store.policy_results[i],
                                                            dynamics_result=batch_store.dynamics_results[i],
                                                            repr_result=batch_store.repr_results[i],
                                                            game_counter=experience_store.game_counter),)) for i in range(num_processes)]
    logging.debug("Starting all parallel tree search")
    for p in xp_processes:
        p.start()

    return batch_store, experience_store, xp_processes

async def main():
    batch_store, xp_store, xp_processes = parallel_search()    
    # now we setup up three coroutines. these can be coroutines bc they are
    # io bound, not compute bound, ie they only need to submit jobs to the GPU
    # and write results into a queue
    await asyncio.gather(batch_store.process_dynamics(), 
                         batch_store.process_policy(), 
                         batch_store.process_repr(), 
                         batch_store.update_weights(),
                         xp_store.update_game_store(),
                         xp_store.write_queue_to_list())    


if __name__ == '__main__':
    mp.freeze_support()
    asyncio.run(main())