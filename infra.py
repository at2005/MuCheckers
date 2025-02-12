from models import RepresentationNet, DynamicsNet, PolicyNet, device
import torch
import multiprocessing as mp
import numpy as np
from concurrent.futures import Future
from collections import defaultdict
from multiprocessing.sharedctypes import Synchronized

def fetch_num_gpus():
    if device == "mps":
        return torch.mps.device_count()
    return torch.cuda.device_count()

class BatchStore:
    def __init__(self):
        self.batch_size = 32 * fetch_num_gpus()
        self.policy_queue = mp.Queue(maxsize=self.batch_size)
        self.dynamics_queue = mp.Queue(maxsize=self.batch_size)
        self.repr_queue = mp.Queue(maxsize=self.batch_size)

        self.policy_net = PolicyNet().to(device)
        self.repr_net = RepresentationNet().to(device)
        self.dynamics_net = DynamicsNet().to(device)

    def process_policy(self):
        futures, inputs = zip(*[self.policy_queue.get() for _ in range(self.batch_size)])
        batched_tensor = torch.from_numpy(np.array(inputs)).to(device)
        policies, values = self.policy_net(batched_tensor).cpu().numpy()
        for future, policy, value in zip(futures, policies, values):
            future.set_result((policy, value))


    def process_dynamics(self):
        futures, hidden_states, actions = zip(*[self.dynamics_queue.get() for _ in range(self.batch_size)])
        batched_hidden_states = torch.from_numpy(np.array(hidden_states)).to(device)
        batched_actions = torch.from_numpy(np.array(actions)).to(device)
        results = self.dynamics_net(batched_hidden_states, batched_actions).cpu().numpy()
        for (future, res) in zip(futures, results):
            future.set_result(res)

    def process_repr(self):
        futures, inputs = zip(*[self.repr_net.get() for _ in range(self.batch_size)])
        batched_tensor = torch.from_numpy(np.array(inputs)).to(device)
        reprs = self.repr_net(batched_tensor).cpu().numpy()
        for (future, repr) in zip(futures, reprs):
            future.set_result(repr)


class ExperienceStore:
    def __init__(self):
        self.repr_net = RepresentationNet().to(device)
        self.dynamics_net = DynamicsNet().to(device)
        self.policy_net = PolicyNet().to(device)
        self.store = defaultdict(mp.Queue)
        self.game_store = defaultdict(str)
        self.game_counter = mp.Value("i", 0, lock=True)
            
    def fetch_latest_weights(self):
        return (self.repr_net, self.dynamics_net, self.policy_net)


class DistributedQueues:
    def __init__(self, experience_queue, game_queue, policy_queue, dynamics_queue, repr_queue, game_counter):
        self.experience_queue : mp.Queue = experience_queue
        self.game_queue : mp.Queue = game_queue
        self.policy_queue : mp.Queue = policy_queue
        self.dynamics_queue : mp.Queue = dynamics_queue
        self.repr_queue : mp.Queue = repr_queue
        self.game_counter : Synchronized = game_counter 
    
    def repr_fn(self, x):
        future = Future()
        self.repr_queue.put((future, x))
        return future
    
    def policy_fn(self, x):
        future = Future()
        self.policy_queue.put((future, x))
        return future
        
    def dynamics_fn(self, h, a):
        future = Future()
        self.dynamics_queue.put((future, h, a))
        return future
    
    def add_experience(self, game_id, timestep, state, mcts_policy, player):
        self.experience_queue.put((game_id, timestep, state, mcts_policy, player))

    def new_game(self):
        with self.game_counter.get_lock():
            game_id = self.game_counter.value
            self.game_counter.value += 1
            return game_id

    def add_game_outcome(self, game_id, winner):
        self.game_queue.put((game_id, winner))

    def end_game(self):
        self.add_experience(game_id=None, time=None, state=None, mcts_policy=None, player=None)