from models import RepresentationNet, DynamicsNet, PolicyNet, device
import torch
import multiprocessing as mp
import numpy as np
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

        self.policy_results = defaultdict(mp.Queue)
        self.dynamics_results = defaultdict(mp.Queue)
        self.repr_results = defaultdict(mp.Queue)


    def process_policy(self):
        pids, inputs = zip(*[self.policy_queue.get() for _ in range(self.batch_size)])
        batched_tensor = torch.from_numpy(np.array(inputs)).to(device)
        
        policies, values = self.policy_net(batched_tensor).cpu().numpy()

        for pid, policy, value in zip(pids, policies, values):
            process_queue = self.policy_results[pid]
            process_queue.put((policy, value))


    def process_dynamics(self):
        pids, hidden_states, actions = zip(*[self.dynamics_queue.get() for _ in range(self.batch_size)])
        batched_hidden_states = torch.from_numpy(np.array(hidden_states)).to(device)
        batched_actions = torch.from_numpy(np.array(actions)).to(device)
        results = self.dynamics_net(batched_hidden_states, batched_actions).cpu().numpy()
        for (pid, res) in zip(pids, results):
            process_queue = self.dynamics_results[pid]
            process_queue.put(res)

            

    def process_repr(self):
        pids, inputs = zip(*[self.repr_net.get() for _ in range(self.batch_size)])
        batched_tensor = torch.from_numpy(np.array(inputs)).to(device)
        reprs = self.repr_net(batched_tensor).cpu().numpy()
        for (pid, repr) in zip(pids, reprs):
            process_queue = self.repr_results[pid]
            process_queue.put(repr)
        

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
    def __init__(self, pid, experience_queue, game_queue, policy_queue, dynamics_queue, repr_queue, policy_result, repr_result, dynamics_result, game_counter):
        self.pid = pid
        self.experience_queue : mp.Queue = experience_queue
        self.game_queue : mp.Queue = game_queue
        self.policy_queue : mp.Queue = policy_queue
        self.dynamics_queue : mp.Queue = dynamics_queue
        self.repr_queue : mp.Queue = repr_queue
        self.game_counter : Synchronized = game_counter

        self.policy_result : mp.Queue = policy_result
        self.dynamics_result : mp.Queue = dynamics_result
        self.repr_result : mp.Queue = repr_result
    
    def poll_policy(self):
        return self.policy_result.get()

    def poll_dynamics(self):
        return self.dynamics_result.get()

    def poll_repr(self):
        return self.repr_result.get()

    def repr_fn(self, x):
        self.repr_queue.put((self.pid, x))        

    def policy_fn(self, x):
        self.policy_queue.put((self.pid, x))

    def dynamics_fn(self, h, a):
        self.dynamics_queue.put((self.pid, h, a))
        

    
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