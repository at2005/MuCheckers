from models import RepresentationNet, DynamicsNet, PolicyNet, device
from simulator import CheckerBoard
from search import play_n_games 
import torch
import multiprocessing as mp
import numpy as np
from concurrent.futures import Future
from typing import TypedDict, cast
from multiprocessing.managers import DictProxy


def fetch_num_gpus():
    if device == "mps":
        return 1
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

class ExperienceStore():
    def __init__(self):
        self.repr_net = RepresentationNet().to(device)
        self.dynamics_net = DynamicsNet().to(device)
        self.policy_net = PolicyNet().to(device)
        self.store = {}
        self.game_store = {}
        self.game_counter = 0
        self.counter_lock = mp.Lock()
    
    def init_buffer(self, pid):
        if pid not in self.store:
            self.store[pid] = mp.Queue() 
        
    def new_game(self):
        with self.counter_lock:
            game_id = self.game_counter
            self.game_counter += 1
            return game_id
        
    def end_game(self, pid):
        self.add_experience(pid, game_id=None, time=None, state=None, mcts_policy=None, player=None)
        
    def add_game_outcome(self, game_id, winner):
        self.game_store[game_id] = winner 
    
    def fetch_latest_weights(self):
        return (self.repr_net, self.dynamics_net, self.policy_net)

    # what do we really need to store here? 
    # we want to store the root game state, the MCTS generated policy vector
    # and the game_id
    def add_experience(self, pid, game_id, timestep, state, mcts_policy, player):
        self.store[pid].put((game_id, timestep, state, mcts_policy, player))


class Store(TypedDict):
    experience_store : ExperienceStore
    batch_store : BatchStore

def get_experience_store() -> ExperienceStore:
    return master_store["experience_store"]

def get_batch_store() -> BatchStore:
    return master_store["batch_store"]

def init_pool(master_store_,):
    global master_store
    master_store : DictProxy[str, Store] = master_store_

def parallel_search():
    num_processes = 1#mp.cpu_count()
    with mp.Manager() as manager:
        master_store: DictProxy[str, Store] = manager.dict()
        master_store["experience_store"] = ExperienceStore(num_processes)
        master_store["batch_store"] = BatchStore()
        with mp.Pool(processes=num_processes, initializer=init_pool, initargs=(master_store,)) as pool:
            # dummy data
            pool.map(play_n_games, range(num_processes))

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    parallel_search()