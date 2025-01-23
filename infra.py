from models import RepresentationNet, DynamicsNet, PolicyNet, device
from simulator import CheckerBoard
from search import play_game
import torch
import multiprocessing as mp
import numpy as np
from concurrent.futures import Future

def fetch_num_gpus():
    if device == "mps":
        return 1
    return torch.cuda.device_count()

class BatchEngine:
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

class ReplayBuffer():
    def __init__(self):
        self.repr_net = RepresentationNet().to(device)
        self.dynamics_net = DynamicsNet().to(device)
        self.policy_net = PolicyNet().to(device)
        self.store = mp.Queue()
        self.game_store = mp.Queue()
    
    def fetch_latest_weights(self):
        return (self.repr_net, self.dynamics_net, self.policy_net)

    # what do we really need to store here? 
    # we want to store the root game state, the MCTS generated policy vector
    # and the game_id
    def add_experience(self, game_id, root_state, mcts_policy, player):
        self.store.put((game_id, root_state, mcts_policy, player))

 
def init_pool(master_store_,):
    global master_store
    master_store = master_store_

def multithreaded_search():
    num_processes = 1#mp.cpu_count()
    with mp.Manager() as manager:
        master_store = manager.dict()
        master_store["experience_buffer"] = ReplayBuffer()
        master_store["batch_store"] = BatchEngine()
        with mp.Pool(processes=num_processes, initializer=init_pool, initargs=(master_store,)) as pool:
            new_games = [CheckerBoard() for _ in range(num_processes)]
            pool.map(play_game, new_games)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    multithreaded_search()