from infra import get_experience_store
from models import PolicyNet, RepresentationNet, DynamicsNet, unroll_steps
import torch
import torch.nn as nn

def unroll_and_avg():
    pass

def training_loop():
    # request batch from experience buffer
    experience_store = get_experience_store
    batch = experience_store.fetch_batch()

    for item in batch:
        game_id, state, target_policy = item
        experience_store.sample_game(game_id, unroll_steps)
        pass


    pass