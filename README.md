### MuZero for International Checkers

Mainly a project to learn:
- How the MuZero algorithm works
- How RL infra works in general, including distributed experience collection, inference process pool etc. 

Each process is responsible for initiating games and playing them to completion. This involves running MCTS from the root node. During search each process queues up a job (a call to the policy, representation or dynamics functions) for the inference processes during MCTS. When the appropriate batch size is reached, we flush the queue and write the results back into another queue.

Each experience (state, action, mcts_policy, etc) is added to the experience buffer which the training coroutine uses to train the policy, representation and dynamics functions. We sample a trajectory from the experience buffer up to five "unroll" steps, and compute a mean KL_div loss between the predicted policy and the computed MCTS policy, as well as a mean MSE loss between the predicted value function and the terminal reward (ie winning or losing the game). This is done for each unroll step, and we propagate each state forward using the dynamics function.

This project uses the default Python multiprocessing library but it has been suggested I use Trio – maybe someday I'll rewrite to use it.

Notes:
- I've been testing this on MacOS, which doesn't support sem_getvalue. I manually employ semaphores to get the size of inference job queues
- This is taking a while to train, so I'll hit pause on it for now.

TODO:
- need to fix promotion