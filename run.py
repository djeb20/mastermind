from doctest import master
import numpy as np
from mastermind import mastermind
from dqn import DQN
from tqdm import tqdm

env = mastermind()
agent = DQN(env.state_dim, env.action_dim,
                 critic_arch=[64, 32], buffer_size=10000, batch_size=64,
                 gamma=0.99, epsilon=0.05, step_size=1e-4, tau=0.001)

num_episodes = int(1e7)

for _ in tqdm(range(num_episodes)):

    state = env.reset()

    while True:

        action = agent.choose_action(state)
        new_state, reward, done, _ = env.step(action)

        agent.store((state, action, reward, new_state, 1 - done))
        batch = agent.get_batch()
        agent.update_critic(*batch)
        agent.update_target()

        if done: break

        state = new_state

# Test the new state

state = env.reset()
env.goal = env.test_goal

while True:

    action = agent.choose_action(state, exp=False)
    new_state, reward, done, _ = env.step(action)

    if done: break

print(env.count // 4)

