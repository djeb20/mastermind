import numpy as np
from mastermind import mastermind

env = mastermind(action_type='guess', reward_struc='basic', num_goals=10)

# from dqn import Agent
# agent = Agent(env, state_dim=env.state_dim, num_actions=env.action_dim,
#                  critic_arch=[128, 64], buffer_size=1e6, batch_size=64, decay=0.999,
#                  gamma=0.99, step_size=1e-4, 
#                  tau=0.001, epsilon_start=1, epsilon_min=0.05)

# agent.fill_buffer(exp=True, multi=False, prop=1)
# agent.train(1e7)

from ppo_agent import Agent
agent = Agent(env, state_dim=env.state_dim, num_actions=env.action_dim,
                 actor_arch=[128, 64], critic_arch=[128, 64],
                 actor_rate=1e-4, critic_rate=1e-4,
                 gamma = 0.99, lam=0.95, 
                 epsilon_clip=0.2, entropy_coef=0.01)

agent.train(num_ite=100, num_epochs=30, num_p=40, steps_per_agent=100, batch_size=64)