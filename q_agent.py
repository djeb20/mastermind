from collections import defaultdict
import numpy as np
from tqdm import tqdm

class q_agent:
    
    def __init__(self, env, state_dim, action_dim, epsilon=0.05, gamma=0.99, alpha=0.2, tol=1e-7):
        
        # Agent hyper-parameters
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.actions = np.arange(self.action_dim)
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        # Tolerance to stop training
        self.tol = tol

        # Environment and coalition
        self.env = env
        
        self.Q_table = defaultdict(lambda: np.zeros(self.action_dim))
        
    def choose_action(self, state, exp=True):
        """
        Chooses action with epsilon greedy policy.
        """

        if np.random.rand() < self.epsilon and exp: return np.random.randint(self.action_dim)
        else: return self.best_action(state)

    def best_action(self, state):
        """
        Finds the greedy best action.
        Can only choose from valid "actions"
        """

        q_values = self.Q_table[tuple(state)]

        return np.random.choice(self.actions[q_values == q_values.max()])
        
    def update(self, state, action, new_state, reward, done):

        # I don't want terminal state to ever appear in q_table
        if done: q_max = 0
        else: q_max = self.Q_table[tuple(new_state)].max()
        
        # Usual update, for only valid "actions"
        td_error = reward + self.gamma * q_max - self.Q_table[tuple(state)][action]
        self.Q_table[tuple(state)][action] += self.alpha * td_error
        
        # Return a update error to stop training.
        return td_error   

    def train(self, num_episodes, scale=1000):
        """
        Trains one agent using Q-Learning.
        """

        # First is TD error, next is sum of TD error, sum of squares, length of list
        errors = [[], 0, 0, 0]
        stds = [np.inf]
        returns = []
        steps = []

        for i in tqdm(range(num_episodes)):

            state, _ = self.env.reset()
            ret = 0

            while True:

                # If GYM environment, might need this saved state
                if str(type(self.env)) == "<class 'gym.wrappers.time_limit.TimeLimit'>": self.save_state(state)
                elif str(type(self.env)) == "<class '__main__.FactoredState'>": self.save_state(state)

                # Usual RL, choose action, execute, update
                action = self.choose_action(state)
                new_state, reward, done, _, _ = self.env.step(action)
                update = self.update(state, action, new_state, reward, done)
                state = new_state

                ret += reward

                # Track errors
                errors[0].append(abs(update)); errors[1] += abs(update); errors[2] += update ** 2; errors[3] += 1
                stds.append((errors[2] / errors[3]) - (errors[1] / errors[3]) ** 2)

                if done: break

            returns.append(ret)
            steps.append(self.env.count // 4)

            if i % scale == 0:
                print()
                print('Average td error: {:0.04f}, STD of STD: {}, Average ret: {:0.02f}, Average steps: {:0.02f}, Num_states: {}'.format(
                    np.mean(errors[0][-scale:]),
                    np.std(stds[-scale:]),
                    np.mean(returns[-scale:]),
                    np.mean(steps[-scale:]),
                    len(self.Q_table)))

            # # Learning converges according to tolerance
            # if np.std(stds[-scale:]) < self.tol: break

        print()
        print('Average td error: {:0.04f}, STD of STD: {}, Average ret: {:0.02f}, Average steps: {:0.02f}'.format(
                    np.mean(errors[0][-scale:]),
                    np.std(stds[-scale:]),
                    np.mean(returns[-scale:]),
                    np.mean(steps[-scale:])))

        # # Top line is for normal, bottom for multiprocessing
        # return dict(self.Q_table)