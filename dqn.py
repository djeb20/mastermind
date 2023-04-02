# Required libraries
import tensorflow as tf
import numpy as np
import keras
from keras.layers import Dense
from keras.layers import Input
from tqdm import tqdm
import pickle
import copy
from collections import defaultdict
from multiprocessing import Manager, Process
import itertools

class Agent:
    
    def __init__(self, env, state_dim, num_actions,
                 critic_arch=[128, 64], buffer_size=1e6, batch_size=64, decay=0.9999,
                 gamma=0.99, step_size=1e-4, 
                 tau=0.001, epsilon_start=1, epsilon_min=0.05):
        
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.actions = np.arange(self.num_actions)

        # Make the critic and target critic
        self.critic = self.make_critic(critic_arch)
        self.target_critic = self.make_critic(critic_arch)
        self.target_critic.set_weights(self.critic.get_weights())

        # Pre saved good weights
        # self.critic.load_weights('agent_models/critic')
        # self.target_critic.load_weights('agent_models/target_critic')
        
        # Step the optimiser used and its stepsize.
        self.optimizer_critic = tf.keras.optimizers.Adam(step_size)
        
        # Create an empty replay buffer
        self.create_buffer(int(buffer_size))
        
        # Save hyper-parameters
        self.gamma = gamma
        self.buffer_size = int(buffer_size)
        self.batch_size = batch_size
        self.tau = tau
        self.decay = decay

        self.epsilon_min = epsilon_min
        # with open('epsilon.pkl', 'rb') as file: self.epsilon = pickle.load(file)
        self.epsilon = epsilon_start

        self.scale = 10
        self.rets = np.full(self.scale, -1e6)
        self.ret_count = 0

        # Environment
        self.env = env
        
    def make_critic(self, critic_arch):
        """
        Creates the Q network for DQN
        """
    
        # Needs an input layer
        inp_state = Input(shape=(self.state_dim,))
        
        # Hidden layers
        out = Dense(critic_arch[0], activation='relu')(inp_state)
        for size in critic_arch[1:]:
            out = Dense(size, activation='relu')(out)

        # Output layer
        out = Dense(self.num_actions)(out)
        
        return keras.Model(inp_state, out)
    
    def create_buffer(self, buffer_size):
        """
        Creates a new memory based on the number of transitions being recorded
        """
        
        # Count to keep track of size of memory
        self.count = 0
        
        # Initiate empty buffer
        self.buffer = {'obs': np.empty((buffer_size, self.state_dim)),
                       'action': np.empty((buffer_size, )),
                       'reward': np.empty((buffer_size, )),
                       'new_obs': np.empty((buffer_size, self.state_dim)),
                       'term': np.empty((buffer_size, ))}
        
    def store(self, tup):
        """ 
        Stores an event tuple in the buffer.
        Taken from RL project code.
        """
        
        index = self.count % self.buffer_size

        self.buffer['obs'][index] = tup[0]
        self.buffer['action'][index] = tup[1]
        self.buffer['reward'][index] = tup[2]
        self.buffer['new_obs'][index] = tup[3]
        self.buffer['term'][index] = tup[4]

        self.count += 1
        
    def get_batch(self):
        """
        Selects a batch from the buffer.
        Using a selected trajectory.
        """
            
        # Index of the randomly chosen transitions        
        index = np.random.choice(np.arange(np.min([self.count, self.buffer_size])), self.batch_size)

        # We have a different index for values as we need the next one too.
        batch = [self.buffer['obs'][index], 
                 self.buffer['action'][index], 
                 self.buffer['reward'][index], 
                 self.buffer['new_obs'][index],
                 1 - self.buffer['term'][index]]
        
        # Make tensors for speed.
        t_trajectory = [tf.convert_to_tensor(item) for item in batch]
        t_trajectory[1] = tf.cast(t_trajectory[1], dtype=tf.int32)
        t_trajectory[2] = tf.cast(t_trajectory[2], dtype=tf.float32)
        t_trajectory[4] = tf.cast(t_trajectory[4], dtype=tf.float32)

        return (t_trajectory[i] for i in range(len(batch)))

    def update(self):
        """
        Performs one update for the agent.
        """

        # Select a batch and update
        self.update_critic(*self.get_batch())
        self.update_target()  

    @tf.function
    def update_critic(self, old_states, actions, rewards, new_states, terms):
        """
        Function to update the critic network
        """

        # Indexes of the buffer, 0 to batch size
        b_size = tf.shape(old_states)[0]
        buf_ind = tf.range(b_size)
        
        # if self.model_arch == 0:      
        with tf.GradientTape() as g:
            
            Q_max = tf.math.reduce_max(self.target_critic(new_states, training=True), axis=1)            
            Q_values = self.critic(old_states, training=True)
            
            Q_values_actions = tf.gather_nd(Q_values, tf.stack((buf_ind, actions), -1))
            
            td_error = rewards + self.gamma * Q_max * terms - Q_values_actions

            # Calculate loss
            loss = tf.math.reduce_mean(tf.math.square(td_error))

        # Calculates gradient and then optimises using it (moving in that direction)
        gradient = g.gradient(loss, self.critic.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(gradient, self.critic.trainable_variables))
        
    @tf.function
    def update_target(self):
        """
        This slowly updates the target network
        """
        
        for w, w_t in zip(self.critic.variables, self.target_critic.variables):
            w_t.assign(w * self.tau + w_t * (1 - self.tau))
        
    def choose_action(self, state, exp=True):
        """
        Given a state returns a chosen action given by policy.
        Has epsilon greedy exploration.
        """
        
        if (np.random.rand() < self.epsilon) and (exp == True): return self.env.sample()
        else: return self.critic(state.reshape(1, -1))[0].numpy().argmax()
            # q_values = self.critic(state.reshape(1, -1))[0].numpy()
            # return np.random.choice(self.actions[np.isclose(q_values, q_values.max(), rtol=1e-2)])

    def fill_buffer(self, exp=True, multi=True, n_p=30, prop=1):

        if multi: self.fill_buffer_multi(exp, n_p, prop)
        else: self.fill_buffer_single(exp, prop)

    def fill_buffer_single(self, exp, prop):

        self.count = 0

        state, info = self.env.reset()
        for _ in tqdm(range(int(self.buffer_size * prop))):

            action = self.choose_action(state, exp)
            n_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.store((state, action, reward, n_state, done))
            if done: state, info = self.env.reset()
            else: state = n_state

        self.all_states = self.buffer['obs'][:self.count].copy()

    def fill_buffer_multi(self, exp, n_p, prop):

        buffer_dict = Manager().dict()
        self.count = int(self.buffer_size * prop)
        self.count_current = 0

        while self.count_current < self.count:

            processes = [Process(target=self.fill_buffer_episode, args=(exp, p_ind, buffer_dict)) for p_ind in range(n_p)]

            for p in processes:    
                p.start()
            for p in processes:
                p.join()

            for mini_buffer in buffer_dict.values():
                new_count = self.count_current + len(mini_buffer['obs'])
                if new_count < self.count:
                    for key, value in mini_buffer.items():
                        self.buffer[key][self.count_current:new_count] = value
                    self.count_current = new_count
                else:
                    for key, value in mini_buffer.items():
                        self.buffer[key][self.count_current:self.count] = value[:self.count - self.count_current]
                    self.count_current = new_count
                    break

            if not int(1000 * self.count_current / self.count) % 10: print('{:0.02f}%'.format(100 * self.count_current / self.count))

        self.all_states = self.buffer['obs'][:self.count].copy()

    def fill_buffer_episode(self, exp, p_ind, buffer_dict):
        """
        Plays out one episode and saves to buffer.
        """

        np.random.seed()

        b_obs = []
        b_action = []
        b_reward = []
        b_n_obs = []
        b_term = []

        state, info = self.env.reset()
        while True:

            action = self.choose_action(state, exp)
            n_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            b_obs.append(state)
            b_action.append(action)
            b_reward.append(reward)
            b_n_obs.append(n_state)
            b_term.append(done)
            if done: break
            else: state = n_state

        buffer_dict[p_ind] = {key: value for key, value in zip(self.buffer, [b_obs, b_action, b_reward, b_n_obs, b_term])}

    def train(self, reward_tol, num_episodes=1e8):
        """
        Trains agent until average return exceeded.
        """

        for _ in range(int(num_episodes)):

            if np.mean(self.rets) >= reward_tol: break
            
            state, info = self.env.reset()
            ret = 0
            while True:
                
                action = self.choose_action(state)
                n_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                self.store((state, action, reward, n_state, done))
                self.update()

                ret += reward

                if done: break
                else: state = n_state

            self.epsilon *= self.decay
            self.epsilon = np.max([self.epsilon, self.epsilon_min])
                    
            self.rets[self.ret_count % self.scale] = ret
            self.ret_count += 1
            if not self.ret_count % self.scale: 
                # self.critic.save_weights('agent_models/critic')
                # self.target_critic.save_weights('agent_models/target_critic')
                # with open('epsilon.pkl', 'wb') as file: pickle.dump(self.epsilon, file)
                print('Episode number: {}, Epsilon: {:0.02f}, Average Return: {:0.02f}'.format(
                    self.ret_count, 
                    self.epsilon, 
                    np.mean(self.rets)))