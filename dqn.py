"""
Implementation of DQN with Hindersight Experience Replay (HER).
BETTER TO TAKE ACTION INTO CRITIC NETWORK OR HAVE N ACTION HEADS?
"""

# Required libraries
import tensorflow as tf
import numpy as np
import keras
from keras.layers import Dense
from keras.layers import Concatenate
from keras.layers import Input

class DQN:
    
    def __init__(self, state_dim, action_dim,
                 critic_arch=[64, 32], buffer_size=10000, batch_size=64,
                 gamma=0.99, epsilon=0.05, step_size=1e-4, tau=0.001, model_arch=0, pos_actions=np.array([])):
        
        # This dictates whether the Q network critic has state, action as input or just state.
        self.model_arch = model_arch
        self.pos_actions = pos_actions

        # Make the critic and target critic
        self.critic = self.make_critic(state_dim, action_dim, critic_arch)
        self.target_critic = self.make_critic(state_dim, action_dim, critic_arch)
        self.target_critic.set_weights(self.critic.get_weights())
        
        # Step the optimiser used and its stepsize.
        self.optimizer_critic = tf.keras.optimizers.Adam(step_size)
        
        # Create an empty replay buffer
        self.create_buffer(buffer_size, state_dim)
        
        # Save hyper-parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.step_size = step_size
        
        self.action_dim = action_dim
        
    def make_critic(self, state_dim, action_dim, critic_arch):
        """
        Creates the Q network for DQN
        """

        if self.model_arch == 0:
            # This version has a state going in and outputs values for every action.
        
            # Needs an input layer
            inp_state = Input(shape=(state_dim,))
            
            # Hidden layers
            out = Dense(critic_arch[0], activation='relu')(inp_state)
            for size in critic_arch[1:]:
                out = Dense(size, activation='relu')(out)

            # Output layer
            out = Dense(action_dim)(out)
            
            return keras.Model(inp_state, out)
    
    def create_buffer(self, buffer_size, state_dim):
        """
        Creates a new memory based on the number of transitions being recorded
        """
        
        # Count to keep track of size of memory
        self.count = 0
        
        # Initiate empty buffer
        # Actions are always single number in a discrete setting?
        self.buffer = {'obs': np.empty((buffer_size, state_dim)),
                       'action': np.empty((buffer_size, )),
                       'reward': np.empty((buffer_size, )),
                       'new_obs': np.empty((buffer_size, state_dim)),
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
                 self.buffer['term'][index]]
        
        # Make tensors for speed.
        t_trajectory = [tf.convert_to_tensor(item) for item in batch]
        t_trajectory[1] = tf.cast(t_trajectory[1], dtype=tf.int32)
        t_trajectory[2] = tf.cast(t_trajectory[2], dtype=tf.float32)
        t_trajectory[4] = tf.cast(t_trajectory[4], dtype=tf.float32)

        return (t_trajectory[i] for i in range(len(batch)))

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
        
        if (np.random.rand() < self.epsilon) & (exp == True):
            
            action = np.random.randint(self.action_dim)
            
        else:
            
            action = self.critic(np.array([state]))[0].numpy().argmax()
        
        return action  