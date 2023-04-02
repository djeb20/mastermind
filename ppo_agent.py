# Required libraries
import tensorflow as tf
import numpy as np
import keras
from keras.layers import Dense
from keras.layers import Input
from multiprocessing import Process, Manager
from tqdm import tqdm

class Agent:
    """ 
    Class to generate and train an agent using PPO
    """
    
    def __init__(self, env, state_dim, num_actions,
                 actor_arch=[128, 64], critic_arch=[128, 64],
                 actor_rate=1e-4, critic_rate=1e-4,
                 gamma = 0.99, lam=0.95, 
                 epsilon_clip=0.2, entropy_coef=0.01):
        
        self.env = env
        
        # The dimensions of the state and action vectors
        self.state_dim = state_dim
        
        # The possible discrete actions and amount of them
        self.num_actions = num_actions
        self.actions = np.arange(num_actions)
        
        # Create our actor and critic
        self.actor = self.create_actor(actor_arch)
        self.critic = self.create_critic(critic_arch)

        # Pre saved good weights
        # self.critic.load_weights('agent_models/critic')
        # self.actor.load_weights('agent_models/actor')
        
        # We need to define our optimisers, I am using Adam as everyone seems to
        self.optimizer_critic = keras.optimizers.Adam(critic_rate)
        self.optimizer_actor = keras.optimizers.Adam(actor_rate)
        
        # My hyper-parameter for calculating advantages and returns
        self.gamma = gamma
        self.lam = lam
        
        # epsilon value for the clipping in PPO
        self.epsilon_clip = epsilon_clip
        
        # Entropy coefficient
        self.entropy_coef = entropy_coef
        
        # Used to avoid numerical instability
        self.jitter = 1e-16
        
    def create_actor(self, actor_arch):
        """
        Function to create actor network architechture.
        The actor is the policy function.
        """
    
        # Needs an input layer
        inp = Input(shape=(self.state_dim,))

        # Hidden layers
        out = Dense(actor_arch[0], activation='relu')(inp)
        for size in actor_arch[1:]:
            out = Dense(size, activation='relu')(out)

        # Output layer
        out = Dense(self.num_actions, activation='softmax')(out)
        
        return keras.Model(inp, out)
    
    def create_critic(self, critic_arch):
        """
        Function to create critic network architechture.
        The critic is our value function.
        """

        # State as input
        inp = Input(shape=(self.state_dim,))
        
        # Hidden layers
        out = Dense(critic_arch[0], activation='relu')(inp)
        for size in critic_arch[1:]:
            out = Dense(size, activation='relu')(out)

        # Output layer
        out = Dense(1)(out) 

        return keras.Model(inp, out)
            
    def create_buffer(self, buffer_size):
        """
        Creates a new memory based on the number of transitions being recorded
        """
        
        # Count to keep track of size of memory
        self.count = 0
        
        # Initiate empty buffer
        self.buffer = {'state': np.empty((buffer_size, self.state_dim)),
                       'action': np.empty((buffer_size, )),
                       'reward': np.empty((buffer_size, )),
                       'term': np.empty((buffer_size, )),
                       'value': np.empty((buffer_size, )), 
                       'prob': np.empty((buffer_size, ))}
    
    def store(self, tup):
        """ 
        Stores an event tuple in the buffer.
        """
        
        self.buffer['state'][self.count] = tup[0]
        self.buffer['action'][self.count] = tup[1]
        self.buffer['reward'][self.count] = tup[2]
        self.buffer['term'][self.count] = tup[3]
        self.buffer['value'][self.count] = tup[4]
        self.buffer['prob'][self.count] = tup[5]

        self.count += 1
        
    def get_batch(self, batch_number, batch_size, agent):
        """
        Selects a batch from the memory.
        Using a selected trajectory.
        """
            
        # Index is simply a set of sequential interactions in the buffer
        start_index = batch_number * batch_size
        end_index = start_index + batch_size - 1 # PROBABLY A BETTER WAY TO DO THIS, NEED ONE EXTRA FOR VALUE.

        # We have a different index for values as we need the next one too.
        trajectory = [self.buffer[agent]['state'][start_index:end_index], 
                      self.buffer[agent]['action'][start_index:end_index], 
                      self.buffer[agent]['reward'][start_index:end_index], 
                      self.buffer[agent]['term'][start_index:end_index],
                      self.buffer[agent]['value'][start_index:end_index+1],
                      self.buffer[agent]['prob'][start_index:end_index]]
        
        # Make tensors for speed.
        t_trajectory = [tf.convert_to_tensor(item) for item in trajectory]
        t_trajectory[1] = tf.cast(t_trajectory[1], dtype=tf.int32)
        t_trajectory[2] = tf.cast(t_trajectory[2], dtype=tf.float32)
        t_trajectory[3] = tf.cast(t_trajectory[3], dtype=tf.float32)
        t_trajectory[4] = tf.cast(t_trajectory[4], dtype=tf.float32)
        t_trajectory[5] = tf.cast(t_trajectory[5], dtype=tf.float32)

        return t_trajectory
    
    # @tf.function(experimental_relax_shapes=True)
    @tf.function
    def calc_advantages(self, trajectory):
        """
        This function will calculate the advantage estimates,
        using the GAE algorithm. Returns returns also.
        """

        # Trajectory size
        n = len(trajectory[2])
        
        returns = tf.TensorArray(tf.float32, size=n)
        
        # The advantages are calculated incrementally
        adv_estimate = 0.
                
        # We are cycling through the rewards backwards
        for i in tf.range(n)[::-1]:
            
            # First we calculate our td-error
            # delta = reward + gamma * next value * done -  value
            td_error = trajectory[2][i] + self.gamma * trajectory[4][i + 1] * trajectory[3][i] - trajectory[4][i]
            
            # Calculate our advantage estimates
            # adv = delta + gamma * lambda * done * prev adv
            adv_estimate = td_error + self.gamma * self.lam * trajectory[3][i] * adv_estimate
            
            j = (n - 1) - i
            returns = returns.write(j, [adv_estimate + trajectory[4][i]])
            
        # Flip the advantages back round and generate advantages.
        returns = returns.concat()
        returns = returns[::-1]
        advantages = returns - trajectory[4][:-1]
        
        # Normalised for stability.
        return returns, (advantages - tf.math.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-16)
                
    def learn(self, num_p, num_epochs, batch_size, steps_per_agent):
        """
        Updates the actor and critic for a specified number of steps, new style update.
        """

        # SHOULD ADVANTAGES ETC ALL BE CALCULATED THEN ADDED TO BUFFER AND EVERYTHING RANDOMLY SAMPLED? PROBABLY
        # WITHOUT REPLACEMENT.
        
        trajectories = [[] for _ in range(num_p)]
        advantages = [[] for _ in range(num_p)]
        returns = [[] for _ in range(num_p)]
        
        for agent in range(num_p):

            for batch_number in range(steps_per_agent // batch_size):

                # Take batch of transitions
                trajectories[agent].append(self.get_batch(batch_number, batch_size, agent))

                # Calc their advantages and returns
                rets, advs = self.calc_advantages(trajectories[agent][batch_number])

                # Save for learning
                returns[agent].append(rets)
                advantages[agent].append(advs)

        for agent in range(num_p):
            for _ in range(num_epochs):
                for batch_number in range(steps_per_agent // batch_size):
                    # The networks are update for some amount of steps
                    
                    # Update the actor
                    self.update_actor(trajectories[agent][batch_number], advantages[agent][batch_number])

                    # Update the critic
                    self.update_critic(trajectories[agent][batch_number], returns[agent][batch_number])
    
    @tf.function
    def update_actor(self, trajectory, advantages):
        """
        Function to update the actor network
        """
        
        # Everything in tape will have derivative taken
        with tf.GradientTape() as g:
            
            # Calculate the ratios, need probabilities
            action_dist = self.actor(trajectory[0], training=True) + self.jitter
            probs = tf.gather_nd(action_dist, tf.stack((tf.constant(np.arange(len(action_dist)), dtype=tf.int32), trajectory[1]), -1))
            
            # Calculate the ratios, in log space for numerical stability
            ratio = tf.math.exp(tf.math.log(probs) - tf.math.log(trajectory[5]))
            
            # Calculate the loss and entropy for each transition
            losses = tf.math.minimum(ratio * advantages, 
                       tf.clip_by_value(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages)
            entropies = tf.reduce_sum(-action_dist * tf.experimental.numpy.log2(action_dist), axis=1)
            
            loss = -tf.math.reduce_mean(losses) - self.entropy_coef * tf.math.reduce_mean(entropies)
                        
        # Calculates gradient and then optimises using it (moving in that direction)
        gradient = g.gradient(loss, self.actor.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(gradient, self.actor.trainable_variables))
        
    @tf.function
    def update_critic(self, trajectory, returns):
        """
        Function to update the critic network
        """
        
        # Everything in tape will have derivative taken
        with tf.GradientTape() as g:
            
            # Calculate loss
            values = tf.reshape(self.critic(trajectory[0], training=True), [-1])
            loss = tf.math.reduce_mean(tf.math.square(returns - values))

        # Calculates gradient and then optimises using it (moving in that direction)
        gradient = g.gradient(loss, self.critic.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(gradient, self.critic.trainable_variables))
        
    def choose_action(self, state):
        """
        Given a state returns a chosen action given by policy.
        """
        
        # Select action using actor
        policy = self.actor(state.reshape(1, -1))[0].numpy() + self.jitter
        action = np.random.choice(self.actions, p=policy)
        
        return action, policy[action]           
        
    def collect_experience(self, steps_per_agent, num_p, return_cache):
        """
        Acts out a series of time steps in the environment
        """
    
        # Create a new empty memory of correct size
        self.create_buffer(steps_per_agent)
        
        processes = [Process(target=self.play_ind, args=(steps_per_agent, agent, return_cache)) for agent in range(num_p)]
        
        for p in processes:
            p.start()

        for p in processes:
            p.join()
            
        # Collect together the agents' memories
        self.buffer = [buffer for key, buffer in return_cache.items() if key not in ['state', 'env', 'ret', 'rets', 'count']]
        
    def play_ind(self, steps_per_agent, agent, return_cache):

        # So each agent does not act out identical experience.
        np.random.seed()

        state = return_cache['state']
        self.env = return_cache['env']
        rets = return_cache['rets']
        ret = return_cache['ret']
        count = return_cache['count']

        for _ in range(steps_per_agent):
            
            # Select action and prob of choosing that action
            action, prob = self.choose_action(state)

            # Take action in environment
            n_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if not agent: ret += reward

            # Store relevant information in buffer, Need the value of the state for advantage calc
            self.store((state, action, reward, 1 - done, self.critic(state.reshape(1, -1))[0].numpy(), prob))

            if done: 
                state, info = self.env.reset()
                if not agent: 
                    rets[count % rets.shape[0]] = ret
                    count += 1
                    ret = 0
            else: state = n_state

        # Returns everything needed to start from same place next time and combine experience.
        if not agent: 
            return_cache['state'] = state
            return_cache['env'] = self.env
            return_cache['rets'] = rets
            return_cache['ret'] = ret
            return_cache['count'] = count

        return_cache[agent] = self.buffer

    def train(self, num_ite, num_epochs, num_p, steps_per_agent, batch_size):
        """
        Trains the PPO agents for a certain number of epochs.
        """

        steps_per_agent = int(np.ceil(steps_per_agent / batch_size)) * batch_size

        state, info = self.env.reset()
        return_cache = Manager().dict()
        return_cache['state'] = state
        return_cache['env'] = self.env
        return_cache['rets'] = np.zeros(1000, dtype=float)
        return_cache['ret'] = 0
        return_cache['count'] = 0

        for ite in tqdm(range(num_ite)):

            self.collect_experience(steps_per_agent, num_p, return_cache) # return_cache = ?
            print('Episode Number (*{}): {}, Average Return: {}'.format(num_p, return_cache['count'], np.mean(return_cache['rets'][:return_cache['count']])))
            self.learn(num_p, num_epochs, batch_size, steps_per_agent)
            
            # if ite % 10 == 0:
            #     self.critic.save_weights('agent_models/critic')
            #     self.actor.save_weights('agent_models/actor')

        self.env = return_cache['env']