import numpy as np
from collections import defaultdict

class mastermind:
    """
    Environment for the game mastermind.
    """

    def __init__(self, action_type='peg', goal_type='fixed', reward_struc='basic'):

        # Colour from game
        self.colour_dict = {0: ' ',
                            1: 'R',
                            2: 'P',
                            3: 'B',
                            4: 'O',
                            5: 'W',
                            6: 'K',
                            7: 'G',
                            8: 'Y'}

        self.action_type = action_type
        self.goal_type = goal_type
        self.reward_struc = reward_struc

        self.width = 4
        self.height = 12

        self.state_dim = self.height * (self.width + 2) # Number of state features

        if self.action_type == 'peg': self.action_dim = len(self.colour_dict) - 1
        elif self.action_type == 'guess': 

            def convert(i):
                guess = []
                for _ in range(4):
                    guess.append(i % 8)
                    i = i // 8
                return np.array(guess)

            self.action_dim = 8 ** 4
            self.guess_dict = {i : convert(i) for i in range(self.action_dim)}

        self.fixed_goal = np.random.randint(0, 8, 4) + 1
        self.test_goal = np.random.randint(0, 8, 4) + 1
        # self.test_goals = [np.random.randint(0, 8, 4) + 1, np.random.randint(0, 8, 4) + 1]

        # Trying to speed up learning
        self.ep_count = 0
        
        # To make super fast, transitions dictionary.
        self.trans = TransDict()
        self.trans.height = self.height
        self.trans.width = self.width
        self.trans.reward_struc = reward_struc

    def reset(self):
        """
        Resets environment to empty grid with new goal.
        """

        self.grid = np.zeros((self.height, self.width + 2), dtype=int)

        if self.goal_type == 'fixed': goal = self.test_goal
        elif self.goal_type == 'changes':

            while True: # Sillyness to not reset to test goal
                goal = np.random.randint(0, 8, 4) + 1
                if not (goal == self.test_goal).all(): break

        self.goal = goal
        self.goal_render = np.array([self.colour_dict[i] for i in self.goal])

        self.count = 0
        self.ep_count += 1

        return self.grid.flatten()
    
    def step(self, action):
        """
        Step in the environment, places piece in next available square.
        """

        if self.action_type == 'peg':

            self.grid, reward, done = self.trans[tuple([self.grid.tobytes(), action, self.goal.tobytes(), self.count])]
            self.count += 1

        elif self.action_type == 'guess':
            for a in self.guess_dict[action]:

                self.grid, reward, done = self.trans[tuple([self.grid.tobytes(), a, self.goal.tobytes(), self.count])]
                self.count += 1

        return self.grid.flatten(), reward, done, False

    def take_step(self, grid, action, goal, count):
        """
        Calculates reward etc for step in environment.
        """

        # Negative reward for each step
        reward = -1
        done = False

        row_ind = count // 4
        col_ind = count % 4

        grid[row_ind, col_ind + 1] = action + 1

        if col_ind == 3: # row is complete

            row = grid[row_ind][1:5] # Row we care about

            # Number not exactly right, then number exactly right
            not_right_ind = row != goal
            right = self.width - np.count_nonzero(not_right_ind)

            n_row = row[not_right_ind]
            n_goal = goal[not_right_ind]

            # Number that are close, need more efficient solution
            close = 0
            for a in n_row:
                if a in n_goal:
                    close += 1
                    n_goal[(n_goal == a).argmax()] = 0

            grid[row_ind, 0] = close
            grid[row_ind, -1] = right

            if row_ind == 11: # Finished game with no win
                done = True

            if right == 4: # Won game
                done = True
                reward += 30

            if self.reward_struc == 'clues': reward += close + 2 * right

        return grid, reward, done

    def render(self):
        """
        Render environment, expensive!
        Could map be used?
        """

        self.grid_render = np.zeros((self.height, self.width + 2), dtype='<U1')

        for i in range(self.height):
            self.grid_render[i, 0] = self.grid[i, 0]
            self.grid_render[i, -1] = self.grid[i, -1]
            for j in range(1, self.width + 1):

                self.grid_render[i, j] = self.colour_dict[self.grid[i, j]]

        print(np.flipud(self.grid_render))

class TransDict(dict, mastermind):
    
    def __missing__(self, key):

        grid, action, goal, count = key
        val = self.take_step(np.frombuffer(grid, dtype=np.int_).reshape(self.height, self.width + 2).copy(),
                             action, 
                             np.frombuffer(goal, dtype=np.int_), 
                             count)
        
        self.__setitem__(key, val)
        
        return val






