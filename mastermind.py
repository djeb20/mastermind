import numpy as np
from collections import defaultdict

class mastermind:
    """
    Environment for the game mastermind.
    """

    def __init__(self):

        # Colour from game
        self.colour_dict = {0: 'R',
                            1: 'P',
                            2: 'B',
                            3: 'O',
                            4: 'W',
                            5: 'K',
                            6: 'G',
                            7: 'Y'}

        self.width = 4
        self.height = 12

        self.state_dim = self.height * (self.width + 2)
        self.action_dim = len(self.colour_dict)

        self.test_goal = np.arange(1, 5)

        # Trying to speed up learning
        self.ep_count = 0
        
        # To make super fast, transitions dictionary.
        self.trans = TransDict()
        self.trans.height = self.height
        self.trans.width = self.width

    def reset(self):
        """
        Resets environment to empty grid with new goal.
        """

        self.grid = np.zeros((self.height, self.width + 2))

        self.goal = self.test_goal

        # if self.ep_count % 100 == 0:

        #     while True:
        #         goal = np.random.randint(0, 8, 4) + 1
        #         if not (goal == self.test_goal).all():
        #             break

        #     self.goal = goal
        #     self.goal_render = np.array([self.colour_dict[i - 1] for i in self.goal])

        self.count = 0
        self.ep_count += 1

        return self.grid.flatten()
    
    def step(self, action):
        """
        Step in the environment, places piece in next available square.
        """

        self.grid, reward, done = self.trans[tuple([self.grid.tobytes(), action, self.goal.tobytes(), self.count])]

        # if tuple([self.grid.tobytes(), action, self.goal.tobytes(), self.count]) not in self.trans:

        #     self.grid, reward, done = self.take_step(action, self.grid, self.goal, self.count)
        #     self.trans[tuple([self.grid.tobytes(), action, self.goal.tobytes(), self.count])] = (self.grid, reward, done)

        # else:

        #     self.grid, reward, done = self.trans[tuple([self.grid.tobytes(), action, self.goal.tobytes(), self.count])]

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

            # Number exactly right
            right_ind = row == goal
            right = (right_ind).sum()

            # Number that are close, need more efficient solution
            d = dict(zip(*np.unique(row[not right_ind], return_counts=True)))
            sum = np.sum([d[colour] for colour in goal if colour in d])
            close = right - sum

            grid[row][0] = close
            grid[row][-1] = right

            if row == 11: # Finished game with no win
                done = True

            if right == 4: # Won game
                done = True
                reward += 30

        return grid, reward, done

    def render(self):
        """
        Render environment, expensive!
        Could map be used?
        """

        self.grid_render = np.full((12, 6), ' ')


        for i, in range(self.height):
            for j in range(1, self.width - 1):

                self.grid_render[i, j] = self.colour_dict[self.grid[i, j]]

        print(np.flipud(self.grid_render))

class TransDict(dict, mastermind):
    
    def __missing__(self, key):

        grid, action, goal, count = key
        val = self.take_step(np.frombuffer(grid, dtype="int64").reshape(self.height, self.width + 2).copy(),
                             action, 
                             np.frombuffer(goal, dtype="int64"), 
                             count)
        # val = self.take_step(action, 
        #                      np.frombuffer(grid, dtype=32").reshape(self.height, self.width + 2), 
        #                      np.frombuffer(goal, dtype="int32"), 
        #                      count)
        
        self.__setitem__(key, val)
        
        return val






