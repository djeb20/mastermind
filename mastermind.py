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

        self.state_dim = 12 * 6
        self.action_dim = len(self.colour_dict)

        self.test_goal = np.arange(4) + 1

    def reset(self):
        """
        Resets environment to empty grid with new goal.
        """

        self.grid = np.zeros((12, 6))
        self.grid_render = np.full((12, 6), ' ', dtype='O')
        self.count = 0

        while True:
            goal = np.random.randint(0, 8, 4) + 1
            if not (goal == self.test_goal).all():
                break

        self.goal = goal
        # self.goal = np.array([0, 1, 2, 3]) + 1
        self.goal_render = np.array([self.colour_dict[i - 1] for i in self.goal])

        return self.grid.flatten()
    
    def step(self, action):
        """
        Step in the environment, places piece in next available square.
        """

        # Negative reward for each step
        reward = -1
        done = False

        row = self.count // 4
        col = self.count % 4

        self.grid[row, col + 1] = action + 1
        self.grid_render[row, col + 1] = self.colour_dict[action]

        if (col - 3) == 0: # row is complete

            # Number exactly right
            right_ind = self.grid[row][1:5] == self.goal
            right = (right_ind).sum()

            # Number that are close, need more efficient solution
            d = dict(zip(*np.unique(self.grid[row][1:5][right_ind], return_counts=True)))
            sum = np.sum([d[colour] for colour in self.goal if colour in d])
            close = right - sum

            self.grid[row][0] = close
            self.grid[row][-1] = right

            # Just for rendering
            self.grid_render[row][0] = close
            self.grid_render[row][-1] = right

            if row == 11: # Finished game with no win
                done = True

            if right == 4: # Won game
                done = True
                reward += 30

        self.count += 1

        return self.grid.flatten(), reward, done, False

    def render(self):

        print(np.flipud(self.grid_render))

        






