import os
import gym
from gym import spaces

import numpy as np
import matplotlib.pyplot as plt
# from scipy.misc import imsave
import imageio

import pdb

class TetrisEnv(gym.Env):
    def __init__(self, width=4, height=10, episode_length=100, render=False,
                 reward_func=None, **kwargs):
        # Grid parameters
        self.width = width
        self.height = height
        self.reward_func=reward_func
        
        self._render=render

        # Grid
        self.grid = np.zeros((self.height, self.width))

        # Next block int (also part of state)
        self.nextBlock = None

        # Timer
        self.time = 0
        self.episode_length = episode_length

        # Done flag
        self.done = False

        # For saving images
        self.img_ctr = 0

        # gym spaces
        self.action_space = spaces.Discrete(12)
        self.observation_space = spaces.Box(low=0, 
                                            high=1, 
                                            shape=(self.width*self.height+1,),
                                            dtype=np.float32)
        self.reset()

    def chooseNextBlock(self):
        '''
        Return random block index (only 2 for 3-tetris)
        '''
        self.nextBlock = np.random.randint(2)

    def _rotateSquare(self, square):
        '''
        Rotate the square (L) piece

        block looks like:
        x0
        xx

        '''
        out = square.copy()
        out[0, 0] = square[0, 1]
        out[0, 1] = square[1, 1]
        out[1, 1] = square[1, 0]
        out[1, 0] = square[0, 0]
        return out

    def getBlock(self, block_id, rotation, column):
        block_map = {0: np.array([[1, 1, 1]]),
                     1: np.array([[1, 1], [0, 1]])}

        # Get block
        block = block_map[block_id]
        
        # Rotate block
        if block_id == 0:
            if rotation % 2 == 1:
                block = block.T
        else:
            for _ in range(rotation):
                block = self._rotateSquare(block)

        # Fix column
        column = max(0, column)
        column = min(self.grid.shape[1] - block.shape[1], column)

        # Return grid sized array with zeros except for
        # block in the correct column
        blockGrid = np.zeros_like(self.grid)
        blockGrid[0:block.shape[0], column:column+block.shape[1]] = block

        return blockGrid

    def simulateDrop(self, block, record=False):
        '''
        Simulate a drop given a `block` from `getBlock`
        and `self.grid`. `record` will save images to 
        the `record` path.
        '''
        # Block is the same dimension as the state
        # but only contains the block at the very top
        # In a loop we roll the block down and detect collisions
        prev = None
        collision = False
        while not collision:
            droppingState = self.grid + block
            # If there is a collision or block is at bottom
            if np.sum(droppingState == 2) != 0:
                collision = True
            else:
                # Set prev to curr
                prev = droppingState

                # Save img if recording
                if record:
                    plt.imshow(prev)
                    plt.savefig(os.path.join(record, '{}.png'.format(str(self.img_ctr).zfill(5))))
                    self.img_ctr += 1
                    plt.clf()

                # If we hit the ground, then collision
                if np.sum(block[-1,:]) > 0:
                    collision = True

                block = np.roll(block, 1, axis=0)

        if prev is None:
            # In this case we lose b/c we hit the top
            self.done = True
        else:
            self.grid = prev

    def destroyRows(self):
        # Bottom up, find lines, replace with zeros, 
        # roll one down from line up to top (bottom zeros move to top)
        rows_cleared = 0
        for i in list(range(self.grid.shape[0]))[::-1]:
            # while row i is line:
            while np.sum(self.grid[i, :]) == self.grid.shape[1]:
                rows_cleared += 1
                self.grid[i, :] = 0
                self.grid[0:i+1, :] = np.roll(self.grid[0:i+1, :], 1, axis=0)
        return rows_cleared 

    def render(self, save=False, mode=None):
        from skimage.transform import rescale, resize, downscale_local_mean
        if save:
            # Use `save` as path
            plt.imsave(save, self.grid, cmap='Greys')
#         else:
#             print('Next block: {}'.format(self.nextBlock))
#             print(self.grid)
            
        img = self.grid
        scaling=8
        shape_ = img.shape 
        img = resize(img, (shape_[0] * scaling, shape_[1]*scaling) ,
                           anti_aliasing=False, order=0)
        img =  np.array(np.moveaxis(np.repeat(np.reshape(img, (1, shape_[0] * scaling, shape_[1]*scaling)), 3, axis=0), 0, -1) * 255, dtype='uint8')
#         print("Tetris img shape: ", img.shape)
        return img

    def step(self, action, record=False):
        assert not self.done, "Can't take action after done"

        # Translate Discrete(12) in to rotation column
        if self.nextBlock == 0:
            action = action % 6
            actions = [(0,0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3)]
            rotation, column = actions[action]
        else:
            rotation, column = action % 4, action // 4

        # Simulate
        block = self.getBlock(self.nextBlock, rotation, column)
        self.simulateDrop(block, record=record)
        rows_cleared = self.destroyRows()

        # Update state
        self.chooseNextBlock()
        self.time += 1

        if (self.reward_func == "rows_cleared"):
            r = rows_cleared
#             print ("rows_cleared: ", rows_cleared)
        else:
            r = self.get_reward(self.grid)

        infos = {'rows_cleared': rows_cleared,
                 "tetris_reward": r}
        if self._render:
            infos["rendering"] = self.render(save=False, mode='rgb_array')
#             print (infos["rendering"].shape)
        return self.get_obs(), r, self.done, infos

    def get_reward(self, obs):
        if self.done:
            return -100
        else:
            return 0

    def get_obs(self):
        grid = self.grid.flatten().copy()
        return np.hstack((grid, self.nextBlock))

    def get_actionSpace(self):
        actionSpaceDict = {0: [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3)],
                           1: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]}
        return actionSpaceDict[self.nextBlock]

    def sampleAction(self):
        # Sample a random valid action
        actionSpace = self.get_actionSpace()
        return actionSpace[np.random.randint(len(actionSpace))]

    def reset(self):
        self.chooseNextBlock()
        self.grid = np.zeros((self.height, self.width))
        self.done = False
        self.time = 0
        self.img_ctr = 0
        return self.get_obs()

'''
env = TetrisEnv()
env.reset()

for i in range(100):
    obs, rew, done, info = env.step(np.random.randint(12))
    print(info)
    env.render()
'''
