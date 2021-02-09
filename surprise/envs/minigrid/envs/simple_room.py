#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym_minigrid.minigrid import *
from gym_minigrid.register import register

from gym import spaces

from operator import add

import matplotlib.pyplot as plt
import pdb

class SimpleEnemyEnv(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, max_steps=500, agent_pos=None, goal_pos=None, num_obstacles=1, seed=1327):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self.door_color = COLOR_NAMES[1]
        self.n_obstacles = num_obstacles
        self._size=10
        super().__init__(grid_size=self._size, max_steps=max_steps, seed=seed)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=np.zeros(49), 
                                                high=np.ones(49))

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # self.grid.vert_wall(width // 2, 0)
        # self.grid.set(width // 2, height // 2, Door(self.door_color))
        # for i in [1,2,3]:
        #     self.grid.set(width // 2, height // 2 - i, Door(self.door_color))
        #     self.grid.set(width // 2, height // 2 + i, Door(self.door_color))

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            #self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
            #self.agent_dir = 0
        else:
            self.place_agent(size=(width // 2, height))
            #self.agent_dir = 0

        '''
        if self._goal_default_pos is not None:
            goal = Goal()
            self.grid.set(*self._goal_default_pos, goal)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())
        '''

        self.mission = 'Reach the goal'

        # Place obstacles
        self.obstacles = []
        
        for i_obst in range(self.n_obstacles):
            self.obstacles.append(Ball())
            self.place_obj(self.obstacles[i_obst], 
                           size=(width // 2, height), 
                           max_tries=100)
        old_pos = self.obstacles[0].cur_pos
        self.obstacles[0].cur_pos = [self._size//2,self._size//2]
        self.grid.set(*old_pos, None)
        self._direction = 1

    def convert_obs(self, obs):
        # Get rid of color channel (1), flatten, binarize
        # We encode where there is an obstacle (inc. closed doors) (7x7)
        # And where the doors are (7x7)
        obs = obs['image']
        obstacles = np.zeros((7,7))
        doors = np.zeros((7,7))

        # 2 is wall, 4 is door, 6 is ball
        for idx in [6]:
            obstacles += (obs[:,:,0] == idx).astype(int)

        # Don't show unlocked doors:
        # Channel 2 is state, 0 -> unlocked, 1 -> locked
        """
        for i in range(obs.shape[0]):
            for j in range(obs.shape[1]):
                if obs[i,j,0] == 4 and obs[i,j,2] == 1:
                    obstacles[i,j] = 1
                if obs[i,j,0] == 4:
                    doors[i,j] = 1
        """
        [self.agent_pos]
        obs_ = np.hstack((obstacles)).flatten()
        # obs_ = np.concatenate((obs_, self.agent_pos))
        return obs_

    def step(self, action):
        action_map = [0,1,2,5]
        action = action_map[action]

        obs, reward, done, info = MiniGridEnv.step(self, action)

        obs = self.convert_obs(obs)

        # Update obstacle positions
        for i_obst in range(len(self.obstacles)):
            old_pos = self.obstacles[i_obst].cur_pos
            top = tuple(map(add, old_pos, (0, 0)))
            # print ("old_pos: ", old_pos, "direction: ", self._direction)
            try:
                # self.grid.set(*old_pos, None)
                if (old_pos[0] == (self._size-2)):
                    self._direction = -1
                    # print ("Chaning direction: ", self._direction)
                elif  (old_pos[0] == 1):
                    self._direction = 1
                    # print ("Chaning direction: ", self._direction)
                self.grid.set(*old_pos, None)
                old_pos = [old_pos[0]+self._direction,old_pos[1]]
                top = tuple(map(add, old_pos, (-1, -1)))
                # print ("new_pos: ",old_pos)
                self.place_obj(self.obstacles[i_obst], top=old_pos, size=(1,1), max_tries=10)
            except:
                pass

        return obs, reward, done, info

    def reset(self):
        #obs = super(MiniGridEnv, self).reset()
        obs = MiniGridEnv.reset(self)
        return self.convert_obs(obs)

    def render(self, mode=None):
        if mode == 'human':
            MiniGridEnv.render(self)
            return 
        else:
            import matplotlib
            matplotlib.use('Agg')
            obs = MiniGridEnv.render(self, mode='rgb_array')
        return obs

        # obs = MiniGridEnv.render(self, mode='rgb_array')
        # plt.imshow(obs)
        # plt.savefig('rollouts/minigrid/{0:04d}.png'.format(self.step_count))
        # plt.clf()

'''
register(
    id='Simple-Enemy-v0',
    entry_point='gym_minigrid.envs:SimpleEnemyEnv'
)
'''
