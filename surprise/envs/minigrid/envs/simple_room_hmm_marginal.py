#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym_minigrid.minigrid import *
from gym_minigrid.register import register

from gym import spaces

from operator import add

import matplotlib.pyplot as plt
import pdb
from surprise.envs.minigrid.envs.simple_room_hmm import SimpleEnemyEnvHMM
from surprise.buffers.buffers import BernoulliBuffer

class SimpleEnemyEnvHMMMarginal(SimpleEnemyEnvHMM):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, max_steps=500, agent_pos=None, goal_pos=None, num_obstacles=1, seed=1327):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self.door_color = COLOR_NAMES[1]
        self.n_obstacles = num_obstacles
        self._size = 6
        self._viz_size = 5
        self._buffer = BernoulliBuffer(obs_dim=self._size * self._size)
        super().__init__(grid_size=self._size, max_steps=max_steps, seed=seed, agent_view_size=self._viz_size)
        self.action_space = gym.spaces.Discrete(4)
        
        # self.observation_space = gym.spaces.Box(low=np.zeros((self._size-2)*(self._size-2)*2+(len(DIR_TO_VEC))), 
        #                                         high=np.ones((self._size-2)*(self._size-2)*2+(len(DIR_TO_VEC))))
        self.observation_space = gym.spaces.Box(low=np.zeros(((self._size)*(2))+(self._viz_size*self._viz_size)), 
                                                high=np.ones(((self._size)*(2))+(self._viz_size*self._viz_size)))
        
        ### State = x loc for agent y loc for agent, x loc for enemey and y loc for enemy.
        self._hmm = np.zeros((self._size, self._size, self._size, self._size))

    def valid_pos(self, pos):
        
        return pos[0] >= 0 and pos[0] <= self._size-1 and pos[1] >= 0 and pos[1] <= self._size-1
    
    

    def step(self, action):
        obs, reward_, _, info = super().step(action)
        
        ### sample true state belief
        state_sample = np.random.choice(range(self._size*self._size), p=self._forward[-1])
        state_ = np.zeros((self._size*self._size))
        state_[state_sample] = 1
        reward_ = self._buffer.logprob(state_)
        self._buffer.add(self.getTrueState().flatten())
        # print ("reward: ", reward_)
        return obs, reward_, False, info
        # return obs, reward, done, info
    
    
    def reset(self):
        #obs = super(MiniGridEnv, self).reset()
        obs = super().reset()
        
        self._buffer.reset()
        self._buffer.add(self.getTrueState().flatten())
        return obs
