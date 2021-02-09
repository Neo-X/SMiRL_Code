#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym_minigrid.minigrid import *
from gym_minigrid.register import register

from gym import spaces

from operator import add

import matplotlib.pyplot as plt
import pdb
from surprise.envs.minigrid.envs.simple_room_tag_hmm import SimpleEnemyTagEnvHMM
from surprise.buffers.buffers import BernoulliBuffer

class SimpleEnemyEnvTagHMMMarginal(SimpleEnemyTagEnvHMM):
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
        
        self.observation_space = gym.spaces.Box(
                np.concatenate(
                    (self.observation_space.low.flatten(), 
                     np.zeros(self._buffer.get_params().shape), 
                     np.zeros(1))
                ),
                np.concatenate(
                    (self.observation_space.high.flatten(), 
                     np.ones(self._buffer.get_params().shape), 
                     np.ones(1)*max_steps)
                )
            )
    

    def step(self, action):
        obs, reward_, _, info = super().step(action)
        
        ### sample true state belief
        state_sample = np.random.choice(range(self._size*self._size), p=self._forward[-1])
        state_ = np.zeros((self._size*self._size))
        state_[state_sample] = 1
        # reward_ = self._buffer.logprob(self._forward[-1])
        reward_ = self._buffer.logprob(state_)
        self._buffer.add(self._forward[-1])
        # print ("reward: ", reward_)
        return self.get_obs2(obs), reward_, False, info
        # return obs, reward, done, info
    
    
    def reset(self):
        #obs = super(MiniGridEnv, self).reset()
        obs = super().reset()
        
        self._buffer.reset()
        self._buffer.add(self.getTrueState().flatten())
        return self.get_obs2(obs)
    
    def get_obs2(self, obs):
        '''
        Augment observation, perhaps with generative model params
        '''
        theta = self._buffer.get_params()
        obs = np.concatenate([obs, theta, [self._buffer.buffer_size]])
        
        return obs
