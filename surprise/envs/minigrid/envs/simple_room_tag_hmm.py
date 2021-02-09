#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym_minigrid.minigrid import *
from gym_minigrid.register import register

from gym import spaces

from operator import add

import matplotlib.pyplot as plt
import pdb
from surprise.envs.minigrid.envs.simple_room_hmm import SimpleEnemyEnvHMM

class SimpleEnemyTagEnvHMM(SimpleEnemyEnvHMM):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, max_steps=500, agent_pos=None, grid_size=8, goal_pos=None, num_obstacles=1, seed=1327, agent_view_size=5):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self.door_color = COLOR_NAMES[1]
        self.n_obstacles = num_obstacles
        self._size=grid_size
        self._viz_size = agent_view_size
        super().__init__(grid_size=self._size, max_steps=max_steps, seed=seed, agent_view_size=self._viz_size)
        self.action_space = gym.spaces.Discrete(5)
        
        
    def get_view_exts(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """

        topX = self.agent_pos[0] + (self._viz_size//2)
        botX = self.agent_pos[0] - (self._viz_size//2)
        topY = self.agent_pos[1] + (self._viz_size//2)
        botY = self.agent_pos[1] - (self._viz_size//2)
       

        return (topX, topY, botX, botY)
    
    def _gen_grid(self, width, height):
        # Create the grid
        
        self._obs_state = []
        
        for i_obst in range(self.n_obstacles):
            self._obs_state.append({"frozen": False})
            
        super()._gen_grid(width, height)

    def step(self, action):
        self.agent_dir = 0
        info = {}
        # obs, reward, done, info = MiniGridEnv.step(self, action)
        old_pos = self.obstacles[0].cur_pos
        prev_state = (old_pos[0]*self._size) + old_pos[1]
        self.move(action)
        # Update obstacle positions
        for i_obst in range(len(self.obstacles)):
            ### Add a bit more randomness to the movements.
            if (np.random.rand() < 0.1):
                continue
            elif (self._obs_state[i_obst]["frozen"] == True):
                # print("particle frozen")
                continue
            top = tuple(map(add, old_pos, (-1, -1)))
            # print ("old_pos: ", old_pos, "direction: ", self._direction)
            try:
                self.place_obj(self.obstacles[i_obst], top=top, size=(3,3), max_tries=100)
                self.grid.set(*old_pos, None)
            except:
                pass
        
        old_pos = self.obstacles[0].cur_pos
        new_state = (old_pos[0]*self._size) + old_pos[1]
        self._forward.append(self._forward[-1])
        check_ = np.sum(self._forward[-1])
        b = self.belief(obs_pos=old_pos, agent_pos=self.agent_pos)
        b_ = (b).flatten()
        
        if (self.seesEnemey()):
            ### Beleif if perfect in this case
            self._forward[-1] = b_
        else:
            for i in range(self._size* self._size):
                for j in range(self._size * self._size):
                    ind = (i*self._size) + j
                    ### transition prob for old states (ind) to new state
                    trans_prob = self._transitionMatrix[j,i]
                    ### Compute belief for next state
                    # check_ = np.sum(b_)
                    # new_state_prob = np.array([ self._forward[-1][k] * trans_prob[k] for k in range(self._size*self._size)]) * b_[ind]
                    # new_state_prob = np.array([ self._forward[-1][k] * trans_prob[k] for k in range(self._size*self._size)])
                    new_state_prob =  self._forward[-1][j] * trans_prob
                    check_ = np.sum(new_state_prob)
                    self._forward[-1][i] = self._forward[-1][i] + new_state_prob
            ### Should this be 1?
        check_ = np.sum(self._forward[-1])
        self._forward[-1] = self._forward[-1] / check_ ### Normalize for error
        # reward_ = self.computeReward()
        assert np.isclose(np.sum(self._forward[-1]),1.0, rtol=1e-01), "np.sum(self._forward[-1])" + str(np.sum(self._forward[-1]))
        # reward_ = - self.entropyState(self._forward[-1])
        reward_ = self.entropy(self.getTrueState().flatten(), self._forward[-1])
        # print ("reward: ", reward_)
        return self.get_obs(), reward_, False, info
        # return obs, reward, done, info
    
    def move(self, action):
        # print ("action: ", action)
        if action < 4:
            super().move(action)
        else:
            ### Freeze particle if next to it.
            if (self.seesEnemey(dist=3)):
                self._obs_state[0]["frozen"] = True
                
            
            
    def reset(self):
        #obs = super(MiniGridEnv, self).reset()
        obs = super().reset()
        self._obs_state = []
        
        for i_obst in range(self.n_obstacles):
            self._obs_state.append({"frozen": False})
        
        return obs
    
   

'''
register(
    id='Simple-Enemy-v0',
    entry_point='gym_minigrid.envs:SimpleEnemyEnv'
)
'''
