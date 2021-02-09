import numpy as np
import gym
from gym.spaces import Box
import pdb

class VisitationCountWrapper(gym.Env):
    def __init__(self, env, bonus_coefficient=1):
        '''
        params
        ======
        env (gym.Env) : environment to wrap

        buffer (Buffer object) : Buffer that tracks history and fits models
        '''
        self.env = env

        # TODO: Create support for discrete and box
        self.obs_dim = env.observation_space.low.shape[0]
        self.visitation_counts = np.zeros(self.obs_dim)
        self.bonus_coefficient = bonus_coefficient

        # Spaces
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def add_visit(self, obs):
        self.visitation_counts[np.argmax(obs)] += 1

    def get_rews(self, obs):
        idx = np.argmax(obs)
        return 1/np.sqrt(self.visitation_counts[idx]) * self.bonus_coefficient

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.add_visit(obs)
        rew = self.get_rews(obs)
        return obs, rew, done, info

    def reset(self):
        return self.env.reset()

    def encode_obs(self, obs):
        '''
        Used to encode the observation before putting on the buffer
        '''
        return obs.copy()

    def render(self):
        return self.env.render()

    def __getattr__(self, attrname):
        return getattr(self.env, attrname)
