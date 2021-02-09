import os
import gym
from gym import spaces

import numpy as np
import matplotlib.pyplot as plt
import cv2

import pdb

class IslandEnv(gym.Env):
    def __init__(self, size=30, islandSize=.1, episode_length=1000):
        # Ocean parameters
        self.size = size

        # Island coords
        self.island = set()
        island_start = int(self.size*(1-islandSize)//2)
        island_size = int(size * islandSize)
        for i in range(island_start, island_start + island_size):
            for j in range(island_start, island_start + island_size):
                self.island.add((i,j))

        # Shaking params
        self.islandShake = 0
        self.oceanShake = 1

        # Agent position
        self.pos = np.random.randint(0, self.size, 2)

        # Timer
        self.time = 0
        self.episode_length = episode_length

        # gym spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(np.zeros(self.size**2), 
                                            np.zeros(self.size**2))

        # Visualization stuff (can delete later?
        # or make more permanent somehow???)
        self.rollout_num = 0
        self.rollouts = []
        self.rollout = []

        # reset
        self.reset()

    def on_island(self):
        return tuple(self.pos) in self.island

    def move(self, direction):
        axis = direction // 2
        direction = (direction % 2) * 2 - 1

        self.pos[axis] += direction

        # Undo if there is a collision
        collision = np.max(self.pos) >= self.size or np.min(self.pos) < 0
        if collision:
            self.pos[axis] -= direction

    def jitter(self):
        # Find out if we're on island and set shake prob
        jitter_prob = self.islandShake if self.on_island() else self.oceanShake

        if np.random.uniform() < jitter_prob:
            self.move(np.random.randint(4))

    def step(self, action):
        # Move
        self.move(action)

        # Jitter
        self.jitter()

        # Update state
        self.time += 1

        # Visualization (delete later or make more peramanent)
        self.rollout.append(self.pos.copy())

        return self.get_obs(), self.get_reward(), self.get_done(), self.get_info()

    def get_info(self):
        return {}

    def get_done(self):
        return self.time >= self.episode_length

    def get_reward(self):
        return int(self.on_island())

    def get_obs(self):
        obs = np.zeros(self.size**2)
        obs[self.pos[0] * self.size + self.pos[1]] = 1
        return obs

    def render(self):
        # Plot island and agent
        island = np.array(list(self.island)).T
        plt.scatter(island[0], island[1])
        plt.scatter(self.pos[0], self.pos[1])

        # Axes stuff
        plt.gca().set_xlim([0, self.size])
        plt.gca().set_ylim([0, self.size])
        plt.gca().set_aspect('equal')

        # plt.show()
        '''
        plt.savefig('rollouts/island/{}_{}.png'.format(
            str(self.rollout_num).zfill(3), 
            str(self.time).zfill(5)))
        print(self.rollout_num, self.time)
        plt.clf()
        '''

    def render_rollouts(self):
        plt.clf()
        rollouts = np.array(self.rollouts)

        for i in range(rollouts.shape[0]):
            plt.plot(rollouts[i,:,0], rollouts[i,:,1])

        plt.gca().set_xlim([0, self.size-1])
        plt.gca().set_ylim([0, self.size-1])
        plt.gca().set_aspect('equal')

        plt.show()

    def reset(self):
        self.pos = np.random.randint(0, self.size, 2)
        self.time = 0
        self.rollout_num += 1

        #if len(self.rollout) == self.episode_length:
        if len(self.rollout) == 50:
            self.rollouts.append(self.rollout)
        self.rollout = []

        return self.get_obs()

if __name__ == "__main__":
    env = IslandEnv()
    env.reset()
    
    for i in range(100):
        env.render()
        print(env.step(np.random.randint(4)))
