import sys
sys.path.append('/home/daniel/Documents/minimalEntropy')

import gym
from gym import spaces
import vizdoom as vzd

from random import choice
from time import sleep
import cv2
import numpy as np
import matplotlib.pyplot as plt
#from scipy.misc import imsave
import torch
from surprise.envs.vizdoom.networks import VAE
import pdb

class VizDoomEnv(gym.Env):
    
    def __init__(self, render=False, config_path='./surprise/envs/vizdoom/scenarios/take_cover.cfg', god=True, num_actions=2, **kwargs):
    #def __init__(self, render=False, config_path='/home/dangengdg/minimalEntropy/tree_search/vizdoom/scenarios/take_cover.cfg', vae=False, god=True, respawn=True):
        # Start game
        self.game = vzd.DoomGame()

        # Set sleep time (for visualization)
        self.sleep_time = 0
        self.__render = render
        if (self.__render == True):
            self.game.set_window_visible(self.__render)
            self.sleep_time = .02 * int(self.__render)
        else:
            self.game.set_window_visible(False)
            
        # Game Configs
        self.game.load_config(config_path)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
#         self.game.set_screen_format(vzd.ScreenFormat.BGR24)
#         self.game.set_screen_format(vzd.ScreenFormat.DOOM_256_COLORS8)
        self.game.set_render_screen_flashes(True)  # Effect upon taking damage
        self.episode_length = 1000
        self.skiprate = 2
        self.game.set_episode_timeout(self.episode_length * self.skiprate)

        # Initialize the game
        self.game.init()

        # Actions are left or right
#         self.actions = [[True, False], [False, True]]
        self.actions = [list(x) for x in np.eye(self.game.get_available_buttons_size()).astype(bool)]

        # Env Variables
        self.done = False
        self.time = 1
        self.downsample_factor = .02
        self.god = god
        self.fireball = 0
        self.in_fireball = 0
        self.health = 100

        # Buffer
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(-1, 1, shape=(120, 160, 3))

        self.reset()

    def get_random_state(self, res):
        '''
        Get a random, gaussian state (roughly the average state)
        '''
        return .3 + np.random.randn(res) / 100

    def encode(self, obs):
        '''
        Encodes a (lowres) buffer observation
        '''
        if self.vae is None:
            return obs
        else:
            obs = torch.tensor(obs).float().unsqueeze(0).to(0)
            obs = self.vae.encode(obs)[0]
            return obs.detach().cpu().numpy()[0]

    def reset_game(self):
        # New episode
        self.game.new_episode()
        self.game.set_render_screen_flashes(True)  # Effect upon taking damage

        # Set invincible
        if self.god:
            self.game.send_game_command('god')

    def reset(self):
        self.reset_game()

        # Env Variables
        self.done = False
        self.time = 1
        self.fireball = 0
        self.in_fireball = 0

        return self.get_obs()

    def _render(self, grayscale=False):
        state = self.game.get_state()
        img = state.screen_buffer
        img = np.moveaxis(state.screen_buffer, 0, 2)
#         img = np.moveaxis(img, 0, 1) 
#         print ("state.screen_buffer: ", img.shape)
#         print ("state.screen_buffer: ", img)
#         plt.imshow(img)
#         plt.show()
        if grayscale:
            return img.mean(0) / 256.0
        else:
            return img / 256.0

    def render(self, mode=False):
        state = self.game.get_state()
        state = state.screen_buffer.transpose(1,2,0)
        state = cv2.resize(state, dsize=(64,48), interpolation=cv2.INTER_AREA)
#         imsave('./rollouts/vizdoom-nogod/{:03d}.png'.format(self.time), state)
#         print ("dom rend state: ", state.shape)
        return state

    def render_obsres(self):
        img = self._render()
#         print("Vizdoom img shape: ", img.shape)
#         img = cv2.resize(img, (0, 0), fx=.1, fy=.1, interpolation=cv2.INTER_AREA)
        return img

    def render_lowres(self, downsample_factor=None, grayscale=True):
        if downsample_factor is None:
            downsample_factor = self.downsample_factor
        img = self._render(grayscale=grayscale)
        if grayscale:
            img = cv2.resize(img, (0, 0), 
                         fx=downsample_factor, 
                         fy=downsample_factor,
                         interpolation=cv2.INTER_AREA)
        else:
            img = np.moveaxis(img, 0, -1)
            img = cv2.resize(img, (0, 0), 
                         fx=downsample_factor, 
                         fy=downsample_factor,
                         interpolation=cv2.INTER_AREA)
        return img

    def get_obs(self):
        # We can reshape in the network
        img_obs = self.render_obsres()
        return img_obs

    def get_info(self):
        return {'lifespan': self.time,
                'fireball': self.fireball,
                'in_fireball': self.in_fireball,
                'health': self.health,
                "self.POSITION_Y": self.POSITION_Y,
                "self.AMMO2": self.AMMO2,
                "self.KILLCOUNT": self.KILLCOUNT}

    def step(self, action):
        # Take action with skiprate
        r = self.game.make_action(self.actions[action], self.skiprate)


#         print ("self.game.is_episode_finished(): ", self.game.is_episode_finished(), r)
        # For visualization
        if self.sleep_time > 0:
            sleep(self.sleep_time)

        self.done = self.game.is_episode_finished()

        # fireball
        if not self.done:
            self.fireball += int(self.game.get_state().screen_buffer.mean() > 120)
            self.in_fireball = int(self.game.get_state().screen_buffer.mean() > 120)
            self.health = self.game.get_state().game_variables[0]
            self.POSITION_Y = self.game.get_state().game_variables[1]
            self.AMMO2 = self.game.get_state().game_variables[2]
            self.KILLCOUNT = self.game.get_state().game_variables[3]
        else:
            self.health = 0

        # We need to save these b/c the doom env is weird
        # After dying, we can't get any observations
        if not self.done:
            self.prev_obs = self.get_obs()
        self.prev_rew = r
        self.time = self.time + 1
        
        infos = self.get_info()
#         print (infos)
        return self.prev_obs, self.prev_rew, self.done, infos

'''
env = TakeCoverEnv(render=False, vae=False, god=True, respawn=True)
done = False
rews = []
pos = []
fb = []
total_rews = 0
while not done:
    obs, rew, done, info = env.step(np.random.randint(2))
    print(rew)
    total_rews += rew
    rews.append(rew)
    pos.append(env.game.get_state().game_variables[1])
    fb.append(info['in_fireball'])
done = False
env.reset()
#env.render()
#img = env.render_lowres()
#img = cv2.resize(img, (0, 0), fx=50, fy=50, interpolation=cv2.INTER_NEAREST)
#imsave('imgs/{:03d}.jpg'.format(env.time), img)
plt.plot(fb)
plt.show()
'''
