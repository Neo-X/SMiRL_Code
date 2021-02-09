import sys

from time import sleep
import pdb

import gym
from gym import spaces
import vizdoom as vzd

import numpy as np
import matplotlib.pyplot as plt
import cv2
#from scipy.misc import imsave

import torch
import torch.nn.functional as F
import torch.optim as optim
import rlkit.torch.pytorch_util as ptu
device = ptu.device

from surprise.envs.vizdoom.buffer import Buffer
from surprise.envs.vizdoom.networks import MLP, VizdoomFeaturizer

class TakeCoverEnv_Curiosity(gym.Env):
    #def __init__(self, render=False, config_path='/home/dangengdg/minimalEntropy/tree_search/vizdoom/scenarios/take_cover.cfg', god=False, respawn=True):
    def __init__(self, render=False, config_path='/home/gberseth/playground/BayesianSurpriseCode/surprise/envs/vizdoom/scenarios/take_cover.cfg', god=False, respawn=True):
        # Start game
        self.game = vzd.DoomGame()

        # Set sleep time (for visualization)
        self.sleep_time = 0
        self.game.set_window_visible(render)
        self.sleep_time = .02 * int(render)

        # Game Configs
        self.game.load_config(config_path)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        self.game.set_render_screen_flashes(True)  # Effect upon taking damage
        self.episode_length = 1000
        self.skiprate = 2
        self.game.set_episode_timeout(self.episode_length * self.skiprate)

        # Initialize the game
        self.game.init()

        # Actions are left or right
        self.actions = [[True, False], [False, True]]

        # Env Variables
        self.done = False
        self.time = 1
        self.downsample_factor = .02
        self.obs_hist = [self.get_random_state(res=(48,64)) for _ in range(4)]
        self.god = god
        self.respawn = respawn
        self.deaths = 0
        self.fireball = 0

        # Spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, self.episode_length, shape=(4, 48, 64))

        # Curiosity stuff
        self.buffer = Buffer(device=device)
        feature_dim = 32
        self.forward_network = MLP([feature_dim + 2, 64, 64, feature_dim]).to(device)
        self.inverse_network = MLP([feature_dim * 2, 32, 16, 8, 2], log_softmax=True).to(device)
        self.featurizer = VizdoomFeaturizer(feature_dim).to(device)
        self.optimizer_forward = optim.Adam(self.forward_network.parameters(), lr=1e-4)
        self.optimizer_inverse = optim.Adam(list(self.inverse_network.parameters()) + 
                                    list(self.featurizer.parameters()), lr=1e-4)
        self.step_freq = 8
        self.inverse_loss = torch.zeros(1)
        self.forward_loss = torch.zeros(1)
        self.loss = torch.zeros(1)

        self.reset()

    def get_random_state(self, res):
        '''
        Get a random, gaussian state (roughly the average state)
        '''
        return .3 + np.random.randn(*res) / 100

    def encode(self, obs):
        '''
        Encodes a (lowres) buffer observation
        '''
        return obs

    def reset_game(self):
        # New episode
        self.game.new_episode()

        # Set invincible
        if self.god:
            self.game.send_game_command('god')

    def reset(self):
        self.reset_game()

        # Env Variables
        self.done = False
        self.time = 1
        self.downsample_factor = .02
        self.obs_hist = [self.get_random_state(res=(48,64)) for _ in range(4)]
        self.deaths = 0
        self.fireball = 0

        # Losses
        self.inverse_loss = torch.zeros(1)
        self.forward_loss = torch.zeros(1)
        self.loss = torch.zeros(1)

        return self.get_obs()

    def _render(self):
        state = self.game.get_state()
        return state.screen_buffer.mean(0) / 256.0

    def render(self):
        state = self.game.get_state()
        state = state.screen_buffer.transpose(1,2,0)
        imsave('./rollouts/vizdoom-nogod/{:03d}.png'.format(self.time), state)

    def render_obsres(self):
        img = self._render()
        img = cv2.resize(img, (0, 0), fx=.1, fy=.1, interpolation=cv2.INTER_AREA)
        return img

    def render_lowres(self):
        img = self._render()
        img = cv2.resize(img, (0, 0), 
                         fx=self.downsample_factor, 
                         fy=self.downsample_factor,
                         interpolation=cv2.INTER_AREA)
        return img

    def get_obs(self):
        # We can reshape in the network
        img_obs = np.array(self.obs_hist).flatten()
        return img_obs

    def get_rews(self, sas):
        # Set phase
        self.featurizer.eval()
        self.forward_network.eval()

        # Unpack
        s,a,sp = sas

        # Convert to tensor
        s = torch.tensor(s).float().to(device).unsqueeze(0)
        a = torch.tensor(a).float().to(device).unsqueeze(0)
        sp = torch.tensor(sp).float().to(device).unsqueeze(0)

        # featurize
        fs, fsp = self.featurizer(s), self.featurizer(sp)

        # forward model
        state_pred = self.forward_network(torch.cat([fs, a], dim=1))

        # losses, save as class parameters to pass to info
        return F.mse_loss(state_pred, fsp).item()

    def get_info(self):
        return {'deaths': self.deaths,
                'inverse_loss': self.inverse_loss.item(),
                'forward_loss': self.forward_loss.item(),
                'loss': self.loss.item(),
                'fireball': self.fireball}

    def step(self, action):
        # Make (s,a,s') to add to buffer
        data = [np.array(self.obs_hist), np.eye(2)[action], None]

        # Take action with skiprate
        r = self.game.make_action(self.actions[action], self.skiprate)

        # If died, then return
        if self.game.is_episode_finished():
            if self.respawn:
                self.deaths += 1
                self.reset_game()
            else:
                self.done = True
                return self.prev_obs, self.prev_rew, self.done, self.get_info()

        # For visualization
        if self.sleep_time > 0:
            sleep(self.sleep_time)

        # Increment time
        self.time += 1

        # If episode finished, set done
        if self.time == self.episode_length:
            self.done = True

        # Add to obs hist
        self.obs_hist.append(self.render_obsres())
        if len(self.obs_hist) > 4:
            self.obs_hist.pop(0)

        # Finish off (s,a,s') tuplet and add to buffer
        data[2] = np.array(self.obs_hist)
        self.buffer.add(tuple(data))

        # We need to save these b/c the doom env is weird
        # After dying, we can't get any observations
        self.prev_obs = self.get_obs()
        self.prev_rew = self.get_rews(data)

        # Update network
        if self.time % self.step_freq == 0:
            self.step_net()

        # Update fireball var
        self.fireball += int(self.game.get_state().screen_buffer.mean() > 120)

        return self.prev_obs, self.prev_rew, self.done, self.get_info()

    def step_net(self, batch_size=64):
        # Set phase
        self.forward_network.train()
        self.inverse_network.train()
        self.featurizer.train()

        # Get data (s,a,s')
        s, a, sp = self.buffer.sample(batch_size)

        # featurize
        fs, fsp = self.featurizer(s), self.featurizer(sp)

        # inverse model
        action_pred = self.inverse_network(torch.cat([fs, fsp], dim=1))

        # forward model
        state_pred = self.forward_network(torch.cat([fs, a], dim=1).detach())

        # losses, save as class parameters to pass to info
        self.inverse_loss = F.nll_loss(action_pred, torch.argmax(a, dim=1))
        self.forward_loss = F.mse_loss(state_pred, fsp)
        self.loss = self.inverse_loss + self.forward_loss

        # Step
        self.optimizer_forward.zero_grad()
        self.optimizer_inverse.zero_grad()
        self.loss.backward()
        self.optimizer_forward.step()
        self.optimizer_inverse.step()

# env = TakeCoverEnv_Curiosity(render=False, god=True)
# done = False
# losses = []
# for_losses = []
# inv_losses = []
# pix = []
# 
# while not done:
#     print(env.time)
#     obs, rew, done, info = env.step(np.random.randint(2))
#     losses.append(info['loss'])
#     for_losses.append(info['forward_loss'])
#     inv_losses.append(info['inverse_loss'])
#     pix.append(info['fireball'])
# done = False
# env.reset()
# 
# plt.plot(pix)
# plt.show()
