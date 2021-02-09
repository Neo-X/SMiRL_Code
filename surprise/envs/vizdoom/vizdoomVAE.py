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
from surprise.envs.vizdoom.buffer import VAEBuffer

# VAE stuff
from surprise.envs.vizdoom.networks import VAE
import torch
import torch.nn.functional as F
from torch import optim

import pdb

class TakeCoverVAEEnv(gym.Env):
    def __init__(self, render=False, config_path='./surprise/envs/vizdoom/scenarios/take_cover.cfg', god=True, respawn=True, device=0, vae=None, eval=False):
    #def __init__(self, render=False, config_path='/home/dangengdg/minimalEntropy/tree_search/vizdoom/scenarios/take_cover.cfg', god=True, respawn=True, device=0, vae=None, eval=False):
        # Start game
        self.game = vzd.DoomGame()

        # Set sleep time (for visualization)
        self.sleep_time = 0
        self.render = render
        if (self.render == True):
            self.game.set_window_visible(self.render)
            self.sleep_time = .02 * int(self.render)
        else:
            self.game.set_window_visible(False)

        # Game Configs
        self.game.load_config(config_path)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        self.game.set_render_screen_flashes(True)  # Effect upon taking damage
        self.episode_length = 500
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
        self.obs_hist = [self.get_random_state(res=48*64) for _ in range(4)]
        self.god = god
        self.respawn = respawn
        self.fireball = 0
        self.in_fireball = 0

        # Buffer
        self.buffer = [self.get_random_state(10*13) for _ in range(2)]

        # VAE Stuff
        self.device = device
        # Let's us point vae to a different network (pretrained or from a 
        # different env)
        if vae is None:
            self.vae = VAE().to(device)
        else:
            self.vae = vae
        self.eval = eval
        self.vae_buffer = VAEBuffer(device=device)
        self.optimizer = optim.Adam(self.vae.parameters(), lr=1e-4)
        self.batch_size = 64
        self.steps = 100
        self.vae_loss = 0

        # Spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, self.episode_length, shape=(12329,))

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
        self.vae.eval()
        obs = torch.tensor(obs).float().unsqueeze(0).to(self.device)
        obs = self.vae.encode(obs)[0]
        return obs.detach().cpu().numpy()[0]

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
        self.buffer = [self.encode(self.get_random_state(10*13)) for _ in range(2)]
        self.obs_hist = [self.get_random_state(res=48*64) for _ in range(4)]
        self.fireball = 0
        self.in_fireball = 0

        # Train vae on reset
        step_skip = 10
        if len(self.vae_buffer) > 0 and (np.random.rand() > (1/step_skip)):
            for _ in range(step_skip):
                self.vae_loss = self.step_vae(self.batch_size, self.steps)

        return self.get_obs()

    def _render(self, grayscale=True):
        state = self.game.get_state()
        if grayscale:
            return state.screen_buffer.mean(0) / 256.0
        else:
            return state.screen_buffer / 256.0

    def render(self):
        state = self.game.get_state()
        state = state.screen_buffer.transpose(1,2,0)
        imsave('./rollouts/vizdoom-nogod/{:03d}.png'.format(self.time), state)

    def render_obsres(self):
        img = self._render()
        img = cv2.resize(img, (0, 0), fx=.1, fy=.1, interpolation=cv2.INTER_AREA)
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
        img_obs = np.array(self.obs_hist).flatten()
        np_buffer = np.array(self.buffer)
        mu = np_buffer.mean(0)
        std = np_buffer.std(0)
        return np.hstack([img_obs, mu, std, self.time])

    def get_rews(self):
        #return -self.in_fireball
        np_buffer = np.array(self.buffer)
        mu = np_buffer.mean(0)
        std = np_buffer.std(0)

        curr_state = self.encode(self.render_lowres().flatten())
        logprob = -np.sum((curr_state - mu)**2 / 2 / std**2)
        logprob -= np.sum(np.log(std))
        return logprob

    def get_info(self):
        return {'lifespan': self.time,
                'fireball': self.fireball,
                'in_fireball': self.in_fireball,
                'vae_loss': self.vae_loss}

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_fn(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 130), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + .1 * KLD

    def step_vae(self, batch_size, n):
        # Don't step if eval
        if self.eval:
            return 0

        self.vae.train()
        train_loss = 0
        steps = min(n, len(self.vae_buffer)//batch_size + 1)
        for i in range(steps):
            data, _ = self.vae_buffer.sample(batch_size)
            
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.vae(data)
            loss = self.loss_fn(recon_batch, data, mu, logvar)
            print ("recon_batch: ", torch.mean(recon_batch), torch.std(recon_batch))
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            print ("loss.item(): ", loss.item())
        return train_loss / steps / batch_size

    def step(self, action):
        # Take action with skiprate
        r = self.game.make_action(self.actions[action], self.skiprate)

        # If died, then return
        if self.game.is_episode_finished():
            if self.respawn:
                self.reset_game()
            else:
                self.done = True
                return self.prev_obs, self.prev_rew, self.done, self.get_info()

        # For visualization
        if self.sleep_time > 0:
            sleep(self.sleep_time)

        # Increment time
        self.time += 1

        # Add to vae buffer
        self.vae_buffer.add(self.render_lowres().flatten())

        # If episode finished, set done
        if self.time == self.episode_length:
            self.done = True

        # Add to obs hist
        self.obs_hist.append(self.render_obsres().flatten())
        if len(self.obs_hist) > 4:
            self.obs_hist.pop(0)

        # fireball
        self.fireball += int(self.game.get_state().screen_buffer.mean() > 120)
        self.in_fireball = int(self.game.get_state().screen_buffer.mean() > 120)

        # We need to save these b/c the doom env is weird
        # After dying, we can't get any observations
        self.prev_obs = self.get_obs()
        self.prev_rew = self.get_rews()
        
        # Add to replay buffer
        self.buffer.append(self.encode(self.render_lowres().flatten()))
        
        infos = self.get_info()
        if self.render=='rgb_array':
            img = self.render_lowres(downsample_factor=0.2, grayscale=False)
            img = np.array(img * 255, dtype='uint8')
            infos["rendering"] = img

        return self.prev_obs, self.prev_rew, self.done, infos

'''
env = TakeCoverVAEEnv(render=False, god=True, respawn=True)
done = False
rews = []
pos = []
fb = []
total_rews = 0
for i in range(100):
    while not done:
        obs, rew, done, info = env.step(np.random.randint(2))
        total_rews += rew
        print(info)
        #rews.append(rew)
        #pos.append(env.game.get_state().game_variables[1])
        #fb.append(info['in_fireball'])
    done = False
    env.reset()
#env.render()
#img = env.render_lowres()
#img = cv2.resize(img, (0, 0), fx=50, fy=50, interpolation=cv2.INTER_NEAREST)
#imsave('imgs/{:03d}.jpg'.format(env.time), img)
#plt.plot(fb)
#plt.show()
'''
