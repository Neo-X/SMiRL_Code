import sys
sys.path.append('/home/dangengdg/minimalEntropy')

import gym
from gym import spaces
import vizdoom as vzd

from time import sleep
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import pdb

'''
True Reward:
    god=False
    respawn=False
    skill=5
    augment_obs=False
    true_rew=True

Surprise:
    god=True
    respawn=True
    skill=3
    augment_obs=True
    true_rew=False

Surprise No Theta:
    god=True
    respawn=True
    skill=3
    augment_obs=False
    true_rew=False
'''

class DefendTheLineEnv(gym.Env):
    def __init__(self, render=False, config_path='surprise/envs/vizdoom/scenarios/defend_the_line.cfg', god=True, respawn=True, skill=3, augment_obs=True, true_rew=False, joint_rew=False):
    #def __init__(self, render=False, config_path='/home/daniel/Documents/minimalEntropy/tree_search/vizdoom/scenarios/defend_the_line.cfg', god=True, respawn=True, skill=3, augment_obs=True, true_rew=False, joint_rew=False):
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
        self.game.set_doom_skill(skill)
        self.episode_length = 1000
        self.skiprate = 2
        self.game.set_episode_timeout(self.episode_length * self.skiprate)

        # Initialize the game
        self.game.init()

        # Actions are left or right
        self.actions = [list(x) for x in np.eye(self.game.get_available_buttons_size()).astype(bool)]

        # Env Variables
        self.done = False
        self.time = 1
        self.downsample_factor = .02
        self.obs_hist = [self.get_random_state(res=48*64) for _ in range(4)]
        self.god = god
        self.respawn = respawn
        self.deaths = 0
        self.augment_obs = augment_obs
        self.true_rew = true_rew
        self.joint_rew = joint_rew

        # Buffer
        self.buffer = [self.get_random_state(10*13) for _ in range(2)]

        # Spaces
        self.action_space = spaces.Discrete(3)

        if self.augment_obs:
            self.observation_space = spaces.Box(0, self.episode_length, shape=(13069,))
        else:
            self.observation_space = spaces.Box(0, self.episode_length, shape=(12289,))

        self.reset()

    def get_random_state(self, res):
        '''
        Get a random, gaussian state (roughly the average state)
        '''
        return .3 + np.random.randn(res) / 100

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
        self.buffer = [self.get_random_state(3*10*13) for _ in range(2)]
        self.obs_hist = [self.get_random_state(res=48*64) for _ in range(4)]
        self.deaths = 0

        return self.get_obs()

    def _render(self):
        state = self.game.get_state()
        return state.screen_buffer.mean(0) / 256.0

    def render(self):
        try:
            state = self.game.get_state()
            state = state.screen_buffer.transpose(1,2,0)
            im = Image.fromarray(state)
            im.save('./rollouts/vizdoom-dtl-surprise/{:04d}.png'.format(self.time))
        except:
            pass

    def render_obsres(self):
        img = self._render()
        img = cv2.resize(img, (0, 0), fx=.1, fy=.1, interpolation=cv2.INTER_AREA)
        return img

    def render_lowres(self):
        #img = self._render()
        state = self.game.get_state()
        img = state.screen_buffer / 256.0
        img = cv2.resize(img.transpose(1,2,0), (0, 0), 
                         fx=self.downsample_factor, 
                         fy=self.downsample_factor,
                         interpolation=cv2.INTER_AREA)
        return img

    def get_obs(self):
        # We can reshape in the network
        img_obs = np.array(self.obs_hist).flatten()
        np_buffer = np.array(self.buffer)
        mu = np_buffer.mean(0)
        std = np_buffer.std(0)
        if self.augment_obs:
            return np.hstack([img_obs, mu, std, self.time])
        else:
            return np.hstack([img_obs, self.time])

    def get_rews(self):
        if self.true_rew:
            return 100
        np_buffer = np.array(self.buffer)
        mu = np_buffer.mean(0)
        std = np_buffer.std(0)

        curr_state = self.render_lowres().flatten()
        logprob = -np.sum((curr_state - mu)**2 / 2 / std**2)
        logprob -= np.sum(np.log(std))
        if self.joint_rew:
            return logprob + 1000
        else:
            return logprob

    def get_info(self):
        return {'lifespan': self.time,
                'deaths': self.deaths,
                'kills': self.game.get_state().game_variables[2]}

    def step(self, action):
        # If only shoot if we have enough bullets, else random op
        if action==2:
            if self.game.get_state().game_variables[0] == 0:
                action = np.random.randint(2)

        # Get info before action, otherwise breaks on death
        info = self.get_info()

        # Take action with skiprate
        r = self.game.make_action(self.actions[action], self.skiprate)

        # If died, then return
        if self.game.is_episode_finished():
            if self.respawn:
                self.deaths += 1
                self.reset_game()
            else:
                self.done = True
                return self.prev_obs, self.prev_rew, self.done, info

        # For visualization
        if self.sleep_time > 0:
            sleep(self.sleep_time)

        # Increment time
        self.time += 1

        # Add to replay buffer
        self.buffer.append(self.render_lowres().flatten())

        # If episode finished, set done
        if self.time == self.episode_length:
            self.done = True

        # Add to obs hist
        self.obs_hist.append(self.render_obsres().flatten())
        if len(self.obs_hist) > 4:
            self.obs_hist.pop(0)

        # We need to save these b/c the doom env is weird
        # After dying, we can't get any observations
        self.prev_obs = self.get_obs()
        self.prev_rew = self.get_rews()

        infos = self.get_info()
        if self.render=='rgb_array':
            img = self.render_lowres(downsample_factor=0.2, grayscale=False)
            img = np.array(img * 255, dtype='uint8')
            infos["rendering"] = img

        return self.prev_obs, self.prev_rew, self.done, infos

class DefendTheLineEnv_TrackThetas(gym.Env):
    def __init__(self, render=False, config_path='/home/dangengdg/minimalEntropy/tree_search/vizdoom/scenarios/defend_the_line.cfg', god=True, respawn=True, skill=3, augment_obs=True, true_rew=False):
    #def __init__(self, render=False, config_path='/home/daniel/Documents/minimalEntropy/tree_search/vizdoom/scenarios/defend_the_line.cfg', god=True, respawn=True, skill=3, augment_obs=True, true_rew=False):
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
        self.game.set_doom_skill(skill)
        self.episode_length = 1000
        self.skiprate = 2
        self.game.set_episode_timeout(self.episode_length * self.skiprate)

        # Initialize the game
        self.game.init()

        # Actions are left or right
        self.actions = [list(x) for x in np.eye(self.game.get_available_buttons_size()).astype(bool)]

        # Env Variables
        self.done = False
        self.time = 1
        self.downsample_factor = .02
        self.obs_hist = [self.get_random_state(res=48*64) for _ in range(4)]
        self.god = god
        self.respawn = respawn
        self.deaths = 0
        self.augment_obs = augment_obs
        self.true_rew = true_rew

        # Buffer
        self.buffer = [self.get_random_state(10*13) for _ in range(2)]

        # Spaces
        self.action_space = spaces.Discrete(3)

        if self.augment_obs:
            self.observation_space = spaces.Box(0, self.episode_length, shape=(13069,))
        else:
            self.observation_space = spaces.Box(0, self.episode_length, shape=(12289,))

        self.thetas = []

        self.reset()

    def get_random_state(self, res):
        '''
        Get a random, gaussian state (roughly the average state)
        '''
        return .3 + np.random.randn(res) / 100

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
        self.buffer = [self.get_random_state(3*10*13) for _ in range(2)]
        self.obs_hist = [self.get_random_state(res=48*64) for _ in range(4)]
        self.deaths = 0
        self.thetas = []

        return self.get_obs()

    def _render(self):
        state = self.game.get_state()
        return state.screen_buffer.mean(0) / 256.0

    def render(self):
        try:
            state = self.game.get_state()
            state = state.screen_buffer.transpose(1,2,0)
            im = Image.fromarray(state)
            im.save('./rollouts/vizdoom-dtl-surprise/{:04d}.png'.format(self.time))
        except:
            pass

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

        self.thetas.append(np.hstack([mu, std]))
        np.save('./rollouts/vizdoom-dtl-surprise/thetas.npy', np.array(self.thetas))

        if self.augment_obs:
            return np.hstack([img_obs, mu, std, self.time])
        else:
            return np.hstack([img_obs, self.time])

    def get_rews(self):
        if self.true_rew:
            return 100
        np_buffer = np.array(self.buffer)
        mu = np_buffer.mean(0)
        std = np_buffer.std(0)

        curr_state = self.render_lowres().flatten()
        logprob = -np.sum((curr_state - mu)**2 / 2 / std**2)
        logprob -= np.sum(np.log(std))
        return logprob

    def get_info(self):
        return {'lifespan': self.time,
                'deaths': self.deaths,
                'kills': self.game.get_state().game_variables[2]}

    def step(self, action):
        # If only shoot if we have enough bullets, else random op
        if action==2:
            if self.game.get_state().game_variables[0] == 0:
                action = np.random.randint(2)

        # Get info before action, otherwise breaks on death
        info = self.get_info()

        # Take action with skiprate
        r = self.game.make_action(self.actions[action], self.skiprate)

        # If died, then return
        if self.game.is_episode_finished():
            if self.respawn:
                self.deaths += 1
                self.reset_game()
            else:
                self.done = True
                return self.prev_obs, self.prev_rew, self.done, info

        # For visualization
        if self.sleep_time > 0:
            sleep(self.sleep_time)

        # Increment time
        self.time += 1

        # Add to replay buffer
        self.buffer.append(self.render_lowres().flatten())

        # If episode finished, set done
        if self.time == self.episode_length:
            self.done = True

        # Add to obs hist
        self.obs_hist.append(self.render_obsres().flatten())
        if len(self.obs_hist) > 4:
            self.obs_hist.pop(0)

        # We need to save these b/c the doom env is weird
        # After dying, we can't get any observations
        self.prev_obs = self.get_obs()
        self.prev_rew = self.get_rews()

        return self.prev_obs, self.prev_rew, self.done, info

class DefendTheLineEnv_InjectThetas(gym.Env):
    def __init__(self, render=False, config_path='/home/dangengdg/minimalEntropy/tree_search/vizdoom/scenarios/defend_the_line.cfg', god=True, respawn=True, skill=3, augment_obs=True, true_rew=False):
    #def __init__(self, render=False, config_path='/home/daniel/Documents/minimalEntropy/tree_search/vizdoom/scenarios/defend_the_line.cfg', god=True, respawn=True, skill=3, augment_obs=True, true_rew=False):
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
        self.game.set_doom_skill(skill)
        self.episode_length = 1000
        self.skiprate = 2
        self.game.set_episode_timeout(self.episode_length * self.skiprate)

        # Initialize the game
        self.game.init()

        # Actions are left or right
        self.actions = [list(x) for x in np.eye(self.game.get_available_buttons_size()).astype(bool)]

        # Env Variables
        self.done = False
        self.time = 1
        self.downsample_factor = .02
        self.obs_hist = [self.get_random_state(res=48*64) for _ in range(4)]
        self.god = god
        self.respawn = respawn
        self.deaths = 0
        self.augment_obs = augment_obs
        self.true_rew = true_rew

        self.thetas = np.load('/home/dangengdg/bin/rlkit_coline/rollouts/vizdoom-dtl-surprise/thetas.npy')

        # Buffer
        self.buffer = [self.get_random_state(10*13) for _ in range(2)]

        # Spaces
        self.action_space = spaces.Discrete(3)

        if self.augment_obs:
            self.observation_space = spaces.Box(0, self.episode_length, shape=(13069,))
        else:
            self.observation_space = spaces.Box(0, self.episode_length, shape=(12289,))

        self.reset()

    def get_random_state(self, res):
        '''
        Get a random, gaussian state (roughly the average state)
        '''
        return .3 + np.random.randn(res) / 100

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
        self.buffer = [self.get_random_state(3*10*13) for _ in range(2)]
        self.obs_hist = [self.get_random_state(res=48*64) for _ in range(4)]
        self.deaths = 0

        return self.get_obs()

    def _render(self):
        state = self.game.get_state()
        return state.screen_buffer.mean(0) / 256.0

    def render(self):
        try:
            state = self.game.get_state()
            state = state.screen_buffer.transpose(1,2,0)
            im = Image.fromarray(state)
            im.save('./rollouts/vizdoom-dtl-surprise/{:04d}.png'.format(self.time))
        except:
            pass

    def render_obsres(self):
        img = self._render()
        img = cv2.resize(img, (0, 0), fx=.1, fy=.1, interpolation=cv2.INTER_AREA)
        return img

    def render_lowres(self):
        #img = self._render()
        state = self.game.get_state()
        img = state.screen_buffer / 256.0
        img = cv2.resize(img.transpose(1,2,0), (0, 0), 
                         fx=self.downsample_factor, 
                         fy=self.downsample_factor,
                         interpolation=cv2.INTER_AREA)
        return img

    def get_obs(self):
        # We can reshape in the network
        img_obs = np.array(self.obs_hist).flatten()
        np_buffer = np.array(self.buffer)

        mu = np_buffer.mean(0)
        std = np_buffer.std(0)

        thetas = self.thetas[self.time - 2]

        if self.augment_obs:
            return np.hstack([img_obs, thetas, self.time])
        else:
            return np.hstack([img_obs, self.time])

    def get_rews(self):
        if self.true_rew:
            return 100
        np_buffer = np.array(self.buffer)
        mu = np_buffer.mean(0)
        std = np_buffer.std(0)

        curr_state = self.render_lowres().flatten()
        logprob = -np.sum((curr_state - mu)**2 / 2 / std**2)
        logprob -= np.sum(np.log(std))
        return logprob

    def get_info(self):
        return {'lifespan': self.time,
                'deaths': self.deaths,
                'kills': self.game.get_state().game_variables[2]}

    def step(self, action):
        # If only shoot if we have enough bullets, else random op
        if action==2:
            if self.game.get_state().game_variables[0] == 0:
                action = np.random.randint(2)

        # Get info before action, otherwise breaks on death
        info = self.get_info()

        # Take action with skiprate
        r = self.game.make_action(self.actions[action], self.skiprate)

        # If died, then return
        if self.game.is_episode_finished():
            if self.respawn:
                self.deaths += 1
                self.reset_game()
            else:
                self.done = True
                return self.prev_obs, self.prev_rew, self.done, info

        # For visualization
        if self.sleep_time > 0:
            sleep(self.sleep_time)

        # Increment time
        self.time += 1

        # Add to replay buffer
        self.buffer.append(self.render_lowres().flatten())

        # If episode finished, set done
        if self.time == self.episode_length:
            self.done = True

        # Add to obs hist
        self.obs_hist.append(self.render_obsres().flatten())
        if len(self.obs_hist) > 4:
            self.obs_hist.pop(0)

        # We need to save these b/c the doom env is weird
        # After dying, we can't get any observations
        self.prev_obs = self.get_obs()
        self.prev_rew = self.get_rews()

        return self.prev_obs, self.prev_rew, self.done, info

'''
env = DefendTheLineEnv(render=False, god=False, respawn=False, skill=5, augment_obs=True, joint_rew=True)
done = False
rews = []
kills = []
for i in range(1):
    while not done:
        #kills.append(env.game.get_state().game_variables[1])
        obs, rew, done, info = env.step(np.random.randint(3))
        #print(obs.shape)
        #print(info)
        #print(env.time)
        #print(obs[-781])
        rews.append(rew)
        #env.render()
    done = False
    env.reset()
    #img = env.render_lowres()
    #img = cv2.resize(img, (0, 0), fx=50, fy=50, interpolation=cv2.INTER_NEAREST)
    #imsave('imgs/{:03d}.jpg'.format(env.time), img)
'''
