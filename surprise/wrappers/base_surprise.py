import numpy as np
import gym
from gym.spaces import Box
import pdb
import util.class_util as classu

class BaseSurpriseWrapper(gym.Env):
    
    @classu.hidden_member_initialize
    def __init__(self, 
                 env, 
                 buffer, 
                 time_horizon, 
                 add_true_rew=False,
                 smirl_rew_scale=None, 
                 buffer_type=None,
                 latent_obs_size=None,
                 obs_label=None,
                 obs_out_label=None):
        '''
        params
        ======
        env (gym.Env) : environment to wrap

        buffer (Buffer object) : Buffer that tracks history and fits models
        '''

        theta = self._buffer.get_params()
        self._num_steps = 0

        # Add true reward to surprise

        # Gym spaces
        self.action_space = env.action_space
        self.env_obs_space = env.observation_space
        self.observation_space = Box(
                np.concatenate(
                    (self.env_obs_space.low.flatten(), 
                     np.zeros(theta.shape), 
                     np.zeros(1))
                ),
                np.concatenate(
                    (self.env_obs_space.high.flatten(), 
                     np.ones(theta.shape), 
                     np.ones(1)*time_horizon)
                )
            )

        self.reset()

    def step(self, action):
        # Take Action
        obs, env_rew, envdone, info = self._env.step(action)
        info['task_reward'] = env_rew


        # Get wrapper outputs
        rew = self._buffer.logprob(self.encode_obs(obs))
        # For numerical stability, clip stds to not be 0
        thresh = 300
        rew = np.clip(rew, a_min=-thresh, a_max=thresh)
        # Add observation to buffer
        self._buffer.add(self.encode_obs(obs))
        if (self._obs_out_label is None):
            info['state_entropy_smirl'] = rew
            info["theta_entropy"] = self._buffer.entropy()
        else:
            info[self._obs_out_label + 'state_entropy_smirl'] = rew
            info[self._obs_out_label + "theta_entropy"] = self._buffer.entropy()
#         print (info)
        if (self._smirl_rew_scale is not None):
            rew = (rew * self._smirl_rew_scale)
#             print("rew: ", rew, self.get_done() or envdone)
        if (self._add_true_rew == "only"):
#             print("esing env reward: ", info) 
            rew = env_rew
        elif self._add_true_rew:
            rew = (rew) + env_rew
        obs = self.get_obs(obs)
        
        self._num_steps += 1
        return obs, rew, self.get_done() or envdone, info

    def get_done(self):
        return self.num_steps >= self._time_horizon

    def get_obs(self, obs):
        '''
        Augment observation, perhaps with generative model params
        '''
        theta = self._buffer.get_params()
        num_samples = np.ones(1)*self._buffer.buffer_size
        if (self._obs_out_label is None):
            obs = np.concatenate([np.array(obs).flatten(), np.array(theta).flatten(), num_samples])
        else:
#             print ("obs: ", obs)
#             import matplotlib.pyplot as plt
#             print ("obs: ", obs)
#             plt.imshow(np.reshape(obs["observation"], (64,48,4)))
#             plt.show()
            obs[self._obs_out_label] = np.concatenate([np.array(theta).flatten(), num_samples])
        
        return obs

    #def get_done(self, env_done):
    def get_done(self):
        '''
        figure out if we're done

        params
        ======
        env_done (bool) : done bool from the wrapped env, doesn't 
            necessarily need to be used
        '''
        return self._num_steps >= self._time_horizon

    def reset(self):
        '''
        Reset the wrapped env and the buffer
        '''
        obs = self._env.reset()
#         print ("surprise obs shape1, ", obs.shape)
        self._buffer.reset()
        self._num_steps = 0
        obs = self.get_obs(obs)
#         print ("surprise obs shape2, ", obs.shape)
        return obs

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def encode_obs(self, obs):
        '''
        Used to encode the observation before putting on the buffer
        '''
        if self._obs_label is None:
            return np.array(obs).flatten().copy()
        else:
            return np.array(obs[self._obs_label]).flatten().copy()
