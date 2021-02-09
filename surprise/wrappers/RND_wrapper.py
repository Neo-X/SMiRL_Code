import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

class RNDWrapper(gym.Env):
    
    def __init__(self, env, network, device=0, obs_key=None, hist_size=5000,
                 reward_func=None,  **kwargs):
#         from surprise.envs.vizdoom.networks import VAEConv
#         from surprise.envs.vizdoom.buffer import VAEBuffer
        from surprise.envs.vizdoom.buffer import SimpleBuffer
        from surprise.envs.vizdoom.networks import VizdoomFeaturizer
        from rlkit.torch.networks import Mlp
        from torch import optim
        from gym import spaces
        '''
        params
        ======
        env (gym.Env) : environment to wrap

        '''
        self.device=device
        self.env = env
        self._obs_key = obs_key
        self._reward_func = reward_func
        
        # Gym spaces
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        
#         RND stuff
        self._buffer = SimpleBuffer(device=self.device, size=hist_size)
        if (kwargs["network_type"] == "flat"):
            self.target_net = Mlp(hidden_sizes=[128, 64],
                                    input_size=self.observation_space.low.size,
                                    output_size=64,).to(self.device)
            self.target_net.eval()
            self.pred_net = Mlp(hidden_sizes=[128, 64, 32],
                                input_size=self.observation_space.low.size,
                                output_size=64,).to(self.device)
        else:
            self.target_net = VizdoomFeaturizer(kwargs["encoding_size"]).to(self.device)
            self.target_net.eval()
            self.pred_net = VizdoomFeaturizer(kwargs["encoding_size"]).to(self.device)
        self.optimizer = optim.Adam(self.pred_net.parameters(), lr=1e-4)
        self.network = self.pred_net
        self.step_freq = 16
        self.loss = torch.zeros(1)

    def step(self, action):
        # Take Action
        obs, rew, done, info = self.env.step(action)
        # Finish off (s,a,s') tuplet and add to buffer
        if (self._obs_key is None):
            target = self.target_net(torch.tensor(obs).float().unsqueeze(0).to(self.device))
            data = [obs, target.detach().cpu().numpy()[0]]
        else:
            target = self.target_net(torch.tensor(obs[self._obs_key]).float().unsqueeze(0).to(self.device))
            data = [obs[self._obs_key], target.detach().cpu().numpy()[0]]
        self._buffer.add(tuple(data))

        # Get wrapper outputs
#         print ("data:", data)
        
                # Update network
        if self._time % self.step_freq == 0:
            self.step_model()
            
        obs = self.encode_obs(obs)
        info["rnd_loss"] = self.loss.item()
        info["task_reward"] = rew
        done = self.get_done(done)
        
        self._time = self._time + 1
        
        rew = self.get_rews(data)
        if (self._reward_func == "add"):
            rew = rew + info["task_reward"]
#             print("Add reward: ", rew)
        return obs, rew, done, info
    
    def step_model(self, batch_size=64):
        # Set phase
        self.pred_net.train()

        # Get data (s,a,s')
        data, target = self._buffer.sample(batch_size)
#         for data_ in data:
#             print ("data: ", np.array(data_).shape)
        # Do i have to tensor-ify (i think so...)
        data = torch.tensor(data).to(self.device).float()
        target = torch.tensor(target).to(self.device).float()

        # forward model
        pred = self.pred_net(data)

        # losses, save as class parameters to pass to info
        self.loss = F.mse_loss(pred, target)

        # Step
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
#         print( "training RND")
    
    def get_obs(self, obs):
        '''
        Augment observation, perhaps with generative model params
        '''
        return obs

    def get_done(self, env_done):
        '''
        figure out if we're done

        params
        ======
        env_done (bool) : done bool from the wrapped env, doesn't 
            necessarily need to be used
        '''
        return env_done

    def reset(self):
        '''
        Reset the wrapped env and the buffer
        '''
        import numpy as np
        self._time = 0
        obs = self.env.reset()
        if (self._obs_key is None):
            target = self.target_net(torch.tensor(obs).float().unsqueeze(0).to(self.device))
            data = [obs, target.detach().cpu().numpy()[0]]
        else:
            target = self.target_net(torch.tensor(obs[self._obs_key]).float().unsqueeze(0).to(self.device))
            data = [obs[self._obs_key], target.detach().cpu().numpy()[0]]
        self._buffer.add(tuple(data))
#         self._buffer.add(obs)
        
#         print( "obs1: ", obs.shape)
        obs = self.encode_obs(obs)
#         print( "obs2: ", obs.shape)
#         step_skip = 10
#         if len(self._buffer) > 0 and (np.random.rand() > (1/step_skip)):
# #             for _ in range(step_skip):
# #                 print("Training VAE")
#             self._loss = self.step_vae(self.batch_size, self.steps)
            
        return obs

    def render(self):
        self.env.render()
        
    def get_rews(self, data):
        # Set phase
        self.pred_net.eval()

        # Convert to tensor
        data, target = data
        data = torch.tensor(data).float().to(self.device).unsqueeze(0)
        target = torch.tensor(target).float().to(self.device).unsqueeze(0)

        # forward model
        pred = self.pred_net(data)

        ### TODO: Add reward scaling via continuous std from RND paper
        # losses, save as class parameters to pass to info
        return F.mse_loss(pred, target).item()

    def encode_obs(self, obs):
        '''
        Used to encode the observation before putting on the buffer
        '''
        return obs
