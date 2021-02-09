import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

class ICMWrapper(gym.Env):
    
    def __init__(self, env, network, device=0, obs_key=None, hist_size=3000,
                 reward_func=None, **kwargs):
#         from surprise.envs.vizdoom.networks import VAEConv
#         from surprise.envs.vizdoom.buffer import VAEBuffer
        from surprise.envs.vizdoom.buffer import SimpleBuffer
        from surprise.envs.vizdoom.networks import MLP, VizdoomFeaturizer
#         from rlkit.torch.networks import Mlp
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
        
#         RND stuff#         from rlkit.torch.networks import Mlp
        print ("hist_size: ", hist_size)
        self._buffer = SimpleBuffer(device=self.device, size=hist_size)
        if (kwargs["network_type"] == "flat"):
            self.forward_network = MLP([self.observation_space.low.size + self.action_space.n, 64, self.observation_space.low.size]).to(self.device)
            self.inverse_network = MLP([self.observation_space.low.size * 2, 32, 16, self.action_space.n], log_softmax=True).to(self.device)
            self.featurizer = MLP([self.observation_space.low.size, 64, self.observation_space.low.size]).to(self.device)
        else:
            self.forward_network = MLP([self.observation_space.low.size + self.action_space.n, 64, 64, self.observation_space.low.size]).to(self.device)
            self.inverse_network = MLP([self.observation_space.low.size * 2, 32, 16, 8, self.action_space.n], log_softmax=True).to(self.device)
            self.featurizer = VizdoomFeaturizer(self.observation_space.low.size).to(self.device)
            
        self.optimizer_forward = optim.Adam(self.forward_network.parameters(), lr=1e-4)
        self.optimizer_inverse = optim.Adam(list(self.inverse_network.parameters()) + 
                                    list(self.featurizer.parameters()), lr=1e-4)
        self.network = self.forward_network
        self.step_freq = 16
        self.inverse_loss = torch.zeros(1)
        self.forward_loss = torch.zeros(1)
        self.loss = torch.zeros(1)

    def step(self, action):
        
        if (self._obs_key is None):
            data = [np.array(self._prev_obs), np.eye(self.action_space.n)[action], None]
        else:
            data = [np.array(self._prev_obs[self._obs_key]), np.eye(self.action_space.n)[action], None]
        # Take Action
        obs, rew, done, info = self.env.step(action)
        # Finish off (s,a,s') tuplet and add to buffer
        if (self._obs_key is None):
            data[2] = np.array(obs)
        else:
            data[2] = np.array(obs[self._obs_key])
#         print ("data: ", data)

        self._buffer.add(tuple(data))

        # Get wrapper outputs
#         print ("data:", data)
        
                # Update network
        if self._time % self.step_freq == 0:
            self.step_model()
            
        obs = self.encode_obs(obs)
        info["task_reward"] = rew
        info.update(self.get_info())
        done = self.get_done(done)
        
        self._time = self._time + 1
        
        rew = self.get_rews(data)
        if (self._reward_func == "add"):
            rew = rew + info["task_reward"]
#             print("Add reward: ", rew)
        self._prev_obs = obs
        return obs, rew, done, info
    
    def step_model(self, batch_size=64):
        # Set phase
        self.forward_network.train()
        self.inverse_network.train()
        self.featurizer.train()
        # Get data (s,a,s')
        s, a, sp = self._buffer.sample(batch_size)
        s = torch.tensor(s).to(self.device).float()
        a = torch.tensor(a).to(self.device).float()
        sp = torch.tensor(sp).to(self.device).float()
#         print ("s.shape: ", s, a, sp)

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
    
    def get_obs(self, obs):
        '''
        Augment observation, perhaps with generative model params
        '''
        return obs
    
    def get_info(self):
        return {
                'inverse_loss': self.inverse_loss.item(),
                'forward_loss': self.forward_loss.item(),
                'loss': self.loss.item(),
                }

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
        self._prev_obs = self.env.reset()
        ### Can't add data to buffer with action.
#         target = self.target_net(torch.tensor(obs).float().unsqueeze(0).to(self.device))
#         data = [obs, target.cpu().detach().numpy()[0]]
#         self._buffer.add(tuple(data))
#         self._buffer.add(obs)
        
#         print( "obs1: ", obs.shape)
        self._prev_obs = self.encode_obs(self._prev_obs)
#         print( "obs2: ", obs.shape)
#         step_skip = 10
#         if len(self._buffer) > 0 and (np.random.rand() > (1/step_skip)):
# #             for _ in range(step_skip):
# #                 print("Training VAE")
#             self._loss = self.step_vae(self.batch_size, self.steps)
            
        return self._prev_obs

    def render(self):
        self.env.render()
        
    def get_rews(self, sas):
        # Set phase
        self.featurizer.eval()
        self.forward_network.eval()

        # Unpack
        s,a,sp = sas

        # Convert to tensor
        s = torch.tensor(s).float().to(self.device).unsqueeze(0)
        a = torch.tensor(a).float().to(self.device).unsqueeze(0)
        sp = torch.tensor(sp).float().to(self.device).unsqueeze(0)

        # featurize
        fs, fsp = self.featurizer(s), self.featurizer(sp)

        # forward model
        state_pred = self.forward_network(torch.cat([fs, a], dim=1))

        # losses, save as class parameters to pass to info
        return F.mse_loss(state_pred, fsp).item()

    def encode_obs(self, obs):
        '''
        Used to encode the observation before putting on the buffer
        '''
        return obs
