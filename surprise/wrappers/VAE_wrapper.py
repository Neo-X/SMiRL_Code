import gym
import torch
import torch.nn.functional as F
import torch.optim as optim

class VAEWrapper(gym.Env):
    
    def __init__(self, env, eval, network=None, 
                 device=0, steps=100, 
                 step_skip=1, hist_size=10000,
                 steps_per_train_delay=500,
                 dtype="float32",
                 **kwargs):
        from surprise.envs.vizdoom.networks import VAEConv, VAE2
        from surprise.envs.vizdoom.VAEConv import ConvVAE as VAEConv2
        from surprise.envs.vizdoom.buffer import VAEBuffer
        from torch import optim
        from gym import spaces
        '''
        params
        ======
        env (gym.Env) : environment to wrap

        '''
        self._reconstructions=None
        self._count=0
        self._kl_term = 0.1
        if ("kl_term" in kwargs):
            self._kl_term = kwargs["kl_term"]
        self._steps_per_train_delay = steps_per_train_delay 
        self.device=device
        self.env = env
        self.step_skip = step_skip
        ### We don't want the eval environment to change or update the model.
        self._eval = eval
        if "obs_label" in kwargs:
            self._obs_label = kwargs["obs_label"]
        else:
            self._obs_label = None
        if "obs_key" in kwargs:
            self._obs_key = kwargs["obs_key"]
        else:
            self._obs_key = None
        if "perform_pretraining" in kwargs:
            self._perform_pretraining = kwargs["perform_pretraining"]
        else:
            self._perform_pretraining = False
        self._hist_size = hist_size
        
#         if network is None:
#             self.network = VAE().to(device)
        print("hist_size: ", self._hist_size )
        if network is None:
            if (kwargs["conv_net"] == "VAE2"):
                self.network = VAEConv2(input_shape=env.observation_space.low.shape, **kwargs).to(device)
            elif (kwargs["conv_net"]):
                self.network = VAEConv(input_shape=env.observation_space.low.shape, **kwargs).to(device)
            else:
                self.network = VAE2(device=device, input_shape=env.observation_space.low.shape, **kwargs).to(device)
            self._buffer = VAEBuffer(device=device, size=self._hist_size, dtype=dtype)
            for param in list(self.network.parameters()):
                print ("VAE parameters size: ", param.shape)
            if ("learning_rate" in kwargs):
                self.optimizer = optim.Adam(self.network.parameters(), lr=kwargs["learning_rate"])
            else:
                self.optimizer = optim.Adam(self.network.parameters(), lr=0.0005)
            print ("self.network.parameters(): ", self.network.parameters())
        else:
            self.network = network
        self.batch_size = 32
        self.steps = steps
        self._loss = 0

        # Gym spaces
        self.action_space = env.action_space
        self.observation_space_old = env.observation_space
        if (self._obs_label is None):
            self.observation_space = self.observation_space = spaces.Box(low=-2, high=2, shape=(kwargs["latent_size"],))
        else:
            self.observation_space = env.observation_space

    def step(self, action):
        import numpy as np
        # Take Action
        obs, rew, done, info = self.env.step(action)
        self._count = self._count + 1
#         print ("self._count: ", self._count)

#         print ("obs: ", np.mean(obs[self._obs_key]), np.std(obs[self._obs_key]))        
        if (not self._eval):
            if (self._obs_key is None):
                self._buffer.add(obs)
            else:
                self._buffer.add(obs[self._obs_key])
        else:
            if (self._obs_key is None):
                vae_reconstruction, mu, logvar = self.network(torch.tensor(obs).float().unsqueeze(0).to(self.device))
                info["vae_reconstruction"] = np.array(vae_reconstruction.detach().cpu().numpy()[0] * 255, dtype="uint8")
#                 print ("vae_reconstruction: ", np.mean(info["vae_reconstruction"]), np.std(info["vae_reconstruction"])) 
            else:
                vae_reconstruction, mu, logvar = self.network(torch.tensor(obs[self._obs_key]).float().unsqueeze(0).to(self.device))
                info["vae_reconstruction"] = np.array(vae_reconstruction.detach().cpu().numpy()[0] * 255, dtype="uint8")
                ### Need to change image to be channel first
#                 info["vae_reconstruction"] = info["vae_reconstruction"].permute(0, 2,3,1)
#                 print ("vae_reconstruction before: ", info["vae_reconstruction"].shape, np.mean(info["vae_reconstruction"]), np.std(info["vae_reconstruction"]))
                info["vae_reconstruction"] = np.moveaxis(info["vae_reconstruction"][:3], 0, -1)
#                 print ("vae_reconstruction: ", info["vae_reconstruction"].shape, np.mean(info["vae_reconstruction"]), np.std(info["vae_reconstruction"]))
                        
#                 print (vae_reconstruction)

        if ((not self._eval) and (len(self._buffer) >= self._hist_size) and 
#             (np.random.rand() > (1/self.step_skip)) and 
             (self._count > self._steps_per_train_delay)
            ):
            self._loss, _ = self.step_vae(self.batch_size, self.steps)
            self._count=0
        # Get wrapper outputs
#         print ("obs:", obs.shape)
        obs = self.encode_obs(obs)
        if (not self._eval and self._loss is not None):
            info["vae_loss"] = self._loss
        
        return obs, rew, done, info
    
    def step_vae(self, batch_size, n):
        # Don't step if eval
        if self._eval:
            return 0

        self.network.train()
        train_loss = 0
#         steps = min(n, len(self._buffer)//batch_size + 1)
        steps = n
#         print ("len(self._buffer): ", len(self._buffer), " n", n)
        if self._perform_pretraining:
            ## Kind of ugly hack to put in some pretraining
            if len(self._buffer) >= self._hist_size:
                steps = self._perform_pretraining
                self._perform_pretraining = False
                print ("Learning steps togo: ", steps, "self._buffer size: ", len(self._buffer))
            else:
#                 steps = 0
                return 0, None
        for i in range(steps):
            data, true_data = self._buffer.sample(batch_size)
            
            self.optimizer.zero_grad()
#             print ("recon data: ", torch.mean(data), torch.std(data))
            recon_batch, mu, logvar = self.network(data)
#             print ("recon_batch: ", torch.mean(recon_batch), torch.std(recon_batch))
#             print ("recon_batch: ", recon_batch.shape)
#             recon_batch = recon_batch
            if (True):
                ### Need to change image to be channel first
                data = data.permute(0, 3,2,1)
            loss = self.loss_fn(recon_batch, data, mu, logvar)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
#             print ("train_loss: ", loss.item())
        return train_loss / steps / batch_size, recon_batch
    

        # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_fn(self, recon_x, x, mu, logvar):
#         BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.observation_space_old.low.size), reduction='sum')
        BCE = F.mse_loss(recon_x, x)
#         BCE = 0
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
#         KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + (self._kl_term * KLD)

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
        obs = self.env.reset()
        if not self._eval:
            if (self._obs_key is None):
                self._buffer.add(obs)
            else:
                self._buffer.add(obs[self._obs_key])
#         print( "obs1: ", obs["observation"].shape)
        obs = self.encode_obs(obs)
#         print( "obs2: ", obs)
            
        return obs

    def render(self):
        self.env.render()

    def encode_obs(self, obs):
        '''
        Used to encode the observation before putting in the buffer
        '''
        self.network.eval()
        if (self._obs_key is None):
            obs = torch.tensor(obs).float().unsqueeze(0).to(self.device)
            obs_ = obs
        else:
            obs_ = torch.tensor(obs[self._obs_key]).float().unsqueeze(0).to(self.device)
        z, mu, logvar = self.network.encode(obs_)
        
        if self._obs_label is None:
            return mu.detach().cpu().numpy()[0]
        else:
            obs[self._obs_label] = mu.detach().cpu().numpy()[0]
            return obs
