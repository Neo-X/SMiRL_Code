import gym
import torch

class TrainingWrapper(gym.Env):
    def __init__(self, env, eval, network, device=0, **kwargs):
        from torch import optim
        from gym import spaces
        '''
        params
        ======
        env (gym.Env) : environment to wrap

        '''
        self.device=device
        self.env = env
        
        self.network = network
        self.eval = eval
        self.vae_buffer = VAEBuffer(device=device, size=10000)
        self.optimizer = optim.Adam(self.vae.parameters(), lr=1e-4)
        self.batch_size = 64
        self.steps = 100
        self.vae_loss = 0

        # Gym spaces
        self.action_space = env.action_space
        self.observation_space = self.observation_space = spaces.Box(low=-2, high=2, shape=(kwargs["latent_size"],))

    def step(self, action):
        # Take Action
        obs, rew, done, info = self.env.step(action)
        self.vae_buffer.add(obs)

        # Get wrapper outputs
        # print ("obs:", obs)
        obs, info_ = self.encode_obs(obs)
        info_["_loss"] = self._loss
        info.update(info_)
        done = self.get_done(done)

        return obs, rew, done, info
    
    def step_model(self, batch_size, n):
        # Don't step if eval
        if self.eval:
            return 0

        self.vae.train()
        train_loss = 0
        steps = min(n, len(self._buffer)//batch_size + 1)
        for i in range(steps):
            data = self.vae_buffer.sample(batch_size)
            
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.network(data)
            loss = self.loss_fn(recon_batch, data, mu, logvar)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
        return train_loss / steps / batch_size
    

        # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_fn(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 130), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + .1 * KLD

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
        self._buffer.add(obs)
#         print( "obs1: ", obs.shape)
        obs, _ = self.encode_obs(obs)
#         print( "obs2: ", obs.shape)
        step_skip = 10
        if len(self._buffer) > 0 and (np.random.rand() > (1/step_skip)):
#             for _ in range(step_skip):
#                 print("Training VAE")
            self._loss = self.step_model(self.batch_size, self.steps)
            
        return obs

    def render(self):
        self.env.render()
        
    def encode_obs(self, obs):
        '''
        Encodes a (lowres) buffer observation
        '''
        return obs
