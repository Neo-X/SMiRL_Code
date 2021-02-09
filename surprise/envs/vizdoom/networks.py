import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class VizdoomQF(nn.Module):
    def __init__(self, actions=2, 
                 style=None, 
                 size=None,
                 type=None, 
                 input_shape=[4,48,64]):
        super().__init__()
        self._input_shape=input_shape
        self.conv = nn.Sequential(nn.Conv2d(4, 64, 5),
                                  nn.MaxPool2d(2, 2),
                                  nn.LeakyReLU(True),

                                  nn.Conv2d(64, 32, 3),
                                  nn.MaxPool2d(2, 2),
                                  nn.LeakyReLU(True),

                                  nn.Conv2d(32, 8, 3))

        # VAE - using encoded latents as buffer obs
        # RGB - 3 channels obs, still augmented state space
        # time - only adding time to state obs
        # else - normal (surprise) augmented state space
        if style == 'vae':
            input_dim = 809
        elif style == 'rgb':
            input_dim = 1549
        elif style == 'time':
            input_dim = 769
        elif size is not None:
            input_dim = size
        else:
            input_dim = 1029

        self.fc = nn.Sequential(nn.Linear(input_dim, 128),
                                nn.ReLU(True),
                                nn.Linear(128, 64),
                                nn.ReLU(True),
                                nn.Linear(64, actions))

    def forward(self, obs):
        import numpy as np
        batch_size = obs.shape[0]
#         print("obs.shape: ", obs.shape)
        # Extract img and parameter info
#         print ("np.prod(self._input_shape): ", np.prod(self._input_shape))
        img_obs, params_obs = obs[:, :np.prod(self._input_shape)], obs[:, np.prod(self._input_shape):]

        # Reshape img and add channel dim
        img_obs = img_obs.view(batch_size, *self._input_shape)

        # Encode img features
        img_feats = self.conv(img_obs).view(batch_size, -1)

        # Concat and return fc output
        concat = torch.cat([img_feats, params_obs], dim=1)
        return self.fc(concat)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(130, 512)
        self.fc11 = nn.Linear(512, 256)
        self.fc21 = nn.Linear(256, 20)
        self.fc22 = nn.Linear(256, 20)
        self.fc30 = nn.Linear(20, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 130)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc11(h1))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc30(z))
        h3 = F.relu(self.fc3(h3))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 130))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class UnFlatten(nn.Module):
#     
#     def __init__(self, size=768):
#         super(nn.Module, self).__init__()
#         self._size=size
        
    def forward(self, input, size=768):
        return input.view(input.size(0), size, 1, 1)
    
class VAEConv(nn.Module):
    def __init__(self, 
                 image_channels=3, 
                 input_shape=(48,64,3),
                 h_dim=768, 
                 latent_size=32, 
                 channel_last=True, 
                 **kwargs):
        super(VAEConv, self).__init__()
        self._channel_last=channel_last
        self._device = device
        self.c0 = []
        self.c0.append(nn.Conv2d(image_channels, 16, kernel_size=4, stride=2, padding=1))
        self.c0.append(nn.LeakyReLU())
        self.c0.append(nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1))
        self.c0.append(nn.LeakyReLU())
        self.c0.append(nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1))
        self.c0.append(nn.LeakyReLU())
#             nn.Conv2d(128, 256, kernel_size=4, stride=2),
#             nn.ReLU(),
        self.c0.append(Flatten())
        
        self.fc1 = [nn.Linear(h_dim, latent_size)]
        self.fc1.append(nn.Tanh())
        
        self.fc2 = [nn.Linear(h_dim, latent_size)]
        self.fc2.append(nn.Softplus())
        
        self.fc3 = nn.Linear(latent_size, h_dim)
        
        self.d0 = []
        self.d0.append(UnFlatten())
#             nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
#             nn.ReLU(),
        self.d0.append(nn.ConvTranspose2d(h_dim, 16, kernel_size=[8,6], stride=2, padding=0 ))
        self.d0.append(nn.LeakyReLU())
        self.d0.append(nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1))
        self.d0.append(nn.LeakyReLU())
        self.d0.append(nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1))
        self.d0.append(nn.LeakyReLU())
        self.d0.append(nn.ConvTranspose2d(16, 4, kernel_size=4, stride=2, padding=1))
        self.d0.append(nn.ReLU())
#         self.d0.append(nn.ConvTranspose2d(1, 6, kernel_size=4, stride=2))
#         self.d0.append(nn.Sigmoid())
        
        print ("self.decoder: ", self.decoder)
#                 hidden_dims = [16, 32, 64, 128, 256]
#         self.c0 = []
#         for h_dim in hidden_dims:
#             self.c0.append(nn.Conv2d(image_channels, h_dim, kernel_size=3, stride=2, padding=1))
#             self.c0.append(nn.BatchNorm2d(h_dim))
#             self.c0.append(nn.LeakyReLU())
#             image_channels = h_dim
# #         self.c0.append(nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1))
# #         self.c0.append(nn.ReLU())
# #         self.c0.append(nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1))
# #         self.c0.append(nn.ReLU())
# #             nn.Conv2d(128, 256, kernel_size=4, stride=2),
# #             nn.ReLU(),
#         self.c0.append(Flatten())
#         
#         self.fc1 = [nn.Linear(hidden_dims[-1]*4, latent_size)]
# #         self.fc1.append(nn.Tanh())
#         
#         self.fc2 = [nn.Linear(hidden_dims[-1]*4, latent_size)]
#         self.fc2.append(nn.Softplus())
#         
#         self.fc3 = nn.Linear(latent_size, hidden_dims[-1])
#         
#         self.d0 = []
#         self.d0.append(UnFlatten())
# #             nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
# #             nn.ReLU(),
#         hidden_dims.reverse()
#         for i in range(len(hidden_dims) - 1):
#             self.d0.append(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], 
#                                               kernel_size=3, stride=2, padding=1, output_padding=1 ))
#             self.d0.append(nn.BatchNorm2d(hidden_dims[i + 1]))
#             self.d0.append(nn.LeakyReLU())
# 
#         self.final_layer = nn.Sequential(
#                     nn.ConvTranspose2d(hidden_dims[-1],
#                                        hidden_dims[-1],
#                                        kernel_size=3,
#                                        stride=2,
#                                        padding=1,
#                                        output_padding=1),
#                     nn.BatchNorm2d(hidden_dims[-1]),
#                     nn.LeakyReLU(),
#                     nn.Conv2d(hidden_dims[-1], out_channels= 3,
#                               kernel_size= image_channels, padding= 1),
#                     nn.Tanh())
# #         self.d0.append(nn.ConvTranspose2d(1, 6, kernel_size=4, stride=2))
# #         self.d0.append(nn.Sigmoid())
#         
#         print ("self.decoder: ", self.decoder)
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.encoder_mu(h), self.encoder_var(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def encoder_var(self, x):
        
        for i in range(len(self.fc2)):
            x = self.fc2[i](x)
#             print ("encoder c shape: ", x.shape)
        return x
    
    def encoder_mu(self, x):
        
        for i in range(len(self.fc1)):
            x = self.fc1[i](x)
#             print ("encoder c shape: ", x.shape)
        return x
    
    def encoder(self, x):
        
        for i in range(len(self.c0)):
            x = self.c0[i](x)
#             print ("encoder c shape: ", x.shape)
        return x
    
    def decoder(self, x):
        
        for i in range(len(self.d0)):
            x = self.d0[i](x)
#             print ("decoder c shape: ", x.shape)
        return x
            
    def encode(self, x):
#         print("self._channel_last: ", self._channel_last)
        if (self._channel_last):
            ### Need to change image to be channel first
            x = x.permute(0, 3,2,1)
        
#         print ("x: ", x.shape)
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
#         print ("z shape: ", z.shape)
        z = self.fc3(z)
#         print ("z shape: ", z.shape)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

    def to(self, device):

        for i in range(len(self.fc1)):
            self.fc1[i].to(device)
        for i in range(len(self.fc2)):
            self.fc2[i].to(device)
        self.fc3.to(device)
        for i in range(len(self.c0)):
            self.c0[i].to(device)
        for i in range(len(self.d0)):
            self.d0[i].to(device)

        return self

class VAE2(nn.Module):
    
    def __init__(self, device, input_shape, **kwargs):
        super(VAE2, self).__init__()
        self._device = device
        self.fc1 = nn.Linear(input_shape[0], 128)
        self.fc11 = nn.Linear(128, 64)
        self.fc21 = nn.Linear(64, kwargs["latent_size"])
        self.fc22 = nn.Linear(64, kwargs["latent_size"])
        
        self.fc30 = nn.Linear(kwargs["latent_size"], 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, input_shape[0])

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self._device)
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc21(h), self.fc22(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
#         print("self._channel_last: ", self._channel_last)
        h1 = F.relu(self.fc1(x))
        h = F.relu(self.fc11(h1))
#         return self.fc21(h1), self.fc22(h1)
#         h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        h3 = F.relu(self.fc30(z))
        h3 = F.relu(self.fc3(h3))
        return torch.sigmoid(self.fc4(h3))
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

class MLP(nn.Module):
    def __init__(self, dims, log_softmax=False):
        '''
        params
        ======
        dims (list[int]) : list of dimensions of layers

        '''

        super().__init__()

        # Add layers to the network
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU(True))

        # Remove the last ReLU
        layers = layers[:-1]

        self.network = nn.Sequential(*layers)

        self.log_softmax = log_softmax

    def forward(self, x):
        if self.log_softmax:
            return F.log_softmax(self.network(x), dim=1)
        else:
            return self.network(x)


class VizdoomFeaturizer(nn.Module):
    def __init__(self, dim, qf=False, channel_last=True, type=None):
        super().__init__()
        self._channel_last = channel_last
        self.conv = nn.Sequential(nn.Conv2d(4, 64, 5),
                                  nn.MaxPool2d(2, 2),
                                  nn.ReLU(True),

                                  nn.Conv2d(64, 32, 3),
                                  nn.MaxPool2d(2, 2),
                                  nn.ReLU(True),

                                  nn.Conv2d(32, 8, 3))

        # Raw
        self.fc = nn.Sequential(nn.Linear(768, 128),
                                nn.ReLU(True),
                                nn.Linear(128, 64),
                                nn.ReLU(True),
                                nn.Linear(64, dim))

        self.qf = qf

    def forward(self, obs):
        batch_size = obs.shape[0]
#         print("obs.shape: ", obs.shape)
        # Extract img and parameter info
        img_obs, params_obs = obs[:, :4*48*64], obs[:, 4*48*64:]

        # Reshape img and add channel dim
        img_obs = img_obs.view(batch_size, 4, 48, 64)
        
#         if (self._channel_last):
#             ### Need to change image to be channel first
#             obs = obs.permute(0, 3,2,1)
#         print ("obs.shape: ", obs.shape)
        # If net is being used as qf, we are
        # accepting flattened inputs
        if self.qf:
            obs = obs.view(batch_size, 4, 48, 64)

        feats = self.conv(img_obs).view(batch_size, -1)
        return self.fc(feats)

class VizdoomForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_translate = nn.Sequential(
            nn.Conv2d(5,16,5),
            nn.ReLU(),
            nn.Conv2d(16,16,3),
            nn.ReLU(),
            nn.ConvTranspose2d(16,16,3),
            nn.ReLU(),
            nn.ConvTranspose2d(16,4,5))

    def forward(self, obs, action):
        batch_size = action.shape[0]
        action = torch.ones(batch_size, 1, 48, 64).to(1) * action[:,0].view(batch_size, 1, 1, 1)
        x = torch.cat([action, obs], dim=1)
        return self.image_translate(x)


