import torch
# from BaseVAE import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class ConvVAE(nn.Module):


    def __init__(self,
                 image_channels: int,
                 latent_size: int,
                 hidden_dims: List = None,
                 channel_last=True, 
                 **kwargs):
        super(ConvVAE, self).__init__()

        self.latent_dim = latent_size
        self._channel_last = channel_last

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
            self._hidden_dims = [32, 64, 128, 256]
        else:
            self._hidden_dims = hidden_dims
            
        self._latent_size = 1024

        # Build Encoder
        for h_dim in self._hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(image_channels, out_channels=h_dim,
                              kernel_size= 4, 
                              stride= 2, 
#                               padding  = 1
                              ),
#                     nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            image_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Sequential(
                    nn.Linear(self._latent_size, self.latent_dim),
#                     nn.BatchNorm2d(h_dim),
                    nn.Tanh())
#         self.fc_var = nn.Linear(self._latent_size, self.latent_dim)
        self.fc_var = nn.Sequential(
                    nn.Linear(self._latent_size, self.latent_dim),
#                     nn.BatchNorm2d(h_dim),
                    nn.Softplus())

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(self.latent_dim, self._latent_size)
        
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=5,
                                       stride = 2,
                                        padding=1,
                                        output_padding=1
                                       ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)
        
        print ("hidden_dims[-1]: ", hidden_dims[-1])
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               out_channels= 4,
                                               kernel_size=6,
                                               stride=2,
#                                                 padding=1,
#                                                 output_padding=1
                                               ),
#                             nn.BatchNorm2d(32),
#                             nn.LeakyReLU(),
#                             nn.Conv2d(hidden_dims[-1], out_channels= 4,
#                                       kernel_size= 3, padding= 1),
#                             nn.Tanh()
                            )
        
    def to(self, device):

#         for i in range(len(self.fc1)):
        self.fc_mu.to(device)
#         for i in range(len(self.fc2)):
        self.fc_var.to(device)
        self.encoder.to(device)
#         for i in range(len(self.c0)):
        self.decoder_input.to(device)
        self.decoder.to(device)
#         for i in range(len(self.d0)):
        self.final_layer.to(device)
        return self

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        if (self._channel_last):
            ### Need to change image to be channel first
            input = input.permute(0, 3,2,1)
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        z = self.reparameterize(mu, log_var)

        return [mu, log_var, z]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(z.size(0), self._hidden_dims[-1], 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
#         if (self._channel_last):
#             ### Need to change image to be channel first
#             result = result.permute(0, 3,2,1)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
#         print ("input.shape: ", input.shape)
        mu, log_var, z = self.encode(input)
#         z = self.reparameterize(mu, log_var)
        return  [self.decode(z), mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
    