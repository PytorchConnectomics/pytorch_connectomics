from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')
from abc import abstractmethod

import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from connectomics.model.utils import *


class VAEBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class VAE(VAEBase):
    """Variational autoencoder with convolutional layers. The input images should be square.
    """
    def __init__(self, img_channels: int, latent_dim: int, hidden_dims: List = [32, 64, 128, 256, 512],
                 width: int = 64, act_mode: str = 'relu', norm_mode: str = 'bn', **kwargs) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = copy.deepcopy(hidden_dims)
        in_channels = img_channels
        sq_sz = self.calc_sz(width)

        modules = [] # build encoder
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, 
                          stride=2, padding=1),
                get_norm_2d(norm_mode, h_dim),
                get_activation(act_mode))
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*sq_sz, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*sq_sz, latent_dim)

        modules = [] # build decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*sq_sz)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[i], hidden_dims[i+1], kernel_size=3,
                    stride=2, padding=1, output_padding=1),
                get_norm_2d(norm_mode, hidden_dims[i+1]),
                get_activation(act_mode))
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1], hidden_dims[-1], kernel_size=3,
                stride=2, padding=1, output_padding=1),
            get_norm_2d(norm_mode, hidden_dims[-1]),
            get_activation(act_mode),
            nn.Conv2d(hidden_dims[-1], out_channels=img_channels,
                      kernel_size=3, padding=1),
            nn.Tanh()) # inputs are normalized to [-1, 1]

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], self.sz, self.sz)
        result = self.decoder(result)
        result = self.final_layer(result)
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
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

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
        z = torch.randn(num_samples, self.latent_dim)
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

    def calc_sz(self, width):
        down_sample = 2 ** len(self.hidden_dims)
        assert width % down_sample == 0, "The input width/height " + \
            f"{width} is not divisible by {2**len(self.hidden_dims)}!"
        self.sz = width // down_sample
        sq_sz = self.sz ** 2
        return sq_sz
