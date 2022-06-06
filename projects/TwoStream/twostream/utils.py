from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELoss(nn.Module):
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    """
    def __init__(self, kld_weight=0.01):
        super().__init__()
        self.kld_weight = kld_weight

    def forward(self, recons, input, mu, log_var) -> dict:
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + self.kld_weight * kld_loss
        loss_vis = {
            'recon_loss' : recons_loss.detach(),
            'KLD_loss' : self.kld_weight * kld_loss.detach()
        }
        return loss, loss_vis
