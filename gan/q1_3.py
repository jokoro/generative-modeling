import argparse
import os
from utils import get_args

import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO 1.3: Implement GAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ##################################################################
    # eps = 1e-5
    # discrim_real_min = torch.min(discrim_real) - eps
    # discrim_real_max = torch.max(discrim_real) + eps
    # discrim_fake_min = torch.min(discrim_fake) - eps
    # discrim_fake_max = torch.max(discrim_fake) + eps
    # discrim_real = (discrim_real - discrim_real_min
    #                 ) / (discrim_real_max - discrim_real_min)
    # discrim_fake = (discrim_fake - discrim_fake_min
    #                 ) / (discrim_fake_max - discrim_fake_min)
    
    loss = F.binary_cross_entropy_with_logits(discrim_real, torch.ones_like(discrim_real)) + \
        F.binary_cross_entropy_with_logits(discrim_fake, torch.zeros_like(discrim_real)) 
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO 1.3: Implement GAN loss for the generator.
    ##################################################################
    # eps = 1e-5
    # discrim_fake_min = torch.min(discrim_fake) - eps
    # discrim_fake_max = torch.max(discrim_fake) + eps
    # discrim_fake = (discrim_fake - discrim_fake_min
    #                 ) / (discrim_fake_max - discrim_fake_min)

    loss = F.binary_cross_entropy_with_logits(discrim_fake, torch.ones_like(discrim_fake)) 
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
