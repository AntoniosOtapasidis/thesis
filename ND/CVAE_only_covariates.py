import torch
import torch.nn as nn
import torch
import numpy as np
from loguru import logger

from math import ceil

from .helpers import KL_standard_normal

class CVAE_only_covariates(nn.Module):
    """
    CVAE with Neural Decomposition (for multiple covariates) as part of the decoder
    """

    def __init__(self, encoder, decoder, lr, device="cpu"):
        super().__init__()

        self.encoder = encoder
        self.encoder.to(device)

        self.decoder = decoder

        self.output_dim = self.decoder.output_dim

        # optimizer for NN pars and likelihood noise
        all_params = list(self.parameters())
        for m in self.decoder.mappings_c:
            all_params.extend(list(m.parameters()))

        self.optimizer = torch.optim.Adam(all_params, lr=lr)
    
        self.device = device

        self.to(device)


    def forward(self, data_subset, beta=1.0, batch_scale=1.0, device="cpu"):
        # we assume data_subset containts three elements
        if len(data_subset) == 2:
            Y, c = data_subset
        elif len(data_subset) == 3:
            Y, c, _ = data_subset

        # decoding step
        y_pred = self.decoder.forward(c, c)
        decoder_loss, penalty, intc = self.decoder.loss(y_pred, Y)

        # no KL(q(z) | p(z)) term because z fixed
        total_loss = decoder_loss

        return total_loss, intc


    def calculate_test_loglik(self, Y, c):
        """
        maps (Y, x) to z and calculates p(y* | x, z_mu)
        :param Y:
        :param c:
        :return:
        """
        Y_pred = self.decoder.forward(c)

        return self.decoder.loglik(Y_pred, Y)


    def optimize(self, data_loader, augmented_lagrangian_lr, n_iter=50000, logging_freq=20, logging_freq_int=100, batch_scale=1.0, account_for_noise=True, temperature_start=4.0, temperature_end=0.2, lambda_start=None, lambda_end=None, verbose=True):

        # sample size
        N = len(data_loader.dataset)

        # number of iterations = (numer of epochs) * (number of iters per epoch)
        n_epochs = ceil(n_iter / len(data_loader))
        if verbose:
            logger.info(f"Fitting Neural Decomposition.\n\tData set size {N}. # iterations = {n_iter} (i.e. # epochs <= {n_epochs})\n")

        loss_values = np.zeros(ceil(n_iter // logging_freq))

        if self.decoder.has_feature_level_sparsity:
            temperature_grid = torch.linspace(temperature_start, temperature_end, steps=n_iter // 10, device=self.device)

        if lambda_start is None:
            lambda_start = self.decoder.lambda0
            lambda_end = self.decoder.lambda0
        lambda_grid = torch.linspace(lambda_start, lambda_end, steps=n_iter // 10, device=self.device)

        # get shapes for integrals
        _int_c = self.decoder.calculate_integrals_numpy()
        # log the integral values
        n_logging_steps = ceil(n_iter / logging_freq_int)
        int_c_values = np.zeros([n_logging_steps, _int_c.shape[0], self.output_dim])

        iteration = 0
        for epoch in range(n_epochs):

            for batch_idx, data_subset in enumerate(data_loader):

                if iteration >= n_iter:
                    break

                loss, int_c = self.forward(data_subset, beta=1.0, batch_scale=batch_scale, device=self.device)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.decoder.has_feature_level_sparsity:
                    self.decoder.set_temperature(temperature_grid[iteration // 10])
                self.decoder.lambda0 = lambda_grid[iteration // 10]

                # update for BDMM
                with torch.no_grad():
                    for j in range(self.decoder.n_covariates):
                        self.decoder.Lambda_c[j] += augmented_lagrangian_lr * int_c[j]

                # logging for the loss function
                if iteration % logging_freq == 0:
                    index = iteration // logging_freq
                    loss_values[index] = loss.item()

                # logging for integral constraints
                if iteration % logging_freq_int == 0:
                    int_c = self.decoder.calculate_integrals_numpy()

                    index = iteration // logging_freq_int
                    int_c_values[index, :] = int_c

                if verbose and iteration % 500 == 0:
                    logger.info(f"\tIter {iteration:5}.\tTotal loss {loss.item():.3f}")

                iteration += 1

        # collect all integral values into one array
        integrals = np.hstack([int_c_values]).reshape(n_iter // logging_freq_int, -1).T

        return loss_values, integrals
