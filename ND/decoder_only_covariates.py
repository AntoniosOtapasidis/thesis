import torch
import numpy as np

import torch.nn as nn
from torch.autograd import Variable

from torch.nn.functional import softplus

from .helpers import expand_grid, approximate_KLqp, rsample_RelaxedBernoulli

from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs

class Decoder_only_covariates(nn.Module):

    def __init__(self, output_dim, n_covariates,
                 grid_c,
                 mappings_c,
                 has_feature_level_sparsity=True,
                 penalty_type="fixed", lambda0=1.0,
                 likelihood="Gaussian",
                 p1=0.2, p2=0.2, p3=0.2, device="cpu"):
        """
        Decoder for multiple covariates (i.e. multivariate c)
        """
        super().__init__()

        self.output_dim = output_dim
        self.likelihood = likelihood
        self.has_feature_level_sparsity = has_feature_level_sparsity
        self.penalty_type = penalty_type
        self.n_covariates = n_covariates

        assert isinstance(grid_c, list), "grid_c must be a list"
        assert len(grid_c) == n_covariates

        self.grid_c = grid_c

        self.grid_c = [c.to(device) for c in grid_c]

        self.n_grid_c = [c.shape[0] for c in grid_c]

        # input -> output
        self.mappings_c = mappings_c
        for i in range(len(self.mappings_c)):
            self.mappings_c[i].to(device)

        if self.likelihood == "Gaussian":
            # feature-specific variances (for Gaussian likelihood)
            self.noise_sd = torch.nn.Parameter(-1.0 * torch.ones(1, output_dim))

        self.intercept = torch.nn.Parameter(torch.zeros(1, output_dim))

        self.Lambda_c = [
            Variable(lambda0*torch.ones(n_covariates, 1, output_dim, device=device), requires_grad=True) for _ in range(self.n_covariates)
        ]

        self.lambda0 = lambda0

        self.device = device

        # RelaxedBernoulli
        self.temperature = 1.0 * torch.ones(1, device=device)

        if self.has_feature_level_sparsity:

            # for the prior RelaxedBernoulli(logits)
            self.logits_c = probs_to_logits(p2 * torch.ones(n_covariates, output_dim).to(device), is_binary=True)

            # for the approx posterior
        self.qlogits_c = torch.nn.Parameter(3.0 * torch.ones(n_covariates, output_dim).to(device))


    def forward_c(self, c):
        if self.has_feature_level_sparsity:
            out = []
            # if self.training:
            #     w = rsample_RelaxedBernoulli(self.temperature, self.qlogits_c)
            # else:
            #     w = torch.sigmoid(self.qlogits_c)
            w = rsample_RelaxedBernoulli(self.temperature, self.qlogits_c)
            for j in range(self.n_covariates):
                out.append(w[j, :] * (self.mappings_c[j](c[:, j:(j + 1)])))
        else:
            out = [(self.mappings_c[j](c[:, j:(j + 1)])) for j in range(self.n_covariates)]

        return out

    def forward(self, z, c):
        return self.intercept + sum(self.forward_c(c))

    def loglik(self, y_pred, y_obs):

        if self.likelihood == "Gaussian":
            sigma = 1e-6 + softplus(self.noise_sd)
            p_data = Normal(loc=y_pred, scale=sigma)
            loglik = p_data.log_prob(y_obs).sum()
        elif self.likelihood == "Bernoulli":
            p_data = Bernoulli(logits=y_pred)
            loglik = p_data.log_prob(y_obs).sum()
        else:
            raise NotImplementedError("Other likelihoods not implemented")
        return loglik
        # loss = nn.MSELoss()
        # rec_loss = loss(y_pred, y_obs).sum()
        # return -rec_loss
        

    def set_temperature(self, x):
        self.temperature = x * torch.ones(1, device=self.device)

    def calculate_integrals(self):

        w_c = rsample_RelaxedBernoulli(self.temperature, self.qlogits_c)
        int_c = [
            # has shape [1, output_dim]
            w_c[j, :] * (self.mappings_c[j](self.grid_c[j]).mean(dim=0).reshape(1, self.output_dim)) for j in
            # (self.mappings_c[j](self.grid_c[j]).mean(dim=0).reshape(1, self.output_dim)) for j in
            range(self.n_covariates)
        ]

        return int_c

    def calculate_integrals_numpy(self):

        with torch.no_grad():

            int_c = np.vstack([
                # has shape [1, output_dim]
                # (w_c[j, :] * self.mappings_c[j](self.grid_c[j]).mean(dim=0).reshape(1, self.output_dim)).cpu().numpy() for j in
                (self.mappings_c[j](self.grid_c[j]).mean(dim=0).reshape(1, self.output_dim)).cpu().numpy() for j in
                range(self.n_covariates)
            ])

            return int_c


    def calculate_penalty(self):
        int_c = self.calculate_integrals()

        # penalty with fixed lambda0
        if self.penalty_type in ["fixed", "MDMM"]:
            penalty0 = torch.zeros(int_c[0].abs().mean().shape).to(self.device)

            for j in range(self.n_covariates):
                penalty0 += self.lambda0 * int_c[j].abs().mean()

        if self.penalty_type in ["BDMM", "MDMM"]:
            penalty_BDMM = torch.zeros(int_c[0].abs().mean().shape).to(self.device)

            for j in range(self.n_covariates):
                penalty_BDMM += (self.Lambda_c[j] * int_c[j]).mean()

        if self.penalty_type == "fixed":
            penalty = penalty0
        elif self.penalty_type == "BDMM":
            penalty = penalty_BDMM
        elif self.penalty_type == "MDMM":
            penalty = penalty_BDMM + penalty0
        else:
            raise ValueError("Unknown penalty type")

        return penalty, int_c

    def loss(self, y_pred, y_obs):

        penalty, int_c = self.calculate_penalty()

        neg_loglik = - self.loglik(y_pred, y_obs) + penalty

        if self.has_feature_level_sparsity:
            KL2 = approximate_KLqp(self.logits_c, self.qlogits_c)
            neg_loglik += 1.0 * KL2

        return neg_loglik, penalty, int_c

    def fraction_of_variance_explained(self, z, c, account_for_noise=False, divide_by_total_var=True, Y_error=None):

        with torch.no_grad():
            # f_c
            f_cs = self.forward_c(c)
            f_c_vars = [f_c.var(dim=0, keepdim=True) for f_c in f_cs]

            f_all_var = torch.cat(f_c_vars, dim=0)

            if Y_error is not None:
                y_err_var = Y_error.var(dim=0, keepdim=True)
                f_all_var = torch.cat(f_c_vars + [y_err_var], dim=0)

            if divide_by_total_var:

                total_var = f_all_var.sum(dim=0, keepdim=True)

                if account_for_noise:
                    total_var += self.noise_sd.reshape(-1) ** 2

                f_all_var /= total_var

            return f_all_var.t()

    def get_feature_level_sparsity_probs(self):

        with torch.no_grad():
            w_c = torch.sigmoid(self.qlogits_c)

            return w_c.t()
