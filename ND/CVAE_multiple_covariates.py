import torch
import torch.nn as nn
import torch
import numpy as np
from loguru import logger
import wandb
import torch.nn.functional as F
from math import ceil

from .helpers import KL_standard_normal
import torch

def corrz(embedding_a: torch.Tensor, embedding_b: torch.Tensor) -> torch.Tensor:
    """
    Computes a decorrelation loss between two embedding vectors by penalizing cross-correlations.
    
    Args:
        embedding_a: Tensor of shape (batch_size, dim_a)
        embedding_b: Tensor of shape (batch_size, dim_b)

    Returns:
        decorrelation loss (scalar tensor)
    """
    # Center the embeddings
    a_centered = embedding_a - embedding_a.mean(dim=0, keepdim=True)
    b_centered = embedding_b - embedding_b.mean(dim=0, keepdim=True)

    # Compute standard deviations
    a_std = a_centered.std(dim=0, unbiased=True).clamp(min=1e-8)  # shape: (dim_a,)
    b_std = b_centered.std(dim=0, unbiased=True).clamp(min=1e-8)  # shape: (dim_b,)

    # Normalize to unit variance
    a_normalized = a_centered / a_std
    b_normalized = b_centered / b_std

    # Compute cross-correlation matrix
    batch_size = embedding_a.size(0)
    corr_ab = (a_normalized.T @ b_normalized) / (batch_size - 1)  # shape: (dim_a, dim_b)

    # Frobenius norm of the cross-correlation matrix
    loss = torch.norm(corr_ab, p='fro')**2

    return loss


def decorrelation_loss(embedding_a: torch.Tensor, embedding_b: torch.Tensor) -> torch.Tensor:
    """
    Computes a decorrelation loss between two embedding vectors of shape (batch_size, dim_a) and (batch_size, dim_b)
    by penalizing cross-covariance.

    Args:
        embedding_a: Tensor of shape (batch_size, dim_a)
        embedding_b: Tensor of shape (batch_size, dim_b)

    Returns:
        decorrelation loss (scalar tensor)
    """
    # Center the embeddings
    a_centered = embedding_a - embedding_a.mean(dim=0, keepdim=True)
    b_centered = embedding_b - embedding_b.mean(dim=0, keepdim=True)

    # Compute cross-covariance matrix
    batch_size = embedding_a.size(0)
    cov_ab = (a_centered.T @ b_centered) / (batch_size - 1)  # shape: (dim_a, dim_b)

    # Compute Frobenius norm of the cross-covariance matrix
    loss = torch.norm(cov_ab, p='fro')**2  # sum of squared covariances

    return loss


def ortho_loss(z1, zs, norm=True, temp=0.1):
    z1 = F.normalize(z1, dim=-1)
    zs = F.normalize(zs, dim=-1)
    if norm:
        return torch.norm(torch.matmul(z1.T, zs)) # yes (type1)
    else:
        raise NotImplementedError('Please set norm=True')


class CVAE_multiple_covariates(nn.Module):
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
        for m in self.decoder.mappings_c + self.decoder.mappings_cz:
            all_params.extend(list(m.parameters()))

        self.optimizer = torch.optim.Adam(all_params, lr=lr)
    
        self.device = device

        self.to(device)


    def forward(self, data_subset, beta=1.0, batch_scale=1.0, device="cpu", epoch=None):
        # we assume data_subset containts two elements
        Y, c = data_subset
        Y, c = Y.to(device), c.to(device)

        # encode
        mu_z, sigma_z = self.encoder(Y, c)
        eps = torch.randn_like(mu_z)
        z = mu_z + sigma_z * eps

        # decode
        y_pred = self.decoder.forward(z, c)
        decor = decorrelation_loss(z, c)
        corr_coef = corrz(z, c)
        neg_loglik, penalty, int1, int2, int3, int4 = self.decoder.loss(y_pred, Y)

        # loss function
        VAE_KL_loss = KL_standard_normal(mu_z, sigma_z)

        # Note that when this loss (neg ELBO) is calculated on a subset (minibatch),
        # we should scale it by data_size/minibatch_size, but it would apply to all terms
        # total_loss = batch_scale * (neg_loglik + beta * VAE_KL_loss) + penalty
        # total_loss = neg_loglik + beta * VAE_KL_loss + 100000*decor
        total_loss = neg_loglik + beta * VAE_KL_loss

        return total_loss, int1, int2, int3, int4, decor, neg_loglik, corr_coef, VAE_KL_loss

    def calculate_test_loglik(self, Y, c):
        """
        maps (Y, x) to z and calculates p(y* | x, z_mu)
        :param Y:
        :param c:
        :return:
        """
        mu_z, sigma_z = self.encoder(Y, c)

        Y_pred = self.decoder.forward(mu_z, c)

        return self.decoder.loglik(Y_pred, Y)


    def optimize(self, data_loader, augmented_lagrangian_lr, n_iter=50000, logging_freq=20, logging_freq_int=100, batch_scale=1.0, account_for_noise=True, temperature_start=4.0, temperature_end=0.2, lambda_start=None, lambda_end=None, verbose=True, val_loader=None, train_eval_loader=None, early_stopping_patience=None, val_check_freq=250):
        """
        Train the model for the full number of iterations (no early stopping by default).

        Args:
            val_loader: DataLoader for validation set (no shuffle) - used for monitoring only
            train_eval_loader: DataLoader for training set without shuffle (for computing train varexp).
                              If None, uses data_loader but note this may be shuffled.
            early_stopping_patience: DISABLED by default (None). Set to a positive integer to enable
                                    early stopping after that many validation checks without improvement.
            val_check_freq: How often to check validation loss (in iterations). Default: 250
        """
        import copy

        # sample size
        N = len(data_loader.dataset)

        # number of iterations = (numer of epochs) * (number of iters per epoch)
        n_epochs = ceil(n_iter / len(data_loader))
        if verbose:
            logger.info(f"Fitting Neural Decomposition.\n\tData set size {N}. # iterations = {n_iter} (i.e. # epochs <= {n_epochs})\n")
            if val_loader is not None:
                logger.info(f"Validation monitoring enabled (no early stopping). Check freq={val_check_freq}\n")

        loss_values = np.zeros(ceil(n_iter // logging_freq))

        # Track overfitting metrics over training
        overfit_history = {
            'iterations': [],
            'train_neg_loglik': [],
            'val_neg_loglik': [],
            'overfit_ratio': [],
            'overfit_diff': [],
            'train_loss': [],
            'val_loss': [],
            # MSE-based metrics (cleaner for overfitting detection)
            'train_mse': [],
            'val_mse': [],
            'mse_ratio': [],  # val_mse / train_mse
            'mse_diff': [],   # val_mse - train_mse
            # Variance explained fractions (the key overfitting signal for ANOVA decomposition)
            'train_varexp_z1': [],      # z1 component variance explained on train
            'val_varexp_z1': [],        # z1 component variance explained on val
            'train_varexp_total': [],   # total variance explained (excluding noise) on train
            'val_varexp_total': [],     # total variance explained (excluding noise) on val
            'varexp_ratio': [],         # val_total / train_total - key overfitting metric
        }

        # Early stopping tracking
        best_val_mse = float('inf')
        best_val_iter = 0
        patience_counter = 0
        best_model_state = None
        best_decoder_state = None
        early_stopped = False

        if self.decoder.has_feature_level_sparsity:
            temperature_grid = torch.linspace(temperature_start, temperature_end, steps=n_iter // 10, device=self.device)

        if lambda_start is None:
            lambda_start = self.decoder.lambda0
            lambda_end = self.decoder.lambda0
        lambda_grid = torch.linspace(lambda_start, lambda_end, steps=n_iter // 10, device=self.device)

        # get shapes for integrals
        _int_z, _int_c, _int_cz = self.decoder.calculate_integrals_numpy()
        # log the integral values
        n_logging_steps = ceil(n_iter / logging_freq_int)
        int_z_values = np.zeros([n_logging_steps, _int_z.shape[0], self.output_dim])
        int_c_values = np.zeros([n_logging_steps, _int_c.shape[0], self.output_dim])
        if self.decoder.n_covariates_interactions != 0:
            int_cz_values = np.zeros([n_logging_steps, _int_cz.shape[0], self.output_dim])
        else:
            int_cz_values = np.zeros([n_logging_steps, _int_c.shape[0], self.output_dim])

        iteration = 0
        for epoch in range(n_epochs):

            for batch_idx, data_subset in enumerate(data_loader):

                if iteration >= n_iter:
                    break

                loss, int_z, int_c, int_cz_dc, int_cz_dz, decor, neg_loglik, corrcoef, kl_loss = self.forward(data_subset, beta=1.0, batch_scale=batch_scale, device=self.device)

                wandb.log({
                    "epoch": epoch,
                    f"{self.__class__.__name__}/loss": loss,
                    f"{self.__class__.__name__}/decorrelation": decor,
                    f"{self.__class__.__name__}/neg_loglik": neg_loglik,
                    f"{self.__class__.__name__}/kl_divergence": kl_loss,
                    f"{self.__class__.__name__}/corrcoef_z_c": corrcoef,
                })

                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
                self.optimizer.step()

                if self.decoder.has_feature_level_sparsity:
                    self.decoder.set_temperature(temperature_grid[iteration // 10])
                self.decoder.lambda0 = lambda_grid[iteration // 10]

                # update for BDMM
                with torch.no_grad():
                    self.decoder.Lambda_z += augmented_lagrangian_lr * int_z
                    for j in range(self.decoder.n_covariates):
                        self.decoder.Lambda_c[j] += augmented_lagrangian_lr * int_c[j]
                    for j in range(self.decoder.n_covariates_interactions):
                        self.decoder.Lambda_cz_1[j] += augmented_lagrangian_lr * int_cz_dc[j]
                        self.decoder.Lambda_cz_2[j] += augmented_lagrangian_lr * int_cz_dz[j]

                # logging for the loss function
                if iteration % logging_freq == 0:
                    index = iteration // logging_freq
                    loss_values[index] = loss.item()

                # logging for integral constraints
                if iteration % logging_freq_int == 0:
                    int_z, int_c, int_cz = self.decoder.calculate_integrals_numpy()

                    index = iteration // logging_freq_int
                    int_z_values[index, :] = int_z
                    int_c_values[index, :] = int_c
                    if self.decoder.n_covariates_interactions != 0:
                        int_cz_values[index, :] = int_cz

                    # Log integrals to wandb
                    wandb.log({
                        f"{self.__class__.__name__}/int_z_mean": np.mean(np.abs(int_z)),
                        f"{self.__class__.__name__}/int_z_max": np.max(np.abs(int_z)),
                        f"{self.__class__.__name__}/int_c_mean": np.mean(np.abs(int_c)),
                        f"{self.__class__.__name__}/int_c_max": np.max(np.abs(int_c)),
                        f"{self.__class__.__name__}/int_cz_mean": np.mean(np.abs(int_cz)) if self.decoder.n_covariates_interactions != 0 else 0.0,
                        f"{self.__class__.__name__}/int_cz_max": np.max(np.abs(int_cz)) if self.decoder.n_covariates_interactions != 0 else 0.0,
                    })

                # Validation check and early stopping
                if val_loader is not None and iteration % val_check_freq == 0:
                    self.eval()
                    with torch.no_grad():
                        val_losses = []
                        val_neglogliks = []
                        val_mses = []
                        for val_batch in val_loader:
                            val_loss_batch, _, _, _, _, val_mse_batch, val_neg_loglik_batch, _, _ = self.forward(
                                val_batch, beta=1.0, batch_scale=batch_scale, device=self.device
                            )
                            val_losses.append(val_loss_batch.item())
                            val_neglogliks.append(val_neg_loglik_batch.item())
                            # val_mse_batch could be 0 for some model variants, handle gracefully
                            if isinstance(val_mse_batch, torch.Tensor):
                                val_mses.append(val_mse_batch.item())
                            else:
                                val_mses.append(float(val_mse_batch))

                        val_loss_mean = np.mean(val_losses)
                        val_neg_loglik_mean = np.mean(val_neglogliks)
                        val_mse_mean = np.mean(val_mses)

                        # Train metrics (from current batch - noisy proxy)
                        train_neg_loglik = neg_loglik.item()
                        train_loss_val = loss.item()
                        # Get train MSE from the 6th return value (index 5)
                        if isinstance(decor, torch.Tensor):
                            train_mse = decor.item()  # decor position now holds MSE for this class
                        else:
                            train_mse = float(decor)

                        # NLL-based ratios (includes penalty - less clean)
                        overfit_ratio = val_neg_loglik_mean / (train_neg_loglik + 1e-8)
                        overfit_diff = val_neg_loglik_mean - train_neg_loglik

                        # MSE-based ratios (cleaner for overfitting detection)
                        mse_ratio = val_mse_mean / (train_mse + 1e-8)
                        mse_diff = val_mse_mean - train_mse

                        # ============================================================
                        # VARIANCE EXPLAINED COMPUTATION (key overfitting metric)
                        # ============================================================
                        # Compute variance explained fractions for train and val
                        # This is the proper metric for ANOVA-style decomposition

                        def compute_varexp_from_loader(loader, decoder, device):
                            """Compute raw variance explained from a data loader.

                            Returns raw variances (not normalized fractions):
                            - Var(f_z), Var(f_c1), ..., Var(error)
                            """
                            # Collect all data from loader
                            Y_list, z_list, c_lists = [], [], [[] for _ in range(decoder.n_covariates)]
                            for batch in loader:
                                Y_list.append(batch[0])
                                z_list.append(batch[1])
                                for i in range(decoder.n_covariates):
                                    c_lists[i].append(batch[2 + i])

                            Y_all = torch.cat(Y_list, dim=0).to(device)
                            z_all = torch.cat(z_list, dim=0).to(device)
                            c_all = [torch.cat(c_list, dim=0).to(device) for c_list in c_lists]

                            # Compute Y_error (residual)
                            y_pred = decoder.forward(z_all, c_all)
                            Y_error = Y_all - y_pred

                            # Compute raw variances (divide_by_total_var=False)
                            varexp = decoder.fraction_of_variance_explained(
                                z_all, c_all, Y_error=Y_error, divide_by_total_var=False
                            )
                            return varexp  # shape: [n_features, n_components+1], raw variances

                        # Compute for validation (raw variances)
                        val_varexp = compute_varexp_from_loader(val_loader, self.decoder, self.device)
                        val_varexp_z1 = val_varexp[:, 0].mean().item()  # z1 variance (mean over features)
                        # Total signal variance = sum of all components except noise (last column)
                        val_varexp_total = val_varexp[:, :-1].sum(dim=1).mean().item()

                        # Compute for train (use train_eval_loader if provided, else use training loader)
                        eval_loader = train_eval_loader if train_eval_loader is not None else data_loader
                        train_varexp = compute_varexp_from_loader(eval_loader, self.decoder, self.device)
                        train_varexp_z1 = train_varexp[:, 0].mean().item()
                        train_varexp_total = train_varexp[:, :-1].sum(dim=1).mean().item()

                        # Variance explained ratio (val/train) - key overfitting signal
                        # Ratio close to 1 = good generalization, <0.8 = overfitting
                        varexp_ratio = val_varexp_total / (train_varexp_total + 1e-8)

                        # Early stopping logic
                        if val_mse_mean < best_val_mse:
                            best_val_mse = val_mse_mean
                            best_val_iter = iteration
                            patience_counter = 0
                            # Save best model state
                            best_model_state = copy.deepcopy(self.state_dict())
                            best_decoder_state = copy.deepcopy(self.decoder.state_dict())
                            if verbose:
                                logger.info(f"\t*** New best val MSE: {best_val_mse:.6f} at iter {iteration} ***")
                        else:
                            patience_counter += 1

                        # Store in history for later analysis
                        overfit_history['iterations'].append(iteration)
                        overfit_history['train_neg_loglik'].append(train_neg_loglik)
                        overfit_history['val_neg_loglik'].append(val_neg_loglik_mean)
                        overfit_history['overfit_ratio'].append(overfit_ratio)
                        overfit_history['overfit_diff'].append(overfit_diff)
                        overfit_history['train_loss'].append(train_loss_val)
                        overfit_history['val_loss'].append(val_loss_mean)
                        overfit_history['train_mse'].append(train_mse)
                        overfit_history['val_mse'].append(val_mse_mean)
                        overfit_history['mse_ratio'].append(mse_ratio)
                        overfit_history['mse_diff'].append(mse_diff)
                        # Variance explained fractions
                        overfit_history['train_varexp_z1'].append(train_varexp_z1)
                        overfit_history['val_varexp_z1'].append(val_varexp_z1)
                        overfit_history['train_varexp_total'].append(train_varexp_total)
                        overfit_history['val_varexp_total'].append(val_varexp_total)
                        overfit_history['varexp_ratio'].append(varexp_ratio)

                        if early_stopping_patience:
                            val_loss_str = f"\tVal MSE {val_mse_mean:.4f}\tVarExp: tr={train_varexp_total:.3f} val={val_varexp_total:.3f} ratio={varexp_ratio:.3f}\tPatience {patience_counter}/{early_stopping_patience}"
                        else:
                            val_loss_str = f"\tVal MSE {val_mse_mean:.4f}\tVarExp: tr={train_varexp_total:.3f} val={val_varexp_total:.3f} ratio={varexp_ratio:.3f}"
                        wandb.log({
                            f"{self.__class__.__name__}/val_loss": val_loss_mean,
                            f"{self.__class__.__name__}/val_neg_loglik": val_neg_loglik_mean,
                            f"{self.__class__.__name__}/train_neg_loglik_at_val": train_neg_loglik,
                            f"{self.__class__.__name__}/overfit_ratio": overfit_ratio,
                            f"{self.__class__.__name__}/overfit_diff": overfit_diff,
                            # MSE metrics
                            f"{self.__class__.__name__}/train_mse": train_mse,
                            f"{self.__class__.__name__}/val_mse": val_mse_mean,
                            f"{self.__class__.__name__}/mse_ratio": mse_ratio,
                            f"{self.__class__.__name__}/mse_diff": mse_diff,
                            f"{self.__class__.__name__}/best_val_mse": best_val_mse,
                            f"{self.__class__.__name__}/best_val_iter": best_val_iter,
                            f"{self.__class__.__name__}/patience_counter": patience_counter,
                            # Variance explained fractions (key overfitting metrics)
                            f"{self.__class__.__name__}/train_varexp_z1": train_varexp_z1,
                            f"{self.__class__.__name__}/val_varexp_z1": val_varexp_z1,
                            f"{self.__class__.__name__}/train_varexp_total": train_varexp_total,
                            f"{self.__class__.__name__}/val_varexp_total": val_varexp_total,
                            f"{self.__class__.__name__}/varexp_ratio": varexp_ratio,
                            "iteration": iteration,
                        })

                    self.train()  # Back to training mode

                    # Log progress
                    if verbose:
                        logger.info(f"\tIter {iteration:5}.\tTotal loss {loss.item():.3f}{val_loss_str}")

                    # Check early stopping (only if patience is enabled)
                    if early_stopping_patience is not None and early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                        logger.info(f"\n{'='*60}")
                        logger.info(f"EARLY STOPPING TRIGGERED at iteration {iteration}")
                        logger.info(f"  No improvement for {patience_counter} checks ({patience_counter * val_check_freq} iterations)")
                        logger.info(f"  Restoring best model from iteration {best_val_iter}")
                        logger.info(f"  Best validation MSE: {best_val_mse:.6f}")
                        logger.info(f"{'='*60}\n")
                        early_stopped = True
                        break

                iteration += 1

            # Break outer loop if early stopped
            if early_stopped:
                break

        # Restore best model only if early stopping was used and triggered
        if early_stopping_patience is not None and early_stopping_patience > 0 and best_model_state is not None:
            logger.info(f"Restoring best model from iteration {best_val_iter} (val MSE: {best_val_mse:.6f})")
            self.load_state_dict(best_model_state)
            self.decoder.load_state_dict(best_decoder_state)

        # collect all integral values into one array (legacy format)
        integrals = np.hstack([int_z_values, int_c_values, int_cz_values]).reshape(n_iter // logging_freq_int, -1).T

        # Return individual arrays for plotting and analysis
        integrals_dict = {
            'int_z': int_z_values,      # shape: [n_logging_steps, n_z_components, output_dim]
            'int_c': int_c_values,      # shape: [n_logging_steps, n_covariates, output_dim]
            'int_cz': int_cz_values,    # shape: [n_logging_steps, n_interactions, output_dim]
            'iterations': np.arange(0, n_iter, logging_freq_int)[:n_logging_steps],
        }

        # Convert overfit_history lists to numpy arrays for easier analysis
        for key in overfit_history:
            overfit_history[key] = np.array(overfit_history[key])

        # Add metadata
        overfit_history['best_val_mse'] = best_val_mse
        overfit_history['best_val_iter'] = best_val_iter
        overfit_history['early_stopped'] = early_stopped
        overfit_history['final_iteration'] = iteration

        # Print summary
        if val_loader is not None:
            logger.info(f"\n{'='*60}")
            logger.info(f"TRAINING SUMMARY")
            logger.info(f"  Final iteration: {iteration}")
            logger.info(f"  Best validation MSE: {best_val_mse:.6f} at iteration {best_val_iter}")
            logger.info(f"  Early stopped: {early_stopped}")
            if early_stopped:
                logger.info(f"  Model restored to iteration {best_val_iter}")
            logger.info(f"{'='*60}\n")

        return loss_values, integrals, integrals_dict, overfit_history


class CVAE_multiple_covariates_with_fixed_z(CVAE_multiple_covariates):
    """
    Same as the above CVAE_multiple_covariates class, but assuming a fixed latent variable z, thus effectively only training the decoder.
    We assume z is given by the data_loader, i.e. we assume it returns tuples (Y, c, z)
    """

    def __init__(self, encoder, decoder, lr, device):
        super().__init__(encoder=encoder, decoder=decoder, lr=lr, device=device)

    def forward(self, data_subset, beta=1.0, batch_scale=1.0, device="cpu"):
        # we assume data_subset containts three elements
        Y, c, z = data_subset

        # decoding step
        y_pred = self.decoder.forward(z, c)
        decor = decorrelation_loss(z, c)
        corr_coef = corrz(z, c)
        decoder_loss, penalty, int1, int2, int3, int4 = self.decoder.loss(y_pred, Y)

        # no KL(q(z) | p(z)) term because z fixed
        total_loss = decoder_loss
        kl_loss = torch.tensor(0.0, device=Y.device)  # No KL for fixed z

        return total_loss, int1, int2, int3, int4, decor, decoder_loss, corr_coef, kl_loss


class CVAE_multiple_latent_spaces_with_covariates(CVAE_multiple_covariates):
    """
    Same as the above CVAE_multiple_covariates class, but assuming an arbitrary number of fixed latent spaces, thus effectively only training the decoders.
    """

    def __init__(self, encoder, decoder, lr, device):
        super().__init__(encoder=encoder, decoder=decoder, lr=lr, device=device)

    def forward(self, data_subset, beta=1.0, batch_scale=1.0, device="cpu"):
        # decoding step
        Y = data_subset[0]
        y_pred = self.decoder.forward(data_subset[1], data_subset[2:]) # temporarily for backward compatibility
        decoder_loss, penalty, int1, int2, int3, int4 = self.decoder.loss(y_pred, Y)

        # Compute pure MSE for overfitting detection (no penalty, no learned noise)
        mse = ((Y - y_pred) ** 2).mean()

        # no KL(q(z) | p(z)) term because z fixed
        total_loss = decoder_loss
        kl_loss = torch.tensor(0.0, device=Y.device)  # No KL for fixed z

        return total_loss, int1, int2, int3, int4, mse, decoder_loss, 0, kl_loss