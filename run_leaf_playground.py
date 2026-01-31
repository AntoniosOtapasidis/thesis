import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import wandb
from datetime import datetime
from statsmodels.stats.multitest import multipletests
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from ND.encoder import cEncoder
from ND.decoder import Decoder
from ND.CVAE import CVAE
from ND.helpers import expand_grid
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.uniform import Uniform
from matplotlib.ticker import FuncFormatter
from tqdm.auto import tqdm
from matplotlib.lines import Line2D

from matplotlib.ticker import FuncFormatter
from DisentangledSSL.algorithms import *
from DisentangledSSL.losses import *
from DisentangledSSL.models import *
from DisentangledSSL.dataset import *
from data_generation.generate_casual import *
from data_generation.generate_MNIST import *
from DisentangledSSL.utils import *
from ND.decoder_multiple_covariates import Decoder_multiple_latents
from ND.CVAE_multiple_covariates import CVAE_multiple_latent_spaces_with_covariates
from sklearn.decomposition import PCA
import numpy as np
import os
import math
import umap
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import math
import umap
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
from ignite.metrics import HSIC
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from torch.utils.data import Subset, DataLoader
import seaborn as sns

#HSIC estimation
from collections import OrderedDict

import torch
from torch import nn, optim
 
from sklearn.manifold import TSNE
import time as time_module
from scipy.stats import ks_2samp, wasserstein_distance
import torch, numpy as np, pandas as pd


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def debugger(name, x):
    try:
        t = type(x).__name__
        if isinstance(x, torch.Tensor):
            print(f"[{name}] type={t}, shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}, requires_grad={x.requires_grad}")
        elif isinstance(x, np.ndarray):
            print(f"[{name}] type={t}, shape={x.shape}, dtype={x.dtype}")
        elif isinstance(x, pd.DataFrame):
            print(f"[{name}] type=DataFrame, shape={x.shape}, index={x.index.dtype}, columns={len(x.columns)}")
            print(f"[{name}] columns[:10]={list(x.columns[:10])}")
        elif isinstance(x, pd.Series):
            print(f"[{name}] type=Series, shape={x.shape}, dtype={x.dtype}, name={x.name}")
        elif isinstance(x, (list, tuple)):
            print(f"[{name}] type={t}, len={len(x)}; types={[type(i).__name__ for i in x[:5]] + (['...'] if len(x)>5 else [])}")
        else:
            print(f"[{name}] type={t}: {x}")
    except Exception as e:
        print(f"[{name}] <print-error> {e}")


def verify_encoder_frozen(model, model_name="disen"):
    """
    Verify that pretrained encoders (zsmodel) are frozen and not receiving gradients.

    Returns:
        dict: Summary statistics about frozen/trainable parameters
    """
    frozen_params = []
    trainable_params = []
    frozen_param_count = 0
    trainable_param_count = 0

    for name, param in model.named_parameters():
        param_size = param.numel()
        if 'zsmodel' in name:
            frozen_params.append(name)
            frozen_param_count += param_size
            if param.requires_grad:
                print(f" WARNING: {name} should be frozen but requires_grad=True!")
        else:
            trainable_params.append(name)
            trainable_param_count += param_size

    print(f"\n{'='*80}")
    print(f"ENCODER FREEZING VERIFICATION - {model_name.upper()}")
    print(f"{'='*80}")
    print(f"Frozen parameters (zsmodel):{len(frozen_params):3d} layers, {frozen_param_count:,} params")
    print(f"Trainable parameters:{len(trainable_params):3d} layers, {trainable_param_count:,} params")
    print(f"Total parameters:{len(frozen_params) + len(trainable_params):3d} layers, {frozen_param_count + trainable_param_count:,} params")
    print(f"Frozen ratio:{100 * frozen_param_count / (frozen_param_count + trainable_param_count):.1f}%")

    # Check if zsmodel exists and verify detach in forward pass
    if hasattr(model, 'zsmodel'):
        print(f"zsmodel exists:True")
        print(f"zsmodel.requires_grad: {model.zsmodel.requires_grad}")
    else:
        print(f"zsmodel does not exist in model!")

    # Verify all zsmodel parameters are frozen
    all_frozen = all(not param.requires_grad for name, param in model.named_parameters() if 'zsmodel' in name)

    if all_frozen and len(frozen_params) > 0:
        print(f"\n VERIFICATION PASSED: All encoder parameters are properly frozen!")
    elif len(frozen_params) == 0:
        print(f"\n  WARNING: No frozen parameters found! Check if zsmodel is properly set.")
    else:
        print(f"\n VERIFICATION FAILED: Some encoder parameters are not frozen!")

    print(f"{'='*80}\n")

    return {
        'frozen_count': len(frozen_params),
        'trainable_count': len(trainable_params),
        'frozen_param_count': frozen_param_count,
        'trainable_param_count': trainable_param_count,
        'all_frozen': all_frozen,
        'verification_passed': all_frozen and len(frozen_params) > 0
    }


def step1_filename(args):
    return f'cib_seed{args.seed}_{args.beta}_kappa{int(args.kappa)}_epoch{args.num_epoch_s1}_dim{args.embed_dim}.tar'

def train_mp(beta, train_loader, test_loader, train_dataset, test_dataset, args):
    step1_name = step1_filename(args)
    out_dir = f'./results_synthetic_task/{args.data_mode}/models/'
    os.makedirs(out_dir, exist_ok=True)
    step1_path = os.path.join(out_dir, step1_name)
    print(f'Training step 1: CIB model for beta:', beta, 'seed:', args.seed)
    cmib = MVInfoMaxModel(
        args.dim_info['X1'], args.dim_info['X2'],
        args.hidden_dim, args.embed_dim,
        initialization='normal', distribution='vmf', vmfkappa=args.kappa,
        layers=args.layers, beta_start_value=beta, beta_end_value=beta,
        beta_n_iterations=800, beta_start_iteration=0,
        head=args.head, simclr=args.simclr
    )
    if args.device != "cpu":
        cmib = cmib.cuda()
    cmib_optim = optim.Adam(cmib.parameters(), lr=args.lr_s1, weight_decay = 1e-4)
    logs = train(cmib, train_loader, cmib_optim, train_dataset, test_dataset, num_epoch=args.num_epoch_s1)
    if not args.debug_mode:
        torch.save(cmib.state_dict(), step1_path)
        print(f'Save model to {step1_path}')
    return logs

def train_step2(args, train_loader, test_loader, train_dataset, test_dataset):
    print("ENTER train_step2", flush=True)

    step1_name = step1_filename(args)
    step1_path = os.path.join(f'./results_synthetic_task/{args.data_mode}/models/', step1_name)
    print("step1_path:", step1_path, flush=True)

    cmib_step1 = MVInfoMaxModel(
        args.dim_info['X1'], args.dim_info['X2'],
        args.hidden_dim, args.embed_dim,
        initialization='normal', distribution='vmf', vmfkappa=args.kappa,
        layers=args.layers, beta_start_value=args.beta, beta_end_value=args.beta,
        beta_n_iterations=800, beta_start_iteration=0
    )
    if args.device != "cpu":
        cmib_step1 = cmib_step1.cuda()

    cmib_step1.load_state_dict(torch.load(step1_path))


    #### FREEZE THE ENDOCER PARAMETERS before feeding them to the disen Model
    for p in cmib_step1.parameters():
        p.requires_grad = False
    cmib_step1.eval()

    print('Training step 2: Disen model')
    disen = DisenModel(
        cmib_step1, args.dim_info['X1'], args.dim_info['X2'],
        args.hidden_dim, args.embed_dim, zs_dim=args.embed_dim,
        initialization='normal',
        layers=args.layers, lmd_start_value=args.lmd_start, lmd_end_value=args.lmd_end,
        lmd_n_iterations=800, lmd_start_iteration=0,
        ortho_norm=args.ortho_norm, condzs=args.condzs, proj=args.proj,
        usezsx=args.usezsx, apdzs=args.apdzs, hsic_weight=args.hsic_weight
    )

    if args.device != "cpu":
        disen = disen.cuda()

    # if hasattr(disen, "zsmodel"):
    #     disen.zsmodel.eval()
    # # Verify encoder freezing before training
    # verify_encoder_frozen(disen, model_name="DisenModel")

    print("cmib_step1 any trainable?", any(p.requires_grad for p in cmib_step1.parameters()))

    # If disen has zsmodel and it's the same module, check that too
    if hasattr(disen, "zsmodel"):
        print("disen.zsmodel is cmib_step1?", disen.zsmodel is cmib_step1)
        print("disen.zsmodel any trainable?", any(p.requires_grad for p in disen.zsmodel.parameters()))

    disen_optim = optim.Adam(disen.parameters(), lr=args.lr_s2, weight_decay = 1e-4)
    logs = train_Disen(disen, train_loader, disen_optim, train_dataset, test_dataset,
                       num_epoch=args.num_epoch_s2, noise_scale=args.noise_scale, drop_scale=args.drop_scale)
    return logs, disen

def gather_embeddings(disen_model, loader, device="cuda"):
    zs1_list, zs2_list, zc1_list, zc2_list = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            input1, input2, *_ = batch
            input1 = input1.to(device)
            input2 = input2.to(device)
            zs1, zs2, zc1, zc2 = disen_model.get_three_embeddings([input1, input2])
            zs1_list.append(zs1.cpu().numpy())
            zs2_list.append(zs2.cpu().numpy())
            zc1_list.append(zc1.cpu().numpy())
            zc2_list.append(zc2.cpu().numpy())
    zs1 = np.concatenate(zs1_list, axis=0)
    zs2 = np.concatenate(zs2_list, axis=0)
    zc1 = np.concatenate(zc1_list, axis=0)
    zc2 = np.concatenate(zc2_list, axis=0)
    return zs1, zs2, zc1, zc2

# def residualize_data(c, *data_tensors, config=None, device="cuda", n_covariates=1):
#     Y_combined = torch.cat(data_tensors, dim=1)
#     N = c.shape[0]
#     z1 = Uniform(-2.0, 2.0).sample((N, 1))
#     z = z1
#     dataset = TensorDataset(Y_combined.to(device), z.to(device), c.to(device))
#     data_loader = DataLoader(dataset, shuffle=False, batch_size=config.batch_size)
#     z_dim = z.shape[1]
#     lim_val = 2.0
#     num_samples = 10000
#     data_dim = Y_combined.shape[1]
#     hidden_dim = 32
#     grid_z = (torch.rand(num_samples, z_dim, device=device) * 2 * lim_val) - lim_val
#     grid_cov = lambda x: (torch.rand(num_samples, x.shape[1], device=device) * 2 * lim_val) - lim_val
#     grid_c = [grid_cov(x) for x in [c]]
#     encoder_mapping = nn.Sequential()
#     decoder_z = nn.Sequential(
#         nn.Linear(z_dim, hidden_dim),
#         nn.Tanh(),
#         nn.Linear(hidden_dim, data_dim)
#     )
#     encoder = cEncoder(z_dim=z_dim, mapping=encoder_mapping)
#     decoders_c = [nn.Sequential(
#         nn.Linear(x.shape[1], hidden_dim),
#         nn.Tanh(),
#         nn.Linear(hidden_dim, data_dim)
#     ) for x in [c]]
#     decoders_cz = []
#     decoder = Decoder_multiple_latents(
#         data_dim, n_covariates,
#         grid_z, grid_c,
#         decoder_z, decoders_c, decoders_cz,
#         has_feature_level_sparsity=False, p1=0.1, p2=0.1, p3=0.1,
#         lambda0=1e3, penalty_type="MDMM", device=device
#     )
#     model = CVAE_multiple_latent_spaces_with_covariates(encoder, decoder, lr=1e-4, device=device)
#     loss, integrals = model.optimize(data_loader, n_iter=config.iters_res, augmented_lagrangian_lr=0.1)
#     with torch.no_grad():
#         Y_error_list, Y_pred_list = [], []
#         for batch in data_loader:
#             Y_batch, z_batch, c_batch = batch
#             confounders = [c_batch.to(device)]
#             Y_pred = decoder.forward_c(confounders)[0]
#             Y_error = Y_batch - Y_pred
#             Y_error_list.append(Y_error)
#             Y_pred_list.append(Y_pred)
#         Y_error = torch.cat(Y_error_list, dim=0)
#     dims = [d.shape[1] for d in data_tensors]
#     Y_parts = torch.split(Y_error, dims, dim=1)
#     return Y_parts


def build_variance_table(varexp, Y_torch, split, metabolite_names=None):
    comp_keys = ["z1", "zs2", "zc1"]
    rows = []
    varexp_np = varexp.numpy()
    total_var = Y_torch.var(dim=0)

    for i in range(Y_torch.shape[1]):
        outcome = metabolite_names[i] if i is not None else f"metabolite_{i}"
        outcome_var = float(total_var[i].item())

        # take only the first three components (z1, zs2, zc1)
        est_abs = varexp_np[i][:len(comp_keys)]
        est_sum = float(np.sum(est_abs))

        # compute fractions for each component
        est_fracs = [float(v)/outcome_var if outcome_var > 0 else 0.0 for v in est_abs]

        # compute noise as leftover variance fraction
        est_noise_abs = max(outcome_var - est_sum, 0.0)
        est_noise_frac = est_noise_abs/outcome_var if outcome_var > 0 else 0.0

        # add component rows
        for key, est_f in zip(comp_keys, est_fracs):
            rows.append({
                "outcome": outcome,
                "component": key,
                "split": split,
                "est_fraction": est_f,
            })

        # add noise row
        rows.append({
            "outcome": outcome,
            "component": "noise",
            "split": split,
            "est_fraction": est_noise_frac,
        })

    return pd.DataFrame(rows)



# def build_variance_table(varexp, Y_torch,gt, split):
#     comp_keys = ["z1", "zs2", "zc1"]
#     rows = []
#     varexp_np = varexp.numpy()
#     total_var = Y_torch.var(dim=0)
#     gt_map = {"z1": "Z1", "zs2": "Z2", "zc1": "Zs"}

#     for i in range(Y_torch.shape[1]):
#         #outcome_name = str(metabolite_names[i]) if metabolite_names is not None else f"metabolite_{i}"
#         outcome_name = f'outcome_{i +1}'
#         outcome_var = float(total_var[i].item())
#         est_abs = varexp_np[i][:len(comp_keys)]
#         est_sum = float(np.sum(est_abs))

#         # compute fractions for each component
#         est_fracs = [float(v)/outcome_var if outcome_var > 0 else 0.0 for v in est_abs]
#         est_noise_abs = max(outcome_var - est_sum, 0.0)
#         est_noise_frac = est_noise_abs/outcome_var if outcome_var > 0 else 0.0
#         # if callable(metabolite_names):
#         #     raise TypeError("metabolite_names is a function; pass a list/array of names.")

#         # # add component rows
#         for key, est_f in zip(comp_keys, est_fracs):
#             rows.append({
#                 "outcome": outcome_name,
#                 "component": key,
#                 "gt_fraction": float(gt["shares"][gt_map[key]][i]),
#                 "split": split,
#                 "est_fraction": est_f,
#             })
#         # add noise row
#         rows.append({
#             "outcome": outcome_name,
#             "component": "noise",
#             "gt_fraction": float(gt["shares"]["noise"][i]),
#             "split": split,
#             "est_fraction": est_noise_frac,
#          })

#     return pd.DataFrame(rows)



def make_mnist_gt(Y_like, add_style_target, w1, ws, wc, sigma_y,
                  z1p_var=1.0, s_var=1.0, c_var=None):
    """
    Ground-truth variance shares for Y columns produced by the MNIST generator.

    Y layout:
      - Always: dim 0 -> digit of image 1 (Z1=1), dim 1 -> digit of image 2 (Z2=1)
      - If add_style_target:
          dim 2 -> y_style_base      (using ws, wc)
          dim 3 -> y_style_01_09     (signal split s:c = 0.1 : 0.9, same total signal power as base)
          dim 4 -> y_style_03_07     (0.3 : 0.7)
          dim 5 -> y_style_06_04     (0.6 : 0.4)
          dim 6 -> y_confonly        (depends only on c + noise)
        Else:
          dim 2 -> y_confonly

    Conventions:
      - We attribute the style 's' component to Z2, the confounder 'c' to C,
        optional z1p (if used via w1) to Z1, and the Gaussian term to 'noise'.
      - Variances: Var(z1p)=z1p_var, Var(s)=s_var, Var(c)=c_var.
        If c_var is None, we default to Bernoulli(0.5) => Var(c)=0.25.
    """
    dim_y = int(Y_like.shape[1])


    shares = {
        "Z1":    np.zeros(dim_y, dtype=np.float32),
        "Z2":    np.zeros(dim_y, dtype=np.float32),
        "Zs":    np.zeros(dim_y, dtype=np.float32),  # unused in this GT
        "noise": np.zeros(dim_y, dtype=np.float32),
    }

    # Digits (deterministic assignments)
    shares["Z1"][0] = 1.0  # digit of image 1
    shares["Z2"][1] = 1.0  # digit of image 2

    def set_pve(idx, w1_, ws_, wc_, sig_):
        """Fill shares for column idx using weights and component variances."""
        num_Z1 = (w1_**2) * z1p_var
        num_Z2 = (ws_**2) * s_var
        num_eps = (sig_**2)
        denom = num_Z1 + num_Z2 + num_eps
        # Guard against zero denom (shouldn't happen with noise present)
        if denom <= 0:
            return
        shares["Z1"][idx]    = num_Z1 / denom
        shares["Zs"][idx]    = num_Z2 / denom
        shares["noise"][idx] = num_eps / denom

    if add_style_target:
        P = max(ws**2 + wc**2, 1e-12)
        splits = [
            (0.1, 0.9),  # y_style_01_09
            (0.3, 0.7),  # y_style_03_07
            (0.6, 0.4),  # y_style_06_04
            (1.0, 0.0),  # y_style_10_00
        ]
        for k, (ps, pc) in enumerate(splits, start=2):
            ws_r = np.sqrt(ps * P)
            wc_r = np.sqrt(pc * P)
            set_pve(idx=k, w1_=0.0, ws_=ws_r, wc_=wc_r, sig_=sigma_y)

        conf_idx = 6
        set_pve(idx=conf_idx, w1_=0.0, ws_=0.0, wc_=wc, sig_=sigma_y)

    else:
        set_pve(idx=2, w1_=0.0, ws_=0.0, wc_=wc, sig_=sigma_y)

    return {"shares": shares}





def validate_unique_filename(filepath):
    """Check if file exists and warn user if it does."""
    if os.path.exists(filepath):
        print(f"WARNING: File already exists and will be overwritten: {filepath}")
        return False
    return True

def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['synthetic_linear', 'synthetic_nonlinear', 'mnist', "microbiome", "microbiome_synthetic","microbiome_synthetic_non_linear","microbiome_synthetic_ilr","COPSAC_clone"], default='microbiome_synthetic')
    parser.add_argument('--data_mode', type=str, default='entangle')
   # parser.add_argument('--num_data', type=int, default=1000)
   # parser.add_argument('--dim_info', type=dict, default={'Z1': 50, 'Zs': 50, 'Z2': 50, 'X1': 100, 'X2': 100})
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iters_pred', type=int, default=5000)
   # parser.add_argument('--iters_res', type=int, default=5000)
    parser.add_argument('--hsic_weight', type=float, default=0.1) #Statistical independence weight
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--kappa', type=float, default=1000)
    parser.add_argument('--ortho_norm', type=str2bool, default=True)
    parser.add_argument('--condzs', type=str2bool, default=True)
    parser.add_argument('--proj', type=str2bool, default=False)
   # parser.add_argument('--proj', type=str, default ='none')
    parser.add_argument('--apdzs', type=str2bool, default=True)
    parser.add_argument('--usezsx', type=str2bool, default=True)
    parser.add_argument('--simclr', type=str2bool, default=False)
    parser.add_argument('--head', type=str, default='none')
    parser.add_argument('--lmd_start', type=float, default=0.01)
    parser.add_argument('--lmd_end', type=float, default=1)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epoch_s1', type=int, default=32)
    parser.add_argument('--num_epoch_s2', type=int, default=32)
    parser.add_argument('--lr_s1', type=float, default=1e-4)
    parser.add_argument('--lr_s2', type=float, default=1e-4)
    parser.add_argument('--noise_scale', type=float, default=0.02)
    parser.add_argument('--drop_scale', type=float, default=1)
    parser.add_argument('--debug_mode', type=str2bool, default=False)
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument(
        '--ablation',
        type=str,
        choices=['both', 'x1_only', 'x2_only'],
        default='both',
        help='Which modalities to use: both, x1_only, x2_only'
    )

    # Neural Decomposition (ND) hyperparameters
    parser.add_argument('--augmented_lagrangian_lr', type=float, default=0.0001,
                        help='Learning rate for augmented Lagrangian multipliers (default: 0.01)')
    parser.add_argument('--lambda0', type=float, default=1e3,
                        help='Penalty strength for integral constraints (default: 1e3)')
    parser.add_argument('--decoder_lr', type=float, default=1e-4,
                        help='Learning rate for ND decoder (default: 1e-4)')
    parser.add_argument('--penalty_type', type=str, default='MDMM',
                        choices=['fixed', 'BDMM', 'MDMM'],
                        help='Penalty type for integral constraints (default: MDMM)')

    # Permutation testing
    parser.add_argument('--run_permutation_test', type=str2bool, default=False,
                        help='Run Y-permutation test after training (default: False)')
    parser.add_argument('--n_permutations', type=int, default=10,
                        help='Number of permutations (default: 1000)')

    # MNIST-specific
    parser.add_argument('--prob_same', type=float, default=1.0/3.0)
    parser.add_argument('--alpha_s', type=float, default=0.75)
    parser.add_argument('--alpha_c', type=float, default=0.4)
    parser.add_argument('--patch', type=int, default=4)
    parser.add_argument('--p_high', type=float, default=0.8)
    parser.add_argument('--p_low', type=float, default=0.2)
    parser.add_argument('--S_high', type=str, default="0,1,2,3,4")
    parser.add_argument('--add_style_target', type=str2bool, default=True)
    parser.add_argument('--w1', type=float, default=0.3)
    parser.add_argument('--ws', type=float, default=0.5)
    parser.add_argument('--wc', type=float, default=0.2)
    parser.add_argument('--sigma_y', type=float, default=0.3)
  #  parser.add_argument('--output_dir', type=str, default="/users/antonios/LEAF_revisit/LEAF/one_hold/test2layers",
  #                      help='Output directory for results')
    args = parser.parse_args()
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project="disentangled_anova", config={
        "iters": args.iters_pred, 
        "batch_size": args.batch_size,
      #  "iters_res": args.iters_res,
    })
    config = wandb.config


#Synthetic microbiome
    #the first step would be to model the microbiome synthetic data and use random data for the viruses not associated with the 
    #metbolites and see the level of disentanglement we can achieve


    #The question that we aim to answer with the qt is to find who interacts with whom in terms of microbiome and virome
    # Ground truth “who affects Y” comes from Cij
    #bacterium j affects metabolite i if Cij[i, j] ≠ 0
    #the strength/sign is in the magnitude and sign of Cij[i, j]
    #Viruses have no interaction with Y at all:
    #So their effective “Y weight” is exactly zero for every virus

    if args.dataset == "COPSAC_clone":
        df_X1 = pd.read_csv(
            "/users/antonios/LEAF_revisit/LEAF/COPSAC_clone/X1_bacteria_synthetic_CLR_COPSAC.csv"
        )
        df_X2 = pd.read_csv(
            "/users/antonios/LEAF_revisit/LEAF/COPSAC_clone/X2_viruses_synthetic_CLR_COPSAC.csv"
        )
        df_Y = pd.read_csv(
            "/users/antonios/LEAF_revisit/LEAF/COPSAC_clone/Y_metabolites_log_synthetic_complex_RA.csv"
        )


        # 1) index by 
        df_X1 = df_X1.set_index("sim")
        df_X2 = df_X2.set_index("sim")
        df_Y  = df_Y.set_index("sim")

        common_idx = (
            df_Y.index
            .intersection(df_X1.index)
            .intersection(df_X2.index)
        )
        common_idx = common_idx.sort_values()

        df_X1 = df_X1.loc[common_idx]
        df_X2 = df_X2.loc[common_idx]
        df_Y  = df_Y.loc[common_idx]

        # 2) drop 'rep', get numpy
        df_X1_feats = df_X1.drop(columns=["rep"])
        df_X2_feats = df_X2.drop(columns=["rep"])
        df_Y_feats  = df_Y.drop(columns=["rep"])

        # Store metabolite names for later use
        metabolite_names = list(df_Y_feats.columns)

        X1_np = df_X1_feats.to_numpy(dtype=np.float32)
        X2_np = df_X2_feats.to_numpy(dtype=np.float32)
        Y_np  = df_Y_feats.to_numpy(dtype=np.float32)

        dim_X1 = X1_np.shape[1]
        dim_X2 = X2_np.shape[1]
        dim_Y  = Y_np.shape[1]
        Y = Y_np

        # 3) load GT variance shares from R
        df_gt = pd.read_csv(
            "/users/antonios/LEAF_revisit/LEAF/COPSAC_clone/GT_virome_variance_shares_complex.csv"
        )

        # ensure metabolite ordering in GT matches Y columns
        # df_gt$met is like "met_1", "met_2", ...
        # df_Y_feats.columns should be same strings
        assert list(df_gt["met"]) == list(df_Y_feats.columns), \
            "GT metabolite order does not match Y columns"


        Z1_share    = df_gt["share_Z1"].to_numpy(dtype=np.float32)
        Z2_share    = df_gt["share_Z2"].to_numpy(dtype=np.float32)
        Zs_share    = df_gt["share_Zs"].to_numpy(dtype=np.float32)
        noise_share = df_gt["share_noise"].to_numpy(dtype=np.float32)

        shares = {
            "Z1":    Z1_share,
            "Z2":    Z2_share,
            "Zs":    Zs_share,
            "noise": noise_share,
        }

        gt = {"shares": shares}

        # 4) data and targets for your model (unchanged)
        #data = np.stack([X1_np, X2_np], axis=0)  # (2, N, dim)
        X1_np= X1_np.astype(np.float32)
        X2_np= X2_np.astype(np.float32)
        #data = (X1_np, X2_np)
        targets1 = Y_np[:, 0]
        targets2 = Y_np[:, 1]
        targets3 = Y_np[:, 2]

        # args.dim_info['X1'] = dim_X1
        # args.dim_info['X2'] = dim_X2
        # args.dim_info['Y']  = dim_Y

        if not hasattr(args, "dim_info") or args.dim_info is None:
            # initialize with some defaults for latent dims; adjust if you want different sizes
            args.dim_info = {
                'X1': dim_X1,
                'X2': dim_X2,
                'Y':  dim_Y,
                "Z1": 10,
                "Z2": 10,
                "Zs": 10,
            }
        else:
            args.dim_info['X1'] = dim_X1
            args.dim_info['X2'] = dim_X2
            args.dim_info['Y']  = dim_Y

        MODEL_NAME = (
            f"microbiome_N{len(common_idx)}"
            f"_X1{dim_X1}_X2{dim_X2}_Y{dim_Y}"
            f"_seed{args.seed}_beta{args.beta}_hsic{args.hsic_weight}"
            f"_layers{args.layers}_embed_dim{args.embed_dim}"
        )
        wandb.config.update(
            {
                "seed": args.seed,
                "model_name": MODEL_NAME,
                "dims": {
                    "X1": dim_X1,
                    "X2": dim_X2,
                    "Y":  dim_Y,
                    "Z1": args.dim_info["Z1"],
                    "Z2": args.dim_info["Z2"],
                    "Zs": args.dim_info["Zs"],
                },
                "gt_weights_table_logged": True,
            },
            allow_val_change=True,
        )
    
    elif args.dataset == "microbiome_synthetic":
        df_X1 = pd.read_csv(
            "/users/antonios/LEAF_revisit/synthetic_microbiome/Bayesian-inference-of-bacteria-metabolite-interactions/X1_bacteria_synthetic_CLR_COPSAC.csv"
        )
        df_X2 = pd.read_csv(
            "/users/antonios/LEAF_revisit/synthetic_microbiome/Bayesian-inference-of-bacteria-metabolite-interactions/X2_viruses_synthetic_CLR_COPSAC.csv"
        )
        df_Y = pd.read_csv(
            "/users/antonios/LEAF_revisit/synthetic_microbiome/Bayesian-inference-of-bacteria-metabolite-interactions/Y_metabolites_log_synthetic_COPSAC.csv"
        )


        # 1) index by sim
        df_X1 = df_X1.set_index("sim")
        df_X2 = df_X2.set_index("sim")
        df_Y  = df_Y.set_index("sim")

        common_idx = (
            df_Y.index
            .intersection(df_X1.index)
            .intersection(df_X2.index)
        )
        common_idx = common_idx.sort_values()

        df_X1 = df_X1.loc[common_idx]
        df_X2 = df_X2.loc[common_idx]
        df_Y  = df_Y.loc[common_idx]

        # 2) drop 'rep', get numpy
        df_X1_feats = df_X1.drop(columns=["rep"])
        df_X2_feats = df_X2.drop(columns=["rep"])
        df_Y_feats  = df_Y.drop(columns=["rep"])

        # Store metabolite names for later use
        metabolite_names = list(df_Y_feats.columns)

        X1_np = df_X1_feats.to_numpy(dtype=np.float32)
        X2_np = df_X2_feats.to_numpy(dtype=np.float32)
        Y_np  = df_Y_feats.to_numpy(dtype=np.float32)

        dim_X1 = X1_np.shape[1]
        dim_X2 = X2_np.shape[1]
        dim_Y  = Y_np.shape[1]
        Y = Y_np

        # 3) load GT variance shares from R
        df_gt = pd.read_csv(
            "/users/antonios/LEAF_revisit/synthetic_microbiome/Bayesian-inference-of-bacteria-metabolite-interactions/GT_virome_variance_shares_final_COPSAC.csv"
        )

        # ensure metabolite ordering in GT matches Y columns
        # df_gt$met is like "met_1", "met_2", ...
        # df_Y_feats.columns should be same strings
        assert list(df_gt["met"]) == list(df_Y_feats.columns), \
            "GT metabolite order does not match Y columns"


        Z1_share    = df_gt["share_Z1"].to_numpy(dtype=np.float32)
        Z2_share    = df_gt["share_Z2"].to_numpy(dtype=np.float32)
        Zs_share    = df_gt["share_Zs"].to_numpy(dtype=np.float32)
        noise_share = df_gt["share_noise"].to_numpy(dtype=np.float32)

        shares = {
            "Z1":    Z1_share,
            "Z2":    Z2_share,
            "Zs":    Zs_share,
            "noise": noise_share,
        }

        gt = {"shares": shares}

        # 4) data and targets for your model (unchanged)
        #data = np.stack([X1_np, X2_np], axis=0)  # (2, N, dim)
        X1_np= X1_np.astype(np.float32)
        X2_np= X2_np.astype(np.float32)
        #data = (X1_np, X2_np)
        targets1 = Y_np[:, 0]
        targets2 = Y_np[:, 1]
        targets3 = Y_np[:, 2]

        # args.dim_info['X1'] = dim_X1
        # args.dim_info['X2'] = dim_X2
        # args.dim_info['Y']  = dim_Y

        if not hasattr(args, "dim_info") or args.dim_info is None:
            # initialize with some defaults for latent dims; adjust if you want different sizes
            args.dim_info = {
                'X1': dim_X1,
                'X2': dim_X2,
                'Y':  dim_Y,
                "Z1": 10,
                "Z2": 10,
                "Zs": 10,
            }
        else:
            args.dim_info['X1'] = dim_X1
            args.dim_info['X2'] = dim_X2
            args.dim_info['Y']  = dim_Y

        MODEL_NAME = (
            f"microbiome_N{len(common_idx)}"
            f"_X1{dim_X1}_X2{dim_X2}_Y{dim_Y}"
            f"_seed{args.seed}_beta{args.beta}_hsic{args.hsic_weight}"
            f"_layers{args.layers}_embed_dim{args.embed_dim}"
        )
        wandb.config.update(
            {
                "seed": args.seed,
                "model_name": MODEL_NAME,
                "dims": {
                    "X1": dim_X1,
                    "X2": dim_X2,
                    "Y":  dim_Y,
                    "Z1": args.dim_info["Z1"],
                    "Z2": args.dim_info["Z2"],
                    "Zs": args.dim_info["Zs"],
                },
                "gt_weights_table_logged": True,
            },
            allow_val_change=True,
        )

    elif args.dataset == "microbiome_synthetic_ilr":
        df_X1 = pd.read_csv(
            "/users/antonios/LEAF_revisit/synthetic_microbiome/Bayesian-inference-of-bacteria-metabolite-interactions/clr_seeds_sparse/ilr_sparse/X1_bacteria_synthetic_ILR_final_sparse.csv"
        )
        df_X2 = pd.read_csv(
            "/users/antonios/LEAF_revisit/synthetic_microbiome/Bayesian-inference-of-bacteria-metabolite-interactions/clr_seeds_sparse/ilr_sparse/X2_viruses_synthetic_ILR_final_sparse.csv"
        )
        df_Y = pd.read_csv(
            "/users/antonios/LEAF_revisit/synthetic_microbiome/Bayesian-inference-of-bacteria-metabolite-interactions/complex_model/sparse/Y_metabolites_log_synthetic_complex_RA_sparse.csv"
        )


        # 1) index by sim
        df_X1 = df_X1.set_index("sim")
        df_X2 = df_X2.set_index("sim")
        df_Y  = df_Y.set_index("sim")

        common_idx = (
            df_Y.index
            .intersection(df_X1.index)
            .intersection(df_X2.index)
        )
        common_idx = common_idx.sort_values()

        df_X1 = df_X1.loc[common_idx]
        df_X2 = df_X2.loc[common_idx]
        df_Y  = df_Y.loc[common_idx]

        # 2) drop 'rep', get numpy
        df_X1_feats = df_X1.drop(columns=["rep"])
        df_X2_feats = df_X2.drop(columns=["rep"])
        df_Y_feats  = df_Y.drop(columns=["rep"])

        X1_np = df_X1_feats.to_numpy(dtype=np.float32)
        X2_np = df_X2_feats.to_numpy(dtype=np.float32)
        Y_np  = df_Y_feats.to_numpy(dtype=np.float32)

        dim_X1 = X1_np.shape[1]
        dim_X2 = X2_np.shape[1]
        dim_Y  = Y_np.shape[1]
        Y = Y_np

        # 3) load GT variance shares from R
        df_gt = pd.read_csv(
            "/users/antonios/LEAF_revisit/synthetic_microbiome/Bayesian-inference-of-bacteria-metabolite-interactions/clr_seeds_sparse/GT_virome_variance_shares_complex_sparse.csv"
        )

        # ensure metabolite ordering in GT matches Y columns
        # df_gt$met is like "met_1", "met_2", ...
        # df_Y_feats.columns should be same strings
        assert list(df_gt["met"]) == list(df_Y_feats.columns), \
            "GT metabolite order does not match Y columns"


        Z1_share    = df_gt["share_Z1"].to_numpy(dtype=np.float32)
        Z2_share    = df_gt["share_Z2"].to_numpy(dtype=np.float32)
        Zs_share    = df_gt["share_Zs"].to_numpy(dtype=np.float32)
        noise_share = df_gt["share_noise"].to_numpy(dtype=np.float32)

        shares = {
            "Z1":    Z1_share,
            "Z2":    Z2_share,
            "Zs":    Zs_share,
            "noise": noise_share,
        }

        gt = {"shares": shares}

        # 4) data and targets for your model (unchanged)
        #data = np.stack([X1_np, X2_np], axis=0)  # (2, N, dim)
        X1_np= X1_np.astype(np.float32)
        X2_np= X2_np.astype(np.float32)
        #data = (X1_np, X2_np)
        targets1 = Y_np[:, 0]
        targets2 = Y_np[:, 1]
        targets3 = Y_np[:, 2]

        # args.dim_info['X1'] = dim_X1
        # args.dim_info['X2'] = dim_X2
        # args.dim_info['Y']  = dim_Y

        if not hasattr(args, "dim_info") or args.dim_info is None:
            # initialize with some defaults for latent dims; adjust if you want different sizes
            args.dim_info = {
                'X1': dim_X1,
                'X2': dim_X2,
                'Y':  dim_Y,
                "Z1": 10,
                "Z2": 10,
                "Zs": 10,
            }
        else:
            args.dim_info['X1'] = dim_X1
            args.dim_info['X2'] = dim_X2
            args.dim_info['Y']  = dim_Y

        MODEL_NAME = (
            f"microbiome_N{len(common_idx)}"
            f"_X1{dim_X1}_X2{dim_X2}_Y{dim_Y}"
            f"_seed{args.seed}_beta{args.beta}_hsic{args.hsic_weight}"
            f"_layers{args.layers}_embed_dim{args.embed_dim}"
        )
        wandb.config.update(
            {
                "seed": args.seed,
                "model_name": MODEL_NAME,
                "dims": {
                    "X1": dim_X1,
                    "X2": dim_X2,
                    "Y":  dim_Y,
                    "Z1": args.dim_info["Z1"],
                    "Z2": args.dim_info["Z2"],
                    "Zs": args.dim_info["Zs"],
                },
                "gt_weights_table_logged": True,
            },
            allow_val_change=True,
        )        
#Data generation
    elif args.dataset == "microbiome":
        #Load the data
        
        print("Loading the data...")
        # --- 1.1  Load CSVs (or any other source)
        def _read_clean(path):
            df = pd.read_csv(path, sep="\t", index_col=0)
            df.index = df.index.astype(str).str.strip()
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
            return df

#       Y values need to be independent with its each other. That is why we are going with log trasnformation
        #df_Y  = pd.read_csv('/users/antonios/code/metabolome.CLR.LEAF.tsv', sep = "\t", index_col = 0)
        #df_X1 = _read_clean('/users/antonios/code/microbiome.CLR.LEAF.tsv')
        #df_X2 = _read_clean('/users/antonios/code/virome.CLR.LEAF.tsv')
        #df_Y  = _read_clean('/users/antonios/data/metabolites.known.tsv')
        df_X1 = _read_clean('/users/antonios/data/MAG.cluster.MGM_embeddings.LEAF.tsv')
        df_X2 = _read_clean('/users/antonios/data/virome.AE_embeddings.REGULARIZED.tsv')
        df_Y  = _read_clean('/users/antonios/data/metabolites.known.tsv')
        debugger("df_X1 (raw)", df_X1)
        debugger("df_X2 (raw)", df_X2)
        debugger("df_Y (raw)",  df_Y)

        common_idx = df_Y.index.intersection(df_X1.index).intersection(df_X2.index)
        common_idx = common_idx.sort_values()

        df_X1 = df_X1.loc[common_idx]
        df_X2 = df_X2.loc[common_idx]
        df_Y  = df_Y.loc[common_idx]

        assert len(df_X1) == len(df_X2) == len(df_Y) , "Row counts must be equal"
        assert df_X1.index.equals(df_X2.index) and df_X1.index.equals(df_Y.index), \
            "Indices must match and be in the same order across X1, X2, Y"

        X1_np = df_X1.to_numpy(dtype=np.float32)
        X2_np = df_X2.to_numpy(dtype=np.float32)
        Y_np  = df_Y.to_numpy(dtype=np.float32)

        assert np.isfinite(X1_np).all(), "X1 contains NaN/inf"
        assert np.isfinite(X2_np).all(), "X2 contains NaN/inf"
        assert np.isfinite(Y_np).all(),  "Y contains NaN/inf"

        debugger("X1_np", X1_np)
        debugger("X2_np", X2_np)
        debugger("Y_np",  Y_np)

        print("Loading the data and preprocessing is finished...")
        print("*" * 15)
        print("Start training")

        metabolite_outcomes = list(df_Y.columns)
        metabolite_names = metabolite_outcomes  # Store for variance table
        print(metabolite_outcomes)
        assert len(metabolite_outcomes) == df_Y.shape[1], "metabolite_names must match Y columns"
                # 2. The targets are the Y matrix.
        # The data contain different number of features.
        # Therefore it is better to use a list to store them 
        #than array
        X1, X2 = X1_np, X2_np
        data = [X1, X2]
        dim_X1 = X1_np.shape[1]
        dim_X2 = X2_np.shape[1]
        Y = Y_np
        dim_Y  = Y.shape[1]  # Y is multi-target (metabolites)
        # dim_C after one-hot (computed later after we build C_train/C_test)

        args.dim_info = {
            "X1": dim_X1, 
            "X2": dim_X2,
            "Y": dim_Y,
        }

        MODEL_NAME = (
            f"MICROBIOME_N{len(common_idx)}"
            f"_X1{dim_X1}_X2{dim_X2}_Y{dim_Y}"
            f"_seed{args.seed}_beta{args.beta}_hsic{args.hsic_weight}"f"_abl_{args.ablation}"
            
        )
        wandb.config.update({
                    "seed": args.seed,
                    "model_name": MODEL_NAME,
                    "dims": args.dim_info
                }, allow_val_change=True)



    elif args.dataset in ["synthetic_linear", "synthetic_nonlinear"]:
        # Synthetic data generation
        grid = make_weight_grid(
            values=(0.0, 0.1, 0.2, 0.3, 0.4),
            max_sum_sq=0.95,
            target_n=100,
            seed=42,
        )
        dim_Y = len(grid)
        W1_vec = np.array([w[0] for w in grid], dtype=float)
        W2_vec = np.array([w[1] for w in grid], dtype=float)
        Ws_vec = np.array([w[2] for w in grid], dtype=float)
        MODEL_NAME = f'{args.dataset}_{args.num_epoch_s1}_{args.dim_info["Zs"]}_{args.dim_info["Z1"]}_{args.dim_info["Z2"]}_{args.dim_info["X1"]}-{args.dim_info["X2"]}_{dim_Y}_3_{args.num_data}_seed{args.seed}_beta{args.beta}_hsic{args.hsic_weight}'
        if args.dataset == "synthetic_linear":
            X1, X2, Y, gt = make_model(
                W1_vec, W2_vec, Ws_vec,
                n=args.num_data,
                dim_Zs=args.dim_info["Zs"], dim_Z1=args.dim_info["Z1"], dim_Z2=args.dim_info["Z2"],
                dim_X1=args.dim_info["X1"], dim_X2=args.dim_info["X2"],  
                dim_Y=dim_Y,
                seed=args.seed,
            )
            
        else:
            X1, X2, Y, C, gt = make_model_non_linear(
                W1_vec, W2_vec, Ws_vec,
                n=args.num_data,
                dim_Zs=args.dim_info["Zs"], dim_Z1=args.dim_info["Z1"], dim_Z2=args.dim_info["Z2"],
                dim_X=args.dim_info["X"], dim_Y=dim_Y,
                seed=args.seed,
            )
        data = np.array([X1, X2])
        targets1 = Y[:, 0]
        targets2 = Y[:, 1]
        targets3 = Y[:, 2]
        #modalities = [X1, X2]         # shapes: (n, d1), (n, d2)
        wandb.config.update({
            "seed": args.seed,
            "model_name": MODEL_NAME,
            "dims": {"Zs": args.dim_info["Zs"], "Z1": args.dim_info["Z1"], "Z2": args.dim_info["Z2"], "X1": args.dim_info["X1"], "Y": dim_Y, "X2":args.dim_info["X2"] },
            "gt_weights_table_logged": True,
        }, allow_val_change=True)
    else:
        # MNIST data generation
        args.dim_info = {'Y': 784, 'Z1': 50, 'Zs': 50, 'X': 784, 'Z2': 50}
        MODEL_NAME = (
            f"mnist_pairs_sharedConf_same{args.prob_same:.2f}"
            f"_alphaS{args.alpha_s}_alphaC{args.alpha_c}_p{args.p_high}-{args.p_low}"
            f"_N{args.num_data}_seed{args.seed}_beta{args.beta}_hsic{args.hsic_weight}"
        )
        data, targets1, targets2, targets3, Y, S_shared = build_mnist_pairs_shared_conf(
            seed=args.seed,
            max_N=args.num_data,
            prob_same=args.prob_same,
            alpha_s=args.alpha_s,
            alpha_c=args.alpha_c,
            patch=args.patch,
            p_high=args.p_high,
            p_low=args.p_low,
            S_high_digits=args.S_high,
            add_style_target=args.add_style_target,
            w1=args.w1,
            ws=args.ws,
            wc=args.wc,
            sigma_y=args.sigma_y
        )
        gt = make_mnist_gt(Y, args.add_style_target, args.w1, args.ws, args.wc, args.sigma_y,
                           z1p_var=1.0, s_var=1.0, c_var=float(C.var()))
        wandb.config.update({
            "seed": args.seed,
            "model_name": MODEL_NAME,
            "dims": {"X1": 784, "X2": 784, "Y_out": int(Y.shape[1])},
            "dataset": "MNIST paired (shared style + confounder)",
        }, allow_val_change=True)

    # Save confounders and targets
    os.makedirs('data/disentangled', exist_ok=True)
    np.save(f'data/disentangled/all_targets_{MODEL_NAME}.npy', Y)



    # # Dataset Initialization 
    # modalities = [X1, X2]
    # dataset = MultiomicDataset(total_data=modalities, total_labels1=Y)

    if args.dataset in {"microbiome","microbiome_synthetic","microbiome_synthetic_ilr","COPSAC_clone"}:
        modalities_raw = (X1_np, X2_np)  # raw CLR inputs

        # Use the new class for multi-omics data (list of modalities, single Y matrix)
        tmp_dataset = MultiomicDataset(total_data=modalities_raw, total_labels1=Y_np)  # 
        num_data = tmp_dataset.num_samples
        print("Number of samples:", num_data)
        print("Multiomic dataset is used")



    #elif args.dataset in {"synthetic_linear", "synthetic_nonlinear", "mnist"}:
        # Use the original class for synthetic data (stacked data, three separate targets)
        # 'data' here is the stacked numpy array with samples in columns
    #    num_data = data.shape[1]  # same as data.shape[1]

        test_idx_path = f"data/splits/test_idx_fixed_{args.dataset}_{args.ablation}_{MODEL_NAME}.npy"
        os.makedirs("data/splits", exist_ok=True)

        if os.path.exists(test_idx_path):
            test_idx = np.load(test_idx_path).astype(int)
        else:
            train_val_tmp, test_tmp = torch.utils.data.random_split(
                tmp_dataset,
                [int(0.9 * num_data), num_data - int(0.9 * num_data)],
                generator=torch.Generator().manual_seed(0),
            )
            test_idx = np.array(test_tmp.indices, dtype=int)
            np.save(test_idx_path, test_idx)

        all_idx = np.arange(num_data, dtype=int)
        train_val_indices = np.setdiff1d(all_idx, test_idx, assume_unique=False)

        rng = np.random.RandomState(0)
        shuffled = train_val_indices.copy()
        rng.shuffle(shuffled)

        n_train = int(0.8 * len(shuffled))
        train_indices = shuffled[:n_train].astype(int)
        val_indices   = shuffled[n_train:].astype(int)



        print("Split sizes:",
            "train", len(train_indices),
            "val", len(val_indices),
            "test", len(test_idx))

        print("Overlap checks:",
            "train∩val", len(set(train_indices) & set(val_indices)),
            "train∩test", len(set(train_indices) & set(test_idx)),
            "val∩test", len(set(val_indices) & set(test_idx)))


        X1_full = X1_np.copy()
        X2_full = X2_np.copy()
        # === Standardize CLR features for X1 and X2 using train indices only ===


        # Bacteria
        X1_mean = X1_full[train_indices].mean(axis=0, keepdims=True)
        X1_std  = X1_full[train_indices].std(axis=0, ddof=0, keepdims=True)
        X1_std[X1_std == 0] = 1.0
        X1_full = (X1_full - X1_mean) / X1_std
        X1_full = X1_full.astype(np.float32)
        # Viruses
        X2_mean = X2_full[train_indices].mean(axis=0, keepdims=True)
        X2_std  = X2_full[train_indices].std(axis=0, ddof=0, keepdims=True)
        X2_std[X2_std == 0] = 1.0
        X2_full = (X2_full - X2_mean) / X2_std
        X2_full = X2_full.astype(np.float32)

        # THIS IS FOR THE MULTIMODAL DATASET
        # # Rebuild data with standardized CLR inputs
        # data = np.stack(
        #     [X1_full.astype(np.float32), X2_full.astype(np.float32)],
        #     axis=0
        # )

        #ΤHIS IS FOR THE MULTIOMICS DATASET ABLATION SETTTING
        if args.ablation == "both":
            modalities = (X1_full, X2_full)
        elif args.ablation == "x1_only":
            modalities = (X1_full, np.zeros_like(X2_full))
        elif args.ablation == "x2_only":
            modalities = (np.zeros_like(X1_full), X2_full)
        else:
            raise ValueError(f"Unknown ablation mode: {args.ablation}")

        # right after MODEL_NAME is constructed and before Y_full = np.load(...)
        np.save(f"data/disentangled/all_targets_{MODEL_NAME}.npy", Y_np)

        # 2) Load and normalize Y using train indices only
        # AFTER:
        Y_full = np.load(f"data/disentangled/all_targets_{MODEL_NAME}.npy").astype(np.float32)
        Y_full = Y_np.astype(np.float32, copy=False)

        # Only log-transform real COPSAC (raw metabolite concentrations)
        # Synthetic data is already in log-space from R generation
        if args.dataset == "microbiome":
            Y_log = np.log1p(Y_full)  # Real data: raw → log
        else:
            Y_log = Y_full  # Synthetic: already log-space

        Y_mean = Y_log[train_indices].mean(axis=0, keepdims=True)
        Y_std  = Y_log[train_indices].std(axis=0, ddof=0, keepdims=True)
        Y_std[Y_std == 0] = 1.0

        Y_final = (Y_log - Y_mean) / Y_std  # shape [num_samples, num_targets]


        # Optional saves
        np.save(f"data/disentangled/all_targets_log_{MODEL_NAME}.npy", Y_final)
        np.save(f"data/disentangled/targets_log_{MODEL_NAME}_train.npy", Y_final[train_indices])
        np.save(f"data/disentangled/targets_log_{MODEL_NAME}_val.npy", Y_final[val_indices])
        #np.save(f"data/disentangled/targets_log_{MODEL_NAME}_test.npy",  Y_final[test_idx])

        # 3) Build the REAL dataset with normalized Y as single label block
        #dataset = MultimodalDataset(data, targets1, targets2, targets3)
        dataset = MultiomicDataset(total_data=modalities, total_labels1=Y_final)


        # 4) Now create subsets from THIS dataset (train, val, test)

        train_dataset = Subset(dataset, train_indices.tolist())
        val_dataset   = Subset(dataset, val_indices.tolist())
        test_dataset  = Subset(dataset, test_idx.tolist())

    else:
        raise ValueError(f"Unknown dataset type: {args.dataset}")





######################################################################## 
########################Synthetic Data Modeling ########################
########################################################################
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=False, batch_size=args.batch_size,
    )
    train_loader_noshuf = DataLoader(
        train_dataset,
        shuffle=False,
        drop_last=False,
        batch_size=args.batch_size,
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, drop_last=False, batch_size=args.batch_size,
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, drop_last=False, batch_size=args.batch_size
    )

    # 5) Train models — these now see log-standardized Y via the dataset
    train_mp(args.beta, train_loader, val_loader, train_dataset, val_dataset, args)  # val used as eval

    print("CALLING train_step2", flush=True)
    logs, disen = train_step2(args, train_loader, val_loader, train_dataset, val_dataset)
    print("RETURNED from train_step2", flush=True)


    # 6) Gather embeddings (unchanged)
    zs1_tr, zs2_tr, zc1_tr, zc2_tr = gather_embeddings(disen, train_loader_noshuf, device=device)
    zs1_val, zs2_val, zc1_val, zc2_val = gather_embeddings(disen, val_loader, device=device)
    zs1_te, zs2_te, zc1_te, zc2_te = gather_embeddings(disen, test_loader, device=device)
    np.save(f'data/disentangled/zs1_tr{MODEL_NAME}.npy', zs1_tr)
    np.save(f'data/disentangled/zs2_tr{MODEL_NAME}.npy', zs2_tr)
    np.save(f'data/disentangled/zc1_tr{MODEL_NAME}.npy', zc1_tr)
    np.save(f'data/disentangled/zc2_tr{MODEL_NAME}.npy', zc2_tr)
    np.save(f'data/disentangled/zs1_test_{MODEL_NAME}.npy', zs1_te)
    np.save(f'data/disentangled/zs2_test_{MODEL_NAME}.npy', zs2_te)
    np.save(f'data/disentangled/zc1_test_{MODEL_NAME}.npy', zc1_te)
    np.save(f'data/disentangled/zc2_test_{MODEL_NAME}.npy', zc2_te)
    np.save(f'data/disentangled/zs1_val_{MODEL_NAME}.npy', zs1_val)
    np.save(f'data/disentangled/zs2_val_{MODEL_NAME}.npy', zs2_val)
    np.save(f'data/disentangled/zc1_val_{MODEL_NAME}.npy', zc1_val)
    np.save(f'data/disentangled/zc2_val_{MODEL_NAME}.npy', zc2_val)

    # ==========================================================================
    # t-SNE VISUALIZATION OF LATENT REPRESENTATIONS
    # ==========================================================================
    print("\n" + "="*70)
    print("GENERATING t-SNE VISUALIZATION OF LATENT SPACES")
    print("="*70)


    # Use test set for visualization and evaluation
    zs1_all = np.vstack([zs1_te])
    zs2_all = np.vstack([zs2_te])
    zc1_all = np.vstack([zc1_te])

    print(f"zs1_all: {zs1_all.shape}, zs2_all: {zs2_all.shape}, zc1_all: {zc1_all.shape}")

    # Check correlations between representations
    print(f"\nCorrelations between latent spaces:")
    print(f"  zs1 vs zs2: {np.corrcoef(zs1_all.flatten(), zs2_all.flatten())[0,1]:.4f}")
    print(f"  zs1 vs zc1: {np.corrcoef(zs1_all.flatten(), zc1_all.flatten())[0,1]:.4f}")
    print(f"  zs2 vs zc1: {np.corrcoef(zs2_all.flatten(), zc1_all.flatten())[0,1]:.4f}")

    # Variance statistics
    print(f"\nVariance per representation:")
    print(f"  zs1 variance: {np.var(zs1_all):.6f}, std: {np.std(zs1_all):.6f}")
    print(f"  zs2 variance: {np.var(zs2_all):.6f}, std: {np.std(zs2_all):.6f}")
    print(f"  zc1 variance: {np.var(zc1_all):.6f}, std: {np.std(zc1_all):.6f}")

    # Combine all for t-SNE
    n_samples = zs1_all.shape[0]
    all_z = np.vstack([zs1_all, zs2_all, zc1_all])
    labels = ['zs1 (specific X1)'] * n_samples + ['zs2 (specific X2)'] * n_samples + ['zc1 (common)'] * n_samples

    print(f"\nRunning t-SNE on {all_z.shape[0]} samples...")
    tsne_start = time_module.time()

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
    z_embedded = tsne.fit_transform(all_z)

    print(f"t-SNE completed in {time_module.time() - tsne_start:.2f} seconds")

    # Plot - Nature style for supplementary
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 7,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 2,
        'ytick.major.size': 2,
    })
    fig, ax = plt.subplots(figsize=(3.5, 3))

    colors = {'zs1 (specific X1)': '#E64B35', 'zs2 (specific X2)': '#4DBBD5', 'zc1 (common)': '#00A087'}
    for label in ['zs1 (specific X1)', 'zs2 (specific X2)', 'zc1 (common)']:
        mask = np.array([l == label for l in labels])
        ax.scatter(z_embedded[mask, 0], z_embedded[mask, 1],
                   c=colors[label], label=label, alpha=0.7, s=15, edgecolors='none')

    ax.legend(fontsize=6, frameon=False, loc='best')
    ax.set_xlabel('t-SNE 1', fontsize=7)
    ax.set_ylabel('t-SNE 2', fontsize=7)
    ax.tick_params(axis='both', labelsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    tsne_plot_path = f'/users/antonios/LEAF_revisit/LEAF/mgm/data/again/final/tsne_latents_{MODEL_NAME}.png'
    os.makedirs(os.path.dirname(tsne_plot_path), exist_ok=True)
    plt.savefig(tsne_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"t-SNE plot saved to: {tsne_plot_path}")
    wandb.log({"tsne_latents": wandb.Image(tsne_plot_path)})
    print("="*70 + "\n")


    # HSIC Independence Testing with Permutation Calibration
    def compute_hsic_with_permutation_test(z_a, z_b, n_permutations=500, alpha=0.05):
        """
        Compute HSIC with permutation null distribution for calibration.

        The raw HSIC value is not inherently meaningful - its scale depends on
        kernel, bandwidth, and sample size. This function calibrates by comparing
        to a permutation null distribution.

        Returns:
            dict with observed HSIC, p-value, percentile, null statistics,
            and whether dependence is significant (above noise floor)
        """
        # Compute observed HSIC
        hsic_metric = HSIC(sigma_x=-1, sigma_y=-1)
        hsic_metric.update((z_a, z_b))
        observed_hsic = float(hsic_metric.compute())

        # Generate permutation null distribution
        permuted_hsics = []
        n_samples = z_b.shape[0]

        for _ in range(n_permutations):
            # Shuffle z_b along batch dimension to break any dependence
            perm_idx = torch.randperm(n_samples)
            z_b_permuted = z_b[perm_idx]

            hsic_perm = HSIC(sigma_x=-1, sigma_y=-1)
            hsic_perm.update((z_a, z_b_permuted))
            permuted_hsics.append(float(hsic_perm.compute()))

        permuted_hsics = np.array(permuted_hsics)

        # Compute statistics
        null_mean = np.mean(permuted_hsics)
        null_std = np.std(permuted_hsics)
        p_value = np.mean(permuted_hsics >= observed_hsic)
        percentile = np.mean(permuted_hsics <= observed_hsic) * 100

        # Decision: significant if above (1-alpha) percentile of null
        threshold = np.percentile(permuted_hsics, (1 - alpha) * 100)
        is_significant = observed_hsic > threshold

        return {
            'observed': observed_hsic,
            'p_value': p_value,
            'percentile': percentile,
            'null_mean': null_mean,
            'null_std': null_std,
            'threshold_95': threshold,
            'is_significant': is_significant,
            'effect_size': (observed_hsic - null_mean) / null_std if null_std > 0 else 0
        }

    print("\n" + "="*70)
    print("HSIC Independence Testing (Permutation Calibrated)")
    print("="*70)
    print("  Note: Raw HSIC values are not comparable across setups.")
    print("  Using permutation null to determine if dependence is above noise floor.")
    print("-"*70)

    # Convert to torch tensors for HSIC
    zs1_tensor = torch.from_numpy(zs1_all).float()
    zs2_tensor = torch.from_numpy(zs2_all).float()
    zc1_tensor = torch.from_numpy(zc1_all).float()

    # Define pairs to test
    pairs = [
        ("zs1", "zs2", zs1_tensor, zs2_tensor, "specific X1 vs specific X2"),
        ("zs1", "zc1", zs1_tensor, zc1_tensor, "specific X1 vs common"),
        ("zs2", "zc1", zs2_tensor, zc1_tensor, "specific X2 vs common"),
    ]

    hsic_results = {}
    for name_a, name_b, z_a, z_b, desc in pairs:
        result = compute_hsic_with_permutation_test(z_a, z_b, n_permutations=500)
        key = f"{name_a}_{name_b}"
        hsic_results[key] = result

        status = "SIGNIFICANT - dependence detected" if result['is_significant'] else "At noise floor - independence achieved"
        print(f"\n  HSIC({name_a}, {name_b}): {desc}")
        print(f"    Observed HSIC:    {result['observed']:.6f}")
        print(f"    Null distribution: {result['null_mean']:.6f} +/- {result['null_std']:.6f}")
        print(f"    95th pctl threshold: {result['threshold_95']:.6f}")
        print(f"    Observed percentile: {result['percentile']:.1f}%")
        print(f"    Effect size (z):  {result['effect_size']:.2f}")
        print(f"    p-value:          {result['p_value']:.4f}")
        print(f"    Status: {status}")

    # Log to wandb - both raw and calibrated metrics
    wandb_hsic_log = {}
    for key, result in hsic_results.items():
        wandb_hsic_log[f"hsic_{key}_observed"] = result['observed']
        wandb_hsic_log[f"hsic_{key}_percentile"] = result['percentile']
        wandb_hsic_log[f"hsic_{key}_effect_size"] = result['effect_size']
        wandb_hsic_log[f"hsic_{key}_p_value"] = result['p_value']
        wandb_hsic_log[f"hsic_{key}_significant"] = int(result['is_significant'])
        wandb_hsic_log[f"hsic_{key}_null_mean"] = result['null_mean']
        wandb_hsic_log[f"hsic_{key}_null_std"] = result['null_std']
    wandb.log(wandb_hsic_log)

    # Save HSIC metrics to CSV file
    hsic_csv_path = "/users/antonios/LEAF_revisit/LEAF/mgm/data/again/final/hsic_independence_metrics.csv"
    hsic_row = {
        "seed": args.seed,
    }
    # Add all metrics for each pair
    for key, result in hsic_results.items():
        hsic_row[f"hsic_{key}_observed"] = result['observed']
        hsic_row[f"hsic_{key}_percentile"] = result['percentile']
        hsic_row[f"hsic_{key}_effect_size"] = result['effect_size']
        hsic_row[f"hsic_{key}_p_value"] = result['p_value']
        hsic_row[f"hsic_{key}_significant"] = int(result['is_significant'])
        hsic_row[f"hsic_{key}_null_mean"] = result['null_mean']

    # Check if file exists to decide whether to write header
    if os.path.exists(hsic_csv_path):
        hsic_df = pd.read_csv(hsic_csv_path)
        hsic_df = pd.concat([hsic_df, pd.DataFrame([hsic_row])], ignore_index=True)
    else:
        hsic_df = pd.DataFrame([hsic_row])

    hsic_df.to_csv(hsic_csv_path, index=False)
    print(f"\n  HSIC metrics saved to: {hsic_csv_path}")

    # Summary
    print("\n" + "-"*70)
    print("  INTERPRETATION GUIDE:")
    print("  - If 'At noise floor': Independence achieved, no need to increase hsic_weight")
    print("  - If 'SIGNIFICANT': Dependence still detectable, may benefit from higher hsic_weight")
    print("  - Effect size > 2: Strong dependence | 1-2: Moderate | < 1: Weak")
    print("="*70 + "\n")
    # 7) For the variance-explained CVAE, directly use the same processed Y (avoid reloading originals)
    Y_final = np.load(f"data/disentangled/all_targets_log_{MODEL_NAME}.npy").astype(np.float32)
    #c_train = torch.from_numpy(train_confounder).float().to(device)
    #c_test  = torch.from_numpy(test_confounder).float().to(device)
  #  Y_all = torch.from_numpy(np.load(f'data/disentangled/all_targets_{MODEL_NAME}.npy')).float().to(device)
    Y_train = torch.from_numpy(Y_final[train_indices]).float().to(device)
    Y_val   = torch.from_numpy(Y_final[val_indices]).float().to(device)
    Y_test  = torch.from_numpy(Y_final[test_idx]).float().to(device)

    # COMMENTED OUT: Use freshly generated representations from memory instead of loading from disk
    # to avoid risk of loading stale/mismatched files from previous runs
    # zs1_tr = torch.from_numpy(np.load(f'data/disentangled/zs1_tr{MODEL_NAME}.npy')).float().to(device)
    # zs2_tr = torch.from_numpy(np.load(f'data/disentangled/zs2_tr{MODEL_NAME}.npy')).float().to(device)
    # zc1_tr = torch.from_numpy(np.load(f'data/disentangled/zc1_tr{MODEL_NAME}.npy')).float().to(device)
    # # zc2 = torch.from_numpy(np.load(f'data/disentangled/zc2_{MODEL_NAME}.npy')).float().to(device)
    # zs1_te = torch.from_numpy(np.load(f'data/disentangled/zs1_test_{MODEL_NAME}.npy')).float().to(device)
    # zs2_te = torch.from_numpy(np.load(f'data/disentangled/zs2_test_{MODEL_NAME}.npy')).float().to(device)
    # zc1_te = torch.from_numpy(np.load(f'data/disentangled/zc1_test_{MODEL_NAME}.npy')).float().to(device)
    # zc2_te = torch.from_numpy(np.load(f'data/disentangled/zc2_test_{MODEL_NAME}.npy')).float().to(device)
    # zs1_val = torch.from_numpy(np.load(f'data/disentangled/zs1_val_{MODEL_NAME}.npy')).float().to(device)
    # zs2_val = torch.from_numpy(np.load(f'data/disentangled/zs2_val_{MODEL_NAME}.npy')).float().to(device)
    # zc1_val = torch.from_numpy(np.load(f'data/disentangled/zc1_val_{MODEL_NAME}.npy')).float().to(device)
    # zc2_val = torch.from_numpy(np.load(f'data/disentangled/zc2_val_{MODEL_NAME}.npy')).float().to(device)

    # Use the freshly generated representations directly (they're already numpy arrays from gather_embeddings)
    zs1_tr = torch.from_numpy(zs1_tr).float().to(device)
    zs2_tr = torch.from_numpy(zs2_tr).float().to(device)
    zc1_tr = torch.from_numpy(zc1_tr).float().to(device)
    zc2_tr = torch.from_numpy(zc2_tr).float().to(device)
    zs1_te = torch.from_numpy(zs1_te).float().to(device)
    zs2_te = torch.from_numpy(zs2_te).float().to(device)
    zc1_te = torch.from_numpy(zc1_te).float().to(device)
    zc2_te = torch.from_numpy(zc2_te).float().to(device)
    zs1_val = torch.from_numpy(zs1_val).float().to(device)
    zs2_val = torch.from_numpy(zs2_val).float().to(device)
    zc1_val = torch.from_numpy(zc1_val).float().to(device)
    zc2_val = torch.from_numpy(zc2_val).float().to(device)

    # Residualize confounder latents (optional; here we DO residualize zc1/zc2; keep z1/zs2 as-is)
   # zc1_res_train, zc2_res_train = residualize_data(c_train, *[zc1, zc2], config=config, device=device)
    #zs1_res_train, zs2_res_train = zs1, zs2
   # zc1_res_test, zc2_res_test = residualize_data(c_test, *[zc1_test, zc2_test], config=config, device=device)
    #zs1_res_test, zs2_res_test = zs1_test, zs2_test
    print("Y_train:", Y_train.shape)
    print("zs1_tr:", zs1_tr.shape)
    print("zs2_tr:", zs2_tr.shape)
    print("zc1_tr:", zc1_tr.shape)



    # Build loaders for the variance explained CVAE
    dataset_train = TensorDataset(
        Y_train,
        zs1_tr.to(device),
        zs2_tr.to(device),
        zc1_tr.to(device),
        #c_train.to(device),
    )
    data_loader_train = DataLoader(dataset_train, shuffle=True,  batch_size=config.batch_size)
    data_loader_train_eval = DataLoader(dataset_train, shuffle=False, batch_size=config.batch_size)

    dataset_val = TensorDataset(
        Y_val,
        zs1_val.to(device),
        zs2_val.to(device),
        zc1_val.to(device),
        #c_val.to(device),
    )
    data_loader_val = DataLoader(dataset_val, shuffle=False, batch_size=config.batch_size)

    dataset_test = TensorDataset(
        Y_test,
        zs1_te.to(device),
        zs2_te.to(device),
        zc1_te.to(device),
         #c_test.to(device),
     )
    data_loader_test = DataLoader(dataset_test, shuffle=False, batch_size=config.batch_size)

    # Fit variance explained model
    data_dim = Y_train.shape[1]
    hidden_dim = 50  # Reduced from 200 to prevent overfitting
    #z_dim = zs1.shape[1]
    n_covariates = 2
    num_samples = 177  # This is just dataset size, not grid size

    z_dim_zs1 = zs1_tr.shape[1]
    z_dim_zs2 = zs2_tr.shape[1]
    z_dim_zc1 = zc1_tr.shape[1]

    def print_covariance(a, b, name_a, name_b):
        a0 = a - a.mean(dim=0, keepdim=True)
        b0 = b - b.mean(dim=0, keepdim=True)

        cov = (a0.T @ b0) / (a.shape[0] - 1)  # [da, db]

        print(f"\nCovariance {name_a} vs {name_b}:")
        print(f"  mean |cov|: {cov.abs().mean().item():.6f}")
        print(f"  max  |cov|: {cov.abs().max().item():.6f}")


    print_covariance(zs1_tr, zs2_tr, "z1", "z2")
    print_covariance(zs1_tr, zc1_tr, "z1", "zc")
    print_covariance(zs2_tr, zc1_tr, "z2", "zc")

    # Diagnostic: Check latent statistics
    print("\n=== Latent Statistics ===")
    print(f"zs1_tr: min={zs1_tr.min().item():.3f}, max={zs1_tr.max().item():.3f}, "
        f"mean={zs1_tr.mean().item():.3f}, std={zs1_tr.std().item():.3f}")
    print(f"zs2_tr: min={zs2_tr.min().item():.3f}, max={zs2_tr.max().item():.3f}, "
        f"mean={zs2_tr.mean().item():.3f}, std={zs2_tr.std().item():.3f}")
    print(f"zc1_tr: min={zc1_tr.min().item():.3f}, max={zc1_tr.max().item():.3f}, "
        f"mean={zc1_tr.mean().item():.3f}, std={zc1_tr.std().item():.3f}")

    # ==========================================================================
    # LATENT DISTRIBUTION SHIFT DIAGNOSTIC
    # ==========================================================================
    print("\n" + "="*70)
    print("LATENT DISTRIBUTION SHIFT DIAGNOSTIC (Train vs Val)")
    print("="*70)


    def compute_distribution_shift(z_tr, z_val, name):
        """Compare latent distributions between train and val splits."""
        z_tr_np = z_tr.cpu().numpy()
        z_val_np = z_val.cpu().numpy()

        print(f"\n--- {name} ---")
        print(f"  {'Split':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"  {'Train':<8} {z_tr_np.mean():>8.3f} {z_tr_np.std():>8.3f} {z_tr_np.min():>8.3f} {z_tr_np.max():>8.3f}")
        print(f"  {'Val':<8} {z_val_np.mean():>8.3f} {z_val_np.std():>8.3f} {z_val_np.min():>8.3f} {z_val_np.max():>8.3f}")

        # Per-dimension KS test and Wasserstein distance
        n_dims = z_tr_np.shape[1]
        ks_train_val = []
        ws_train_val = []

        for d in range(n_dims):
            # KS test (p-value < 0.05 indicates significant distribution shift)
            ks_tv = ks_2samp(z_tr_np[:, d], z_val_np[:, d])
            ks_train_val.append(ks_tv.pvalue)

            # Wasserstein distance (Earth Mover's Distance)
            ws_tv = wasserstein_distance(z_tr_np[:, d], z_val_np[:, d])
            ws_train_val.append(ws_tv)

        ks_train_val = np.array(ks_train_val)
        ws_train_val = np.array(ws_train_val)

        print(f"\n  KS test p-values (< 0.05 = significant shift):")
        print(f"    Train vs Val:  min={ks_train_val.min():.4f}, mean={ks_train_val.mean():.4f}, #sig={np.sum(ks_train_val < 0.05)}/{n_dims}")

        print(f"\n  Wasserstein distance (higher = more shift):")
        print(f"    Train vs Val:  mean={ws_train_val.mean():.4f}, max={ws_train_val.max():.4f}")

        # Flag if there's significant shift
        if np.sum(ks_train_val < 0.05) > n_dims * 0.2:
            print(f"  ⚠️  WARNING: Significant distribution shift detected in {name} (Train vs Val)!")

        return {
            'name': name,
            'ks_train_val': ks_train_val,
            'ws_train_val': ws_train_val,
        }

    shift_stats = {}
    shift_stats['zs1'] = compute_distribution_shift(zs1_tr, zs1_val, 'zs1 (View 1 specific)')
    shift_stats['zs2'] = compute_distribution_shift(zs2_tr, zs2_val, 'zs2 (View 2 specific)')
    shift_stats['zc1'] = compute_distribution_shift(zc1_tr, zc1_val, 'zc1 (Common/shared)')

    print("\n" + "="*70)

   # lim_val = 2.0
    #grid_z = (torch.rand(num_samples, z_dim_zs1, device=device) * 2 * lim_val) - lim_val
    #grid_cov = lambda x: (torch.rand(num_samples, x.shape[1], device=device) * 2 * lim_val) - lim_val
    #grid_c = [grid_cov(x) for x in (zs2, zc1)]

        # Use fixed prior grids (standard normal) to prevent overfitting
    # Grid size should be large enough for proper integration
    lim_val = 0.25  


    def create_empirical_grid(z_data, K, jitter=0.0, clamp_to_data_range=False):
        """
        Sample a grid from the empirical distribution of z_data (with optional Gaussian jitter).

        Args:
            z_data: Tensor [N, D]
            K: number of grid points
            jitter: std of Gaussian noise added to sampled points (0.0 = no noise)
            clamp_to_data_range: if True, clamp jittered samples to [min(z_data), max(z_data)] per-dimension

        Returns:
            grid: Tensor [K, D] detached
        """
        assert z_data.dim() == 2, f"z_data must be [N, D], got {tuple(z_data.shape)}"
        N, D = z_data.shape

        # Sample indices with replacement from empirical distribution
        idx = torch.randint(0, N, (K,), device=z_data.device)
        grid = z_data[idx].clone()

        # Optional jitter to avoid duplicates / add local coverage
        if jitter and jitter > 0.0:
            grid = grid + jitter * torch.randn_like(grid)

            if clamp_to_data_range:
                z_min = z_data.min(dim=0)[0]
                z_max = z_data.max(dim=0)[0]
                grid = torch.max(torch.min(grid, z_max), z_min)

        return grid.detach()


    # --- replace your uniform grids with empirical ones ---

    K = 1000  # or whatever you already use

    grid_z = create_empirical_grid(zs1_tr, K, jitter=0.01, clamp_to_data_range=False)

    grid_c = [
        create_empirical_grid(zs2_tr, K, jitter=0.01, clamp_to_data_range=False),
        create_empirical_grid(zc1_tr, K, jitter=0.01, clamp_to_data_range=False),
    ]


    # --- optional diagnostics (same style as yours) ---
    print("\n=== Empirical Grid Diagnostics ===")
    print(f"grid_z range: [{grid_z.min().item():.3f}, {grid_z.max().item():.3f}]")
    print(f"zs1_tr range: [{zs1_tr.min().item():.3f}, {zs1_tr.max().item():.3f}]")
    print(f"grid_c[0] range: [{grid_c[0].min().item():.3f}, {grid_c[0].max().item():.3f}]")
    print(f"zs2_tr range: [{zs2_tr.min().item():.3f}, {zs2_tr.max().item():.3f}]")
    print(f"grid_c[1] range: [{grid_c[1].min().item():.3f}, {grid_c[1].max().item():.3f}]")
    print(f"zc1_tr range: [{zc1_tr.min().item():.3f}, {zc1_tr.max().item():.3f}]")

    # Check overlap
    def compute_coverage(data, grid):
        data_min, data_max = data.min().item(), data.max().item()
        grid_min, grid_max = grid.min().item(), grid.max().item()
        coverage = (grid_min <= data_min) and (grid_max >= data_max)
        return coverage, (data_max - data_min) / (grid_max - grid_min)

    cov_z, ratio_z = compute_coverage(zs1_tr, grid_z)
    print(f"grid_z covers zs1_tr: {cov_z}, data/grid ratio: {ratio_z:.3f}")



    assert Y_train.shape[0] == zs1_tr.shape[0] == zs2_tr.shape[0] == zc1_tr.shape[0]
    assert Y_val.shape[0]   == zs1_val.shape[0]
  #  assert Y_test.shape[0]  == zs1_te.shape[0]


    decoder_z = nn.Sequential(
        nn.Linear(z_dim_zs1, hidden_dim),
        nn.Tanh(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_dim, data_dim)
    )
    encoder_mapping = nn.Sequential()
    encoders = nn.ModuleList([
        cEncoder(z_dim=z_dim_zs2, mapping=nn.Sequential()),  # for zs2
        cEncoder(z_dim=z_dim_zc1, mapping=nn.Sequential())   # for zc1
    ])

# --- unchanged logic, but now explicit about input dims for each decoder_c ---
    decoders_c = [nn.Sequential(
        nn.Linear(x.shape[1], hidden_dim),
        nn.Tanh(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_dim, data_dim)
    ) for x in (zs2_tr, zc1_tr)]

    decoders_cz = []
    decoder = Decoder_multiple_latents(
        data_dim, n_covariates,
        grid_z, grid_c,
        decoder_z, decoders_c, decoders_cz,
        has_feature_level_sparsity=False, p1=0.1, p2=0.1, p3=0.1,
        lambda0=args.lambda0, penalty_type=args.penalty_type,
        device=device
    )
    model = CVAE_multiple_latent_spaces_with_covariates(encoders, decoder, lr=args.decoder_lr, device=device)
    loss_values, integrals, integrals_dict, overfit_history = model.optimize(
                        data_loader_train,
                        augmented_lagrangian_lr=0.01,
                        n_iter=config.iters,
                        verbose=True,
                        val_loader=data_loader_val,
                        train_eval_loader=data_loader_train_eval,  # For computing train varexp without shuffle
                        val_check_freq=100,
                    )
    cz_dx2_traces = None

    # ====================================================================
    # OVERFITTING ANALYSIS: Plot and save train/val gap over iterations
    # ====================================================================
    if len(overfit_history['iterations']) > 0:
        print("\n" + "="*70)
        print("OVERFITTING ANALYSIS: Train vs Val over iterations")
        print("="*70)


        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Train vs Val Negative Log-Likelihood
        ax1 = axes[0, 0]
        ax1.plot(overfit_history['iterations'], overfit_history['train_neg_loglik'], 'b-', label='Train NLL', linewidth=2)
        ax1.plot(overfit_history['iterations'], overfit_history['val_neg_loglik'], 'r-', label='Val NLL', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Negative Log-Likelihood')
        ax1.set_title('Train vs Val Negative Log-Likelihood')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Overfitting Ratio (val/train)
        ax2 = axes[0, 1]
        ax2.plot(overfit_history['iterations'], overfit_history['overfit_ratio'], 'g-', linewidth=2)
        ax2.axhline(y=1.0, color='k', linestyle='--', label='No overfit (ratio=1)')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Val NLL / Train NLL')
        ax2.set_title('Overfitting Ratio (>1 = overfitting)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Overfitting Difference (val - train)
        ax3 = axes[1, 0]
        ax3.plot(overfit_history['iterations'], overfit_history['overfit_diff'], 'm-', linewidth=2)
        ax3.axhline(y=0.0, color='k', linestyle='--', label='No overfit (diff=0)')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Val NLL - Train NLL')
        ax3.set_title('Overfitting Gap (>0 = overfitting)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Train vs Val Total Loss
        ax4 = axes[1, 1]
        ax4.plot(overfit_history['iterations'], overfit_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax4.plot(overfit_history['iterations'], overfit_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Total Loss')
        ax4.set_title('Train vs Val Total Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        overfit_plot_path = f"/users/antonios/LEAF_revisit/LEAF/mgm/data/again/overfit_analysis_{MODEL_NAME}_seed{args.seed}.png"
        os.makedirs(os.path.dirname(overfit_plot_path), exist_ok=True)
        plt.savefig(overfit_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved overfitting plot to: {overfit_plot_path}")

        # Save overfit history as CSV for further analysis
        overfit_csv_path = f"/users/antonios/LEAF_revisit/LEAF/mgm/data/again/overfit_history_{MODEL_NAME}_seed{args.seed}.csv"
        overfit_df = pd.DataFrame(overfit_history)
        overfit_df.to_csv(overfit_csv_path, index=False)
        print(f"Saved overfitting history to: {overfit_csv_path}")

        # Print summary statistics
        print(f"\nOverfitting Summary:")
        print(f"  First iteration logged: {overfit_history['iterations'][0]}")
        print(f"  Last iteration logged:  {overfit_history['iterations'][-1]}")
        print(f"  Initial overfit ratio:  {overfit_history['overfit_ratio'][0]:.4f}")
        print(f"  Final overfit ratio:    {overfit_history['overfit_ratio'][-1]:.4f}")
        print(f"  Max overfit ratio:      {np.max(overfit_history['overfit_ratio']):.4f} at iter {overfit_history['iterations'][np.argmax(overfit_history['overfit_ratio'])]}")
        print(f"  Min overfit ratio:      {np.min(overfit_history['overfit_ratio']):.4f} at iter {overfit_history['iterations'][np.argmin(overfit_history['overfit_ratio'])]}")

        # Find when overfitting starts (ratio consistently > 1.1)
        overfit_threshold = 1.1
        overfit_mask = overfit_history['overfit_ratio'] > overfit_threshold
        if overfit_mask.any():
            first_overfit_idx = np.argmax(overfit_mask)
            print(f"  Overfitting starts (ratio > {overfit_threshold}) at iteration: {overfit_history['iterations'][first_overfit_idx]}")
        else:
            print(f"  No significant overfitting detected (ratio never > {overfit_threshold})")

        print("="*70)

    # ====================================================================
    # COMPREHENSIVE DIAGNOSTICS
    # ====================================================================
    print("\n" + "="*70)
    print("COMPREHENSIVE POST-TRAINING DIAGNOSTICS")
    print("="*70)

    decoder.eval()

    # 1. Check latent INPUT correlations (should be near zero from upstream encoder)
    print("\n### 1. LATENT INPUT CORRELATIONS (from upstream encoder) ###")
    def compute_corr_matrix(tensors, names):
        """Compute correlation matrix between multiple tensors"""
        all_data = torch.cat([t.reshape(t.shape[0], -1).mean(dim=1, keepdim=True) for t in tensors], dim=1)
        corr_matrix = torch.corrcoef(all_data.T)
        print("\nCorrelation Matrix (averaged over dimensions):")
        print("       ", "  ".join(f"{n:>6}" for n in names))
        for i, name in enumerate(names):
            row_str = f"{name:>6}: " + "  ".join(f"{corr_matrix[i,j].item():>6.3f}" for j in range(len(names)))
            print(row_str)
        return corr_matrix

    input_corr = compute_corr_matrix(
        [zs1_tr, zs2_tr, zc1_tr],
        ["zs1", "zs2", "zc1"]
    )

    # 2. Check decoder OUTPUT correlations (should be near zero for identifiability)
    print("\n### 2. DECODER OUTPUT CORRELATIONS ###")
    with torch.no_grad():
        f_z1 = decoder.forward_z(zs1_tr)
        f_cs = decoder.forward_c([zs2_tr, zc1_tr])
        f_z2 = f_cs[0]
        f_zc = f_cs[1]

    output_corr = compute_corr_matrix(
        [f_z1, f_z2, f_zc],
        ["f_z1", "f_z2", "f_zc"]
    )

    # 3. Detailed per-metabolite correlations
    print("\n### 3. PER-METABOLITE OUTPUT CORRELATIONS ###")
    with torch.no_grad():
        # Compute correlations for each metabolite separately
        corr_z1_z2_per_met = []
        corr_z1_zc_per_met = []
        corr_z2_zc_per_met = []

        for d in range(min(5, data_dim)):  # Show first 5 metabolites
            corr_z1_z2 = torch.corrcoef(torch.stack([f_z1[:, d], f_z2[:, d]]))[0, 1]
            corr_z1_zc = torch.corrcoef(torch.stack([f_z1[:, d], f_zc[:, d]]))[0, 1]
            corr_z2_zc = torch.corrcoef(torch.stack([f_z2[:, d], f_zc[:, d]]))[0, 1]

            corr_z1_z2_per_met.append(corr_z1_z2.item())
            corr_z1_zc_per_met.append(corr_z1_zc.item())
            corr_z2_zc_per_met.append(corr_z2_zc.item())

            print(f"Metabolite {d}: corr(f_z1, f_z2)={corr_z1_z2:.4f}, "
                  f"corr(f_z1, f_zc)={corr_z1_zc:.4f}, corr(f_z2, f_zc)={corr_z2_zc:.4f}")

    print(f"\nAverage correlations across metabolites:")
    print(f"  |corr(f_z1, f_z2)|: {np.mean(np.abs(corr_z1_z2_per_met)):.4f}")
    print(f"  |corr(f_z1, f_zc)|: {np.mean(np.abs(corr_z1_zc_per_met)):.4f}")
    print(f"  |corr(f_z2, f_zc)|: {np.mean(np.abs(corr_z2_zc_per_met)):.4f}")

    # 4. Check if decoder is learning meaningful functions
    print("\n### 4. DECODER OUTPUT STATISTICS ###")
    print(f"f_z1: mean={f_z1.mean():.4f}, std={f_z1.std():.4f}, "
          f"range=[{f_z1.min():.4f}, {f_z1.max():.4f}]")
    print(f"f_z2: mean={f_z2.mean():.4f}, std={f_z2.std():.4f}, "
          f"range=[{f_z2.min():.4f}, {f_z2.max():.4f}]")
    print(f"f_zc: mean={f_zc.mean():.4f}, std={f_zc.std():.4f}, "
          f"range=[{f_zc.min():.4f}, {f_zc.max():.4f}]")

    # 5. Integral convergence
    print("\n### 5. INTEGRAL CONVERGENCE ###")
    int_z_final, int_c_final, int_cz_final = decoder.calculate_integrals_numpy()
    print(f"Final int_z_mean_abs: {np.abs(int_z_final).mean():.6f}")
    print(f"Final int_c0_mean_abs: {np.abs(int_c_final[0]).mean():.6f}")
    print(f"Final int_c1_mean_abs: {np.abs(int_c_final[1]).mean():.6f}")

    # ============================================================================
    # Plot integrals (int_z, int_c) across iterations (mean only) - Nature style
    # ============================================================================
    print("\n### 5b. PLOTTING INTEGRAL CONSTRAINTS ACROSS ITERATIONS ###")

    iterations = integrals_dict['iterations']
    int_z_hist = integrals_dict['int_z']      # [n_steps, n_z_components, output_dim]
    int_c_hist = integrals_dict['int_c']      # [n_steps, n_covariates, output_dim]

    # Compute mean absolute value across output dimensions for each timestep
    int_z_mean = np.abs(int_z_hist).mean(axis=(1, 2))
    int_c_mean = np.abs(int_c_hist).mean(axis=(1, 2))

    # Nature-style plot settings
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))  # Nature single column ~89mm, double ~183mm

    # Panel a: int_z
    axes[0].plot(iterations, int_z_mean, color='#1f77b4', linewidth=1.2)
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel(r'Mean integrals values (f_z) ')
    axes[0].set_yscale('log')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].text(-0.15, 1.05, 'a', transform=axes[0].transAxes, fontsize=12, fontweight='bold', va='top')

    # Panel b: int_c
    axes[1].plot(iterations, int_c_mean, color='#2ca02c', linewidth=1.2)
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel(r'Mean integral values (f_c)')
    axes[1].set_yscale('log')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].text(-0.15, 1.05, 'b', transform=axes[1].transAxes, fontsize=12, fontweight='bold', va='top')

    plt.tight_layout()

    integrals_plot_dir = "/users/antonios/LEAF_revisit/LEAF/mgm/data/again"
    os.makedirs(integrals_plot_dir, exist_ok=True)
    integrals_plot_path = os.path.join(integrals_plot_dir, f"integrals_across_iterations_seed{args.seed}.pdf")
    plt.savefig(integrals_plot_path, dpi=300, bbox_inches='tight')
    integrals_plot_path_png = os.path.join(integrals_plot_dir, f"integrals_across_iterations_seed{args.seed}.png")
    plt.savefig(integrals_plot_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved integrals plot to: {integrals_plot_path}")

    # Save raw integral history for later plotting adjustments
    integrals_history_path = os.path.join(integrals_plot_dir, f"integrals_history_seed{args.seed}.npz")
    np.savez(integrals_history_path,
             iterations=integrals_dict['iterations'],
             int_z=integrals_dict['int_z'],
             int_c=integrals_dict['int_c'],
             int_cz=integrals_dict['int_cz'])
    print(f"Saved integrals history to: {integrals_history_path}")

    # 6. Variance decomposition preview
    print("\n### 6. VARIANCE DECOMPOSITION (training set) ###")
    y_hat_full = f_z1 + f_z2 + f_zc
    residual = Y_train.to(device) - y_hat_full

    var_z1 = f_z1.var(dim=0).mean()
    var_z2 = f_z2.var(dim=0).mean()
    var_zc = f_zc.var(dim=0).mean()
    var_res = residual.var(dim=0).mean()
    var_total = Y_train.var(dim=0).mean()

    print(f"Var(f_z1):  {var_z1:.6f} ({var_z1/var_total*100:.2f}%)")
    print(f"Var(f_z2):  {var_z2:.6f} ({var_z2/var_total*100:.2f}%)")
    print(f"Var(f_zc):  {var_zc:.6f} ({var_zc/var_total*100:.2f}%)")
    print(f"Var(resid): {var_res:.6f} ({var_res/var_total*100:.2f}%)")
    print(f"Sum:        {(var_z1+var_z2+var_zc+var_res)/var_total*100:.2f}%")
    print(f"Var(Y):     {var_total:.6f} (100%)")

    print("\n" + "="*70)
    print("END OF DIAGNOSTICS")
    print("="*70 + "\n")

        # ------------------------------------------------------------------
    # Switch to evaluation mode BEFORE computing errors / variance
    # ------------------------------------------------------------------

    # decoder.eval()
    # model.eval()
    # decoder.eval()  # Already called above

    with torch.no_grad():
        f_z1 = decoder.forward_z(zs1_tr)                  # [N, D]
        f_cs = decoder.forward_c([zs2_tr, zc1_tr])         # list of [N, D]
        f_z2 = f_cs[0]
        f_zc = f_cs[1]

        yhat = f_z1 + f_z2 + f_zc
        eps  = (Y_train.to(device) - yhat)               # residual noise, shape [N, D]


    def print_covcorr_outputs(a, b, name_a, name_b, eps_floor=1e-8):
        # a,b: [N, D]
        a0 = a - a.mean(dim=0, keepdim=True)
        b0 = b - b.mean(dim=0, keepdim=True)

        cov = (a0 * b0).mean(dim=0)  # ddof=0, per-feature cov [D]

        a_std = a0.std(dim=0, unbiased=False).clamp(min=eps_floor)
        b_std = b0.std(dim=0, unbiased=False).clamp(min=eps_floor)
        corr = cov / (a_std * b_std)

        print(f"\n{name_a} vs {name_b}:")
        print(f"  mean |cov| : {cov.abs().mean().item():.6f}")
        print(f"  max  |cov| : {cov.abs().max().item():.6f}")
        print(f"  mean |corr|: {corr.abs().mean().item():.6f}")
        print(f"  max  |corr|: {corr.abs().max().item():.6f}")

    # Independence of residual from components (second-order check)
    print_covcorr_outputs(f_z1, eps, "f_z1", "eps")
    print_covcorr_outputs(f_z2, eps, "f_z2", "eps")
    print_covcorr_outputs(f_zc, eps, "f_zc", "eps")

    # Also check residual vs full prediction (important)
    print_covcorr_outputs(yhat, eps, "yhat", "eps")


    def eval_split(decoder, data_loader, n_covariates=2):
        with torch.no_grad():
            Y_err_list, Y_pred_list = [], []
            for batch in data_loader:
                Y_b, zs1_b, zs2_b, zc1_b = batch
                confs = [zs2_b, zc1_b]
                Y_pred = decoder(zs1_b, confs)
                Y_err  = Y_b - Y_pred
                Y_err_list.append(Y_err)
                Y_pred_list.append(Y_pred)
            Y_err  = torch.cat(Y_err_list,  dim=0)
            Y_pred = torch.cat(Y_pred_list, dim=0)
        return Y_err, Y_pred



    Y_error_tr, Y_pred_tr = eval_split(decoder, data_loader_train_eval, n_covariates=2)
    confounders_tr = [zs2_tr.to(device), zc1_tr.to(device)]
    varexp_tr = decoder.fraction_of_variance_explained(
        zs1_tr.to(device), confounders_tr, Y_error=Y_error_tr.to(device), divide_by_total_var=False
    ).cpu()

    Y_error_val, Y_pred_val = eval_split(decoder, data_loader_val, n_covariates=2)
    confounders_val = [zs2_val.to(device), zc1_val.to(device)]
    varexp_val = decoder.fraction_of_variance_explained(
        zs1_val.to(device), confounders_val, Y_error=Y_error_val.to(device), divide_by_total_var=False
    ).cpu()

    Y_error_te, Y_pred_te = eval_split(decoder, data_loader_test, n_covariates=2)
    confounders_te = [zs2_te.to(device), zc1_te.to(device)]
    varexp_te = decoder.fraction_of_variance_explained(
        zs1_te.to(device), confounders_te, Y_error=Y_error_te.to(device), divide_by_total_var=False
    ).cpu()

    # ---- OVERFITTING DIAGNOSTICS: Train vs Val variance explained gap ----
    print("\n" + "="*60)
    print("OVERFITTING DIAGNOSTICS: Train vs Val Variance Explained Gap")
    print("="*60)

    # varexp shape: (n_features, n_components)
    # Components depend on decoder configuration:
    # - With interactions: [z, c0, c1, int0, int1, noise] (6 cols)
    # - Without interactions: [z, c0, c1, zeros0, zeros1, noise] (6 cols)
    # Note: noise is only present if Y_error was passed to fraction_of_variance_explained

    varexp_tr_mean = varexp_tr.mean(dim=0).numpy()  # mean across features
    varexp_val_mean = varexp_val.mean(dim=0).numpy()
    varexp_te_mean = varexp_te.mean(dim=0).numpy()

    n_cols = varexp_tr_mean.shape[0]
    print(f"\nvarexp shape: {varexp_tr.shape} ({n_cols} components)")

    # Build component names dynamically based on decoder config
    n_covariates = decoder.n_covariates
    n_interactions = decoder.n_covariates_interactions

    component_names = ["z (microbiome)"]
    for j in range(n_covariates):
        component_names.append(f"c{j}")
    for j in range(n_covariates):  # interactions or zeros placeholders
        if n_interactions > 0:
            component_names.append(f"int_z_c{j}")
        else:
            component_names.append(f"(unused int{j})")
    component_names.append("noise/residual")  # last column if Y_error provided

    # Truncate to actual number of columns
    component_names = component_names[:n_cols]

    print(f"\nMean Variance Explained per component:")
    print(f"{'Component':<20} {'Train':<12} {'Val':<12} {'Test':<12} {'Val-Train Gap':<15} {'Val/Train Ratio':<15}")
    print("-" * 86)

    for i in range(n_cols):
        tr_val = varexp_tr_mean[i]
        val_val = varexp_val_mean[i]
        te_val = varexp_te_mean[i]
        gap = val_val - tr_val
        ratio = val_val / (tr_val + 1e-8)
        name = component_names[i] if i < len(component_names) else f"col{i}"
        print(f"{name:<20} {tr_val:<12.4f} {val_val:<12.4f} {te_val:<12.4f} {gap:<15.4f} {ratio:<15.4f}")

    # Identify meaningful component indices (skip zero placeholders)
    # z is always index 0, covariates are 1 to n_covariates, noise is last
    idx_z = 0
    idx_covariates = list(range(1, 1 + n_covariates))
    idx_noise = n_cols - 1 if n_cols > (1 + 2*n_covariates) else None  # noise is last if Y_error was provided

    # Sum of meaningful variance (z + covariates, excluding interaction placeholders and noise)
    meaningful_indices = [idx_z] + idx_covariates
    total_varexp_tr = sum(varexp_tr_mean[i] for i in meaningful_indices)
    total_varexp_val = sum(varexp_val_mean[i] for i in meaningful_indices)
    total_varexp_te = sum(varexp_te_mean[i] for i in meaningful_indices)

    # Log to wandb
    log_dict = {
        "varexp/train_z_mean": varexp_tr_mean[idx_z],
        "varexp/val_z_mean": varexp_val_mean[idx_z],
        "varexp/test_z_mean": varexp_te_mean[idx_z],
        "varexp/gap_val_train_z": varexp_val_mean[idx_z] - varexp_tr_mean[idx_z],
        "varexp/ratio_val_train_z": varexp_val_mean[idx_z] / (varexp_tr_mean[idx_z] + 1e-8),
    }

    for j, idx in enumerate(idx_covariates):
        log_dict[f"varexp/train_c{j}_mean"] = varexp_tr_mean[idx]
        log_dict[f"varexp/val_c{j}_mean"] = varexp_val_mean[idx]
        log_dict[f"varexp/test_c{j}_mean"] = varexp_te_mean[idx]
        log_dict[f"varexp/gap_val_train_c{j}"] = varexp_val_mean[idx] - varexp_tr_mean[idx]
        log_dict[f"varexp/ratio_val_train_c{j}"] = varexp_val_mean[idx] / (varexp_tr_mean[idx] + 1e-8)

    if idx_noise is not None:
        log_dict["varexp/train_noise_mean"] = varexp_tr_mean[idx_noise]
        log_dict["varexp/val_noise_mean"] = varexp_val_mean[idx_noise]
        log_dict["varexp/test_noise_mean"] = varexp_te_mean[idx_noise]
        log_dict["varexp/gap_val_train_noise"] = varexp_val_mean[idx_noise] - varexp_tr_mean[idx_noise]
        log_dict["varexp/ratio_val_train_noise"] = varexp_val_mean[idx_noise] / (varexp_tr_mean[idx_noise] + 1e-8)

    log_dict["varexp/total_train"] = total_varexp_tr
    log_dict["varexp/total_val"] = total_varexp_val
    log_dict["varexp/total_test"] = total_varexp_te
    log_dict["varexp/total_gap_val_train"] = total_varexp_val - total_varexp_tr
    log_dict["varexp/total_ratio_val_train"] = total_varexp_val / (total_varexp_tr + 1e-8)

    wandb.log(log_dict)

    print(f"\n{'Total (z + covs)':<20} {total_varexp_tr:<12.4f} {total_varexp_val:<12.4f} {total_varexp_te:<12.4f} {total_varexp_val - total_varexp_tr:<15.4f} {total_varexp_val / (total_varexp_tr + 1e-8):<15.4f}")

    if idx_noise is not None:
        print(f"{'Noise/residual':<20} {varexp_tr_mean[idx_noise]:<12.4f} {varexp_val_mean[idx_noise]:<12.4f} {varexp_te_mean[idx_noise]:<12.4f}")

    print("="*60)

    # assert varexp_te.shape[1] == 4, (
    #     f"Expected 4 columns (z1, zs2, zc1, noise), got {varexp_te.shape[1]}"
    # )

    out_dir = "/users/antonios/LEAF_revisit/LEAF/mgm/1000_kappa/"
    os.makedirs(out_dir, exist_ok=True)


    # # ---- PEARSON CORRELATIONS: Component predictions vs True values ----
    # print("\n" + "="*60)
    # print("PEARSON CORRELATIONS: Component predictions vs True metabolites")
    # print("="*60)

    # with torch.no_grad():
    #     # Get component-specific predictions
    #     f_zs1_test = decoder.forward_z(zs1_te).detach().cpu().numpy()  # (N, n_metabolites)
    #     f_cs_test = decoder.forward_c([zs2_te, zc1_te])
    #     f_zs2_test = f_cs_test[0].detach().cpu().numpy()  # (N, n_metabolites)
    #     f_zc1_test = f_cs_test[1].detach().cpu().numpy()  # (N, n_metabolites)

    # from scipy.stats import pearsonr

    # n_metabolites = Y_test_np.shape[1]
    # corr_zs1_per_met = []
    # corr_zs2_per_met = []
    # corr_zc1_per_met = []

    # for j in range(n_metabolites):
    #     # Correlation between zs1 predictions and true values for metabolite j
    #     corr_zs1, _ = pearsonr(f_zs1_test[:, j], Y_test_np[:, j])
    #     corr_zs1_per_met.append(corr_zs1)

    #     # Correlation between zs2 predictions and true values for metabolite j
    #     corr_zs2, _ = pearsonr(f_zs2_test[:, j], Y_test_np[:, j])
    #     corr_zs2_per_met.append(corr_zs2)

    #     # Correlation between zc1 predictions and true values for metabolite j
    #     corr_zc1, _ = pearsonr(f_zc1_test[:, j], Y_test_np[:, j])
    #     corr_zc1_per_met.append(corr_zc1)

    # corr_zs1_per_met = np.array(corr_zs1_per_met)
    # corr_zs2_per_met = np.array(corr_zs2_per_met)
    # corr_zc1_per_met = np.array(corr_zc1_per_met)

    # print(f"\nzs1 (microbiome-specific) correlations:")
    # print(f"  Mean:   {np.mean(corr_zs1_per_met):.3f}")
    # print(f"  Median: {np.median(corr_zs1_per_met):.3f}")
    # print(f"  Std:    {np.std(corr_zs1_per_met):.3f}")
    # print(f"  Range:  [{np.min(corr_zs1_per_met):.3f}, {np.max(corr_zs1_per_met):.3f}]")

    # print(f"\nzs2 (shared) correlations:")
    # print(f"  Mean:   {np.mean(corr_zs2_per_met):.3f}")
    # print(f"  Median: {np.median(corr_zs2_per_met):.3f}")
    # print(f"  Std:    {np.std(corr_zs2_per_met):.3f}")
    # print(f"  Range:  [{np.min(corr_zs2_per_met):.3f}, {np.max(corr_zs2_per_met):.3f}]")

    # print(f"\nzc1 (covariate-specific) correlations:")
    # print(f"  Mean:   {np.mean(corr_zc1_per_met):.3f}")
    # print(f"  Median: {np.median(corr_zc1_per_met):.3f}")
    # print(f"  Std:    {np.std(corr_zc1_per_met):.3f}")
    # print(f"  Range:  [{np.min(corr_zc1_per_met):.3f}, {np.max(corr_zc1_per_met):.3f}]")

    # # Save correlations
    # corr_dir = os.path.join(out_dir, "correlations")
    # os.makedirs(corr_dir, exist_ok=True)

    # np.save(os.path.join(corr_dir, f"corr_zs1_per_met_{MODEL_NAME}_seed{args.seed}.npy"), corr_zs1_per_met)
    # np.save(os.path.join(corr_dir, f"corr_zs2_per_met_{MODEL_NAME}_seed{args.seed}.npy"), corr_zs2_per_met)
    # np.save(os.path.join(corr_dir, f"corr_zc1_per_met_{MODEL_NAME}_seed{args.seed}.npy"), corr_zc1_per_met)

    # print(f"\nCorrelations saved to: {corr_dir}")
    # print("="*60)


    # # Save metabolite names (only once, not per seed)
    # metabolite_names_file = os.path.join(out_dir, "metabolite_names.txt")
    # if not os.path.exists(metabolite_names_file):
    #     try:
    #         with open(metabolite_names_file, 'w') as f:
    #             for name in metabolite_names:
    #                 f.write(f"{name}\n")
    #         print(f"Saved metabolite names to: {metabolite_names_file}")
    #     except NameError:
    #         print("Warning: metabolite_names not defined, skipping save")



    # Build variance tables
    df_train_long = build_variance_table(varexp_tr, Y_train, "train", metabolite_names=metabolite_names)
    df_val_long  = build_variance_table(varexp_val, Y_val, "val", metabolite_names=metabolite_names)
    df_test_long  = build_variance_table(varexp_te, Y_test, "test", metabolite_names=metabolite_names)

    df_both = pd.concat([df_train_long,df_val_long,df_test_long ], ignore_index=True)
    df_piv  = df_both.pivot_table(
        index=["outcome","component"],
        columns="split",
        values="est_fraction",
        aggfunc="mean",
    ).reset_index()


    print("Var(Y_train) min/mean/max:",
        float(Y_train.var(dim=0, unbiased=False).min()),
        float(Y_train.var(dim=0, unbiased=False).mean()),
        float(Y_train.var(dim=0, unbiased=False).max()))

    # After line 2632 (after optimize())
    print("\n=== Post-Training Diagnostics ===")
    int_z_final, int_c_final, int_cz_final = decoder.calculate_integrals_numpy()
    print(f"Final int_z_mean_abs: {np.abs(int_z_final).mean():.6f}")
    print(f"Final int_c0_mean_abs: {np.abs(int_c_final[0]).mean():.6f}")
    print(f"Final int_c1_mean_abs: {np.abs(int_c_final[1]).mean():.6f}")

    # Check component correlations
    with torch.no_grad():
        f_z = decoder.forward_z(zs1_val.to(device))
        f_c0 = decoder.mappings_c[0](zs2_val.to(device))
        f_c1 = decoder.mappings_c[1](zc1_val.to(device))
        
        cov_z_c0 = torch.corrcoef(torch.cat([f_z[:, 0:1], f_c0[:, 0:1]], dim=1).T)[0, 1]
        cov_z_c1 = torch.corrcoef(torch.cat([f_z[:, 0:1], f_c1[:, 0:1]], dim=1).T)[0, 1]
        cov_c0_c1 = torch.corrcoef(torch.cat([f_c0[:, 0:1], f_c1[:, 0:1]], dim=1).T)[0, 1]
        
        print(f"Component correlations (first metabolite):")
        print(f"  corr(f_z, f_c0) = {cov_z_c0:.4f}")
        print(f"  corr(f_z, f_c1) = {cov_z_c1:.4f}")
        print(f"  corr(f_c0, f_c1) = {cov_c0_c1:.4f}")

    def compute_avg_covcorr_from_decoder(decoder_obj, zs1_in, zs2_in, zc1_in, Y_in=None):
        """
        Measures dependence between component outputs (per-outcome), averaged across outcomes.

        Returns:
        avg_corr: (3,3) mean correlation across outcomes (NaN-safe)
        avg_cov:  (3,3) mean covariance across outcomes
        cov_share_stats: dict with cross-covariance share summaries
            - relative_to_var_yhat: uses Var(Yhat_j)
            - relative_to_var_y:    uses Var(Y_j) if Y_in provided
        """
        with torch.no_grad():
            # component predictions: all should be (N, D_out)
            f_z1 = decoder_obj.forward_z(zs1_in).detach().cpu().numpy()

            f_cs = decoder_obj.forward_c([zs2_in, zc1_in])
            f_zs2 = f_cs[0].detach().cpu().numpy()
            f_zc1 = f_cs[1].detach().cpu().numpy()

        n_outcomes = f_z1.shape[1]
        cov_mats = []
        corr_mats = []

        cov_share_var_yhat = []
        cov_share_var_y = []

        for j in range(n_outcomes):
            X = np.column_stack([f_z1[:, j], f_zs2[:, j], f_zc1[:, j]])  # (N,3)

            # covariance is always well-defined (centers internally)
            cov = np.cov(X.T, ddof=0)
            cov_mats.append(cov)

            # NaN-safe correlation: if any component variance is 0, skip corr for this outcome
            vars_ = np.diag(cov)
            if np.all(vars_ > 0):
                corr = cov / np.sqrt(np.outer(vars_, vars_))
                corr_mats.append(corr)

            # cross-covariance term in Var(sum)
            cov_term = 2.0 * (cov[0, 1] + cov[0, 2] + cov[1, 2])

            # normalize by Var(Yhat_j)
            yhat_j = X.sum(axis=1)
            var_yhat = np.var(yhat_j, ddof=0)
            if var_yhat > 0:
                cov_share_var_yhat.append(cov_term / var_yhat)

            # optional: normalize by Var(Y_j)
            if Y_in is not None:
                yj = Y_in[:, j].detach().cpu().numpy()
                var_y = np.var(yj, ddof=0)
                if var_y > 0:
                    cov_share_var_y.append(cov_term / var_y)

        avg_cov = np.mean(cov_mats, axis=0)

        # If corr_mats is empty (all outcomes degenerate), return NaNs
        if len(corr_mats) > 0:
            avg_corr = np.mean(corr_mats, axis=0)
        else:
            avg_corr = np.full((3, 3), np.nan, dtype=float)

        cov_share_stats = {
            "relative_to_var_yhat": {
                "mean": float(np.mean(cov_share_var_yhat)) if len(cov_share_var_yhat) else None,
                "median": float(np.median(cov_share_var_yhat)) if len(cov_share_var_yhat) else None,
                "max_abs": float(np.max(np.abs(cov_share_var_yhat))) if len(cov_share_var_yhat) else None,
            }
        }

        if Y_in is not None:
            cov_share_stats["relative_to_var_y"] = {
                "mean": float(np.mean(cov_share_var_y)) if len(cov_share_var_y) else None,
                "median": float(np.median(cov_share_var_y)) if len(cov_share_var_y) else None,
                "max_abs": float(np.max(np.abs(cov_share_var_y))) if len(cov_share_var_y) else None,
            }

        return avg_corr, avg_cov, cov_share_stats

    avg_corr_tr, avg_cov_tr, cov_stats_tr = compute_avg_covcorr_from_decoder(
        decoder, zs1_tr, zs2_tr, zc1_tr, Y_in=Y_train
    )
    avg_corr_val, avg_cov_val, cov_stats_val = compute_avg_covcorr_from_decoder(
        decoder, zs1_val, zs2_val, zc1_val, Y_in=Y_val
    )
    # avg_corr_te, avg_cov_te, cov_stats_te = compute_avg_covcorr_from_decoder(
    #     decoder, zs1_te, zs2_te, zc1_te, Y_in=Y_test
    # )

    print("Train cov share stats:", cov_stats_tr)
    print("Val   cov share stats:", cov_stats_val)
   # print("Test  cov share stats:", cov_stats_te)
    print("Avg corr (train):\n", avg_corr_tr)


#Check for overfitting by correlating train and test variance explained
    corrs = []
    for i in range(varexp_tr.shape[1]):  # each component
        corr = np.corrcoef(varexp_tr[:, i], varexp_val[:, i])[0, 1]
        corrs.append(corr)

    df_corr = pd.DataFrame({
        'component': [f'comp{i+1}' for i in range(len(corrs))],
        'train_mean': varexp_tr.mean(0).numpy(),
        'val_mean': varexp_val.mean(0).numpy(),
        'corr_train_val': corrs
    })
    print(df_corr)

    # avg_corr_te, avg_cov_te, cov_share_stats_te = compute_avg_covcorr_from_decoder(
    #     decoder,
    #     zs1_te.to(device),
    #     zs2_te.to(device),
    #     zc1_te.to(device),
    #     Y_test  # <-- important if you want covariance share relative to Var(Y)
    # )

    # print("\nTEST component correlation:\n", avg_corr_te)
    # print("max |off-diag corr|:", float(np.max(np.abs(avg_corr_te[~np.eye(3, dtype=bool)]))))
    # print("\nTEST component covariance:\n", avg_cov_te)
    # print("max |off-diag cov|:", float(np.max(np.abs(avg_cov_te[~np.eye(3, dtype=bool)]))))
    # print("\nTEST covariance share stats (2*sum cov / Var(y)):\n", cov_share_stats_te)


    def debug_cov_share(decoder, zs1, zs2, zc1, Y, top_k=10):
        decoder.eval()
        with torch.no_grad():
            f_z1 = decoder.forward_z(zs1).detach().cpu().numpy()           # (N,D)
            f_zs2 = decoder.forward_c([zs2, zc1])[0].detach().cpu().numpy()
            f_zc1 = decoder.forward_c([zs2, zc1])[1].detach().cpu().numpy()
            Y_np = Y.detach().cpu().numpy()

        N, D = f_z1.shape
        cov_share = np.zeros(D, dtype=float)
        varY = np.zeros(D, dtype=float)
        cov_term = np.zeros(D, dtype=float)

        for j in range(D):
            X = np.column_stack([f_z1[:, j], f_zs2[:, j], f_zc1[:, j]])  # (N,3)
            cov = np.cov(X.T, ddof=0)
            cov_term[j] = 2.0 * (cov[0,1] + cov[0,2] + cov[1,2])
            varY[j] = np.var(Y_np[:, j], ddof=0)
            cov_share[j] = cov_term[j] / varY[j] if varY[j] > 0 else np.nan

        idx = np.argsort(np.abs(cov_share))[::-1][:top_k]
        print("\nTop metabolites by |cov_share| (relative to Var(Y)):")
        for j in idx:
            print(f"j={j:3d}  cov_share={cov_share[j]: .3f}  cov_term={cov_term[j]: .3e}  Var(Y)={varY[j]: .3e}")

        print("\nVar(Y) summary:")
        print("  min:", np.nanmin(varY), "median:", np.nanmedian(varY), "mean:", np.nanmean(varY), "max:", np.nanmax(varY))

    # # call it
    # debug_cov_share(decoder, zs1_te, zs2_te, zc1_te, Y_test, top_k=10)

    # # df_piv should have columns: outcome, component, train, val, test (depending on what you included)
    # # After you build df_both = concat([df_train_long, df_val_long, df_test_long])

    if ("train" in df_piv.columns) and ("val" in df_piv.columns) and ("test" in df_piv.columns):
        df_piv["est_fraction_mean"] = df_piv[["train", "val", "test"]].mean(axis=1)
        cols = ["outcome", "component", "train", "val", "test", "est_fraction_mean"]

    elif ("train" in df_piv.columns) and ("val" in df_piv.columns):
        df_piv["est_fraction_mean"] = df_piv[["train", "val"]].mean(axis=1)
        df_piv["test"] = np.nan
        cols = ["outcome", "component", "train", "val", "test", "est_fraction_mean"]

    elif "train" in df_piv.columns:
        df_piv["est_fraction_mean"] = df_piv["train"]
        df_piv["val"] = np.nan
        df_piv["test"] = np.nan
        cols = ["outcome", "component", "train", "val", "test", "est_fraction_mean"]

    elif "val" in df_piv.columns:
        df_piv["est_fraction_mean"] = df_piv["val"]
        df_piv["train"] = np.nan
        df_piv["test"] = np.nan
        cols = ["outcome", "component", "train", "val", "test", "est_fraction_mean"]

    elif "test" in df_piv.columns:
        df_piv["est_fraction_mean"] = df_piv["test"]
        df_piv["train"] = np.nan
        df_piv["val"] = np.nan
        cols = ["outcome", "component", "train", "val", "test", "est_fraction_mean"]

    else:
        raise ValueError("df_piv does not contain train/val/test columns")

    final_table = df_piv.reindex(columns=cols).copy()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"5000_Variance_comparison_{MODEL_NAME}_{run_id}.csv")
    final_table.to_csv(out_path, index=False)

    print("Wrote:", out_path)
    print(final_table.head(12))

# -----------------------------
# Metrics (TRAIN + VAL + TEST)
# -----------------------------
    from scipy.stats import spearmanr

    def per_met_scc(y_true_np, y_pred_np):
        out = np.zeros(y_true_np.shape[1], dtype=np.float64)
        for j in range(y_true_np.shape[1]):
            rho, _ = spearmanr(y_true_np[:, j], y_pred_np[:, j])
            out[j] = rho if np.isfinite(rho) else 0.0
        return out

    def per_met_rmse(y_true_np, y_pred_np):
        mse = np.mean((y_true_np - y_pred_np) ** 2, axis=0)
        return np.sqrt(mse)

    def per_met_r2(y_true_np, y_pred_np, ybar_train_np, eps=1e-12):
        ss_res = np.sum((y_true_np - y_pred_np) ** 2, axis=0)
        ss_tot = np.sum((y_true_np - ybar_train_np) ** 2, axis=0)
        return 1.0 - (ss_res / (ss_tot + eps))

    # Convert tensors once
    Y_train_np = Y_train.detach().cpu().numpy()
    Y_val_np   = Y_val.detach().cpu().numpy()

    Y_pred_tr_np  = Y_pred_tr.detach().cpu().numpy()
    Y_pred_val_np = Y_pred_val.detach().cpu().numpy()

    # Baseline predictor: mean of train for each metabolite
    ybar_train = Y_train_np.mean(axis=0, keepdims=True)

    def summarize_split(name, Y_np, Yp_np):
        scc = per_met_scc(Y_np, Yp_np)
        rmse = per_met_rmse(Y_np, Yp_np)
        r2 = per_met_r2(Y_np, Yp_np, ybar_train)

        return {
            f"mean_rmse_{name}": float(np.mean(rmse)),
            f"median_rmse_{name}": float(np.median(rmse)),
            f"mean_scc_{name}": float(np.mean(scc)),
            f"median_scc_{name}": float(np.median(scc)),
            f"frac_scc_pos_{name}": float(np.mean(scc > 0.0)),
            f"mean_r2_{name}": float(np.mean(r2)),
            f"median_r2_{name}": float(np.median(r2)),
            f"frac_r2_pos_{name}": float(np.mean(r2 > 0.0)),
            f"frac_r2_gt_005_{name}": float(np.mean(r2 > 0.05)),
        }

    metrics = {}
    metrics.update(summarize_split("tr",  Y_train_np, Y_pred_tr_np))
    metrics.update(summarize_split("val", Y_val_np,   Y_pred_val_np))

    # --- TEST metrics (only if available) ---
    has_test = ("Y_test" in locals()) and ("Y_pred_test" in locals()) and (Y_test is not None) and (Y_pred_test is not None)
    if has_test:
        Y_test_np = Y_test.detach().cpu().numpy()
        Y_pred_test_np = Y_pred_te.detach().cpu().numpy()
        metrics.update(summarize_split("test", Y_test_np, Y_pred_test_np))
    else:
        # Keep columns consistent even if you skip test
        metrics.update({
            "mean_rmse_test": np.nan,
            "median_rmse_test": np.nan,
            "mean_scc_test": np.nan,
            "median_scc_test": np.nan,
            "frac_scc_pos_test": np.nan,
            "mean_r2_test": np.nan,
            "median_r2_test": np.nan,
            "frac_r2_pos_test": np.nan,
            "frac_r2_gt_005_test": np.nan,
        })

    # -----------------------------
    # Append to CSV
    # -----------------------------
    summary_path = "/users/antonios/LEAF_revisit/LEAF/one_hold/test/metrics_summary.csv"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    row = pd.DataFrame([{
        "seed": args.seed,
        "beta": args.beta,
        "kappa": args.kappa,
        **metrics
    }])

    if os.path.exists(summary_path):
        row.to_csv(summary_path, mode="a", header=False, index=False)
    else:
        row.to_csv(summary_path, index=False)

    print("Wrote metrics row to:", summary_path)
    print(row.to_string(index=False))

    # -----------------------------
    # Save final integral values to CSV
    # -----------------------------
    integrals_csv_path = "/users/antonios/LEAF_revisit/LEAF/one_hold/2_layers/hopefully_last/grid_search/integrals_final_summary.csv"

    # Compute final integral statistics
    int_z_final_max = np.abs(int_z_final).max()
    int_z_final_mean = np.abs(int_z_final).mean()
    int_c_final_max = max([np.abs(ic).max() for ic in int_c_final])
    int_c_final_mean = np.mean([np.abs(ic).mean() for ic in int_c_final])
    int_cz_final_max = np.abs(int_cz_final).max() if len(int_cz_final) > 0 else 0.0
    int_cz_final_mean = np.abs(int_cz_final).mean() if len(int_cz_final) > 0 else 0.0

    integrals_row = pd.DataFrame([{
        "seed": args.seed,
        "beta": args.beta,
        "kappa": args.kappa,
        "int_z_max": int_z_final_max,
        "int_z_mean": int_z_final_mean,
        "int_c_max": int_c_final_max,
        "int_c_mean": int_c_final_mean,
        "int_cz_max": int_cz_final_max,
        "int_cz_mean": int_cz_final_mean,
    }])

    if os.path.exists(integrals_csv_path):
        integrals_row.to_csv(integrals_csv_path, mode="a", header=False, index=False)
    else:
        integrals_row.to_csv(integrals_csv_path, index=False)

    print(f"Wrote integrals row to: {integrals_csv_path}")
    print(integrals_row.to_string(index=False))

    print("train sum min/mean/max:",
        varexp_tr.sum(1).min(), varexp_tr.sum(1).mean(), varexp_tr.sum(1).max())
    print("val sum min/mean/max:",
        varexp_val.sum(1).min(), varexp_val.sum(1).mean(), varexp_val.sum(1).max())

    # sums should be ~1
    print("val sum min/mean/max:",
        varexp_val.sum(1).min(), varexp_val.sum(1).mean(), varexp_val.sum(1).max())

    # noise bounds
    noise = varexp_val[:, 3]
    print("noise val min/mean/max:", noise.min(), noise.mean(), noise.max())
    print("fraction of seeds with noise>1:", (noise > 1).sum().item() / noise.numel())


 


    def bh_fdr(pvals: np.ndarray) -> np.ndarray:
        p = np.asarray(pvals, dtype=float)
        m = p.size
        order = np.argsort(p)
        ranked = p[order]
        q = ranked * m / (np.arange(1, m + 1))
        q = np.minimum.accumulate(q[::-1])[::-1]
        q = np.clip(q, 0.0, 1.0)
        qte = np.empty_like(q)
        qte[order] = q
        return qte


    def permutation_test_componentwise_val(
        decoder,
        Y_te: torch.Tensor,
        zs1_te: torch.Tensor,
        zs2_te: torch.Tensor,
        zc1_te: torch.Tensor,
        out_dir: str,
        MODEL_NAME: str,
        run_id: str,
        metabolite_names=None,
        n_permutations: int = 200,
        divide_by_total_var: bool = False,
        batch_size: int = 256,
        device: str = "cuda",
        do_fdr: bool = True,
        seed: int = 0,
        make_plots: bool = True,
    ):
        decoder.eval()

        # -----------------------------
        # Alignment checks
        # -----------------------------
        N = Y_te.shape[0]
        assert (
            zs1_te.shape[0] == N and zs2_te.shape[0] == N and zc1_te.shape[0] == N
        ), "Alignment error: Y_test and latent tensors must have the same number of samples in the same order."

        # Ensure tensors on device
        Y_te = Y_te.to(device)
        zs1_te = zs1_te.to(device)
        zs2_te = zs2_te.to(device)
        zc1_te = zc1_te.to(device)

        # Base VAL loader (shuffle=False)
      #  ds_te = TensorDataset(Y_te, zs1_te, zs2_te, zc1_te)
        data_loader_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, drop_last=False)

        def eval_split(decoder_, data_loader_):
            with torch.no_grad():
                Y_err_list, Y_pred_list = [], []
                for batch in data_loader_:
                    Y_b, z1_b, z2_b, c_b = batch
                    confs = [z2_b, c_b]
                    Y_pred = decoder_(z1_b, confs)
                    Y_err = Y_b - Y_pred
                    Y_err_list.append(Y_err)
                    Y_pred_list.append(Y_pred)
                Y_err = torch.cat(Y_err_list, dim=0)
                Y_pred = torch.cat(Y_pred_list, dim=0)
            return Y_err, Y_pred

        # -----------------------------
        # Real evaluation
        # -----------------------------
        Y_error_te_real, _ = eval_split(decoder, data_loader_te)

        V_real = (
            decoder.fraction_of_variance_explained(
                zs1_te,
                [zs2_te, zc1_te],
                Y_error=Y_error_te_real,
                divide_by_total_var=divide_by_total_var,
            )
            .detach()
            .cpu()
            .numpy()
        )

        D, Kc = V_real.shape

        # Global mean per component (tested component only)
        T_real = V_real.mean(axis=0)  # shape (Kc,)

        # Name components
        component_names = ["z1", "zs2", "zc1"]
        if Kc >= 4:
            for i in range(3, Kc - 1):
                component_names.append(f"extra_{i - 2}")
            component_names.append("noise")
        while len(component_names) < Kc:
            component_names.append(f"comp_{len(component_names) + 1}")

        # We only test these (because we only permute these latents)
        perm_targets = [("z1", 0), ("zs2", 1), ("zc1", 2)]

        # Metabolite names
        if metabolite_names is None:
            print(f"Warning: metabolite_names is None, using indices 0-{D-1}")
            met_names = np.arange(D)
        else:
            met_names = np.asarray(metabolite_names)
            print(f"Received metabolite_names: {met_names[:5]}... (showing first 5)")
            if len(met_names) != D:
                print(f"Warning: metabolite_names length ({len(met_names)}) != Y_test columns ({D})")
                print("Using indices instead of names")
                met_names = np.arange(D)

        # RNG
        g = torch.Generator(device=device)
        g.manual_seed(seed)

        os.makedirs(out_dir, exist_ok=True)
        plot_dir = os.path.join(out_dir, "permtest_plots")
        if make_plots:
            os.makedirs(plot_dir, exist_ok=True)

        results_global = []
        results_per_met = []

        for comp_name, comp_idx in perm_targets:
            print(f"\n--- Permutation test on VAL: permuting {comp_name} ---")

            exceed_global = 0
            T_perm_mean = 0.0
            T_perm_M2 = 0.0

            exceed_frac = np.zeros(D, dtype=np.int64)
            V_null_mean = np.zeros(D, dtype=np.float64)
            V_null_M2 = np.zeros(D, dtype=np.float64)

            # store global null distribution for plotting
            T_perm_all = np.zeros(n_permutations, dtype=np.float64)

            for perm_i in range(n_permutations):
                if (perm_i + 1) % 10 == 0:
                    print(f" Progress: {perm_i + 1}/{n_permutations}")

                perm = torch.randperm(N, generator=g, device=device)

                if comp_name == "z1":
                    z1_perm = zs1_te[perm]
                    z2_perm = zs2_te
                    c_perm = zc1_te
                elif comp_name == "zs2":
                    z1_perm = zs1_te
                    z2_perm = zs2_te[perm]
                    c_perm = zc1_te
                elif comp_name == "zc1":
                    z1_perm = zs1_te
                    z2_perm = zs2_te
                    c_perm = zc1_te[perm]
                else:
                    raise RuntimeError(f"Unknown component to permute: {comp_name}")

                ds_perm = TensorDataset(Y_te, z1_perm, z2_perm, c_perm)
                loader_perm = DataLoader(ds_perm, batch_size=batch_size, shuffle=False, drop_last=False)

                Y_error_perm, _ = eval_split(decoder, loader_perm)

                V_perm = (
                    decoder.fraction_of_variance_explained(
                        z1_perm,
                        [z2_perm, c_perm],
                        Y_error=Y_error_perm,
                        divide_by_total_var=divide_by_total_var,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

                # Global stat for permuted component
                T_perm = float(V_perm[:, comp_idx].mean())
                T_perm_all[perm_i] = T_perm

                if T_perm >= float(T_real[comp_idx]):
                    exceed_global += 1

                # online mean/var for global null
                delta = T_perm - T_perm_mean
                T_perm_mean += delta / (perm_i + 1)
                delta2 = T_perm - T_perm_mean
                T_perm_M2 += delta * delta2

                # Per-metabolite null tracking for permuted component only
                Vp = V_perm[:, comp_idx]
                exceed_frac += (Vp >= V_real[:, comp_idx]).astype(np.int64)

                delta_f = Vp - V_null_mean
                V_null_mean += delta_f / (perm_i + 1)
                delta2_f = Vp - V_null_mean
                V_null_M2 += delta_f * delta2_f

            # p-values
            p_global = (1.0 + exceed_global) / (1.0 + n_permutations)

            denom = max(1, n_permutations - 1)
            T_perm_std = float(np.sqrt(T_perm_M2 / denom))

            df_global = pd.DataFrame(
                {
                    "permuted_component": [comp_name],
                    "tested_component": [component_names[comp_idx]],
                    "T_real_test_mean": [float(T_real[comp_idx])],
                    "T_perm_test_mean": [float(T_perm_mean)],
                    "T_perm_test_std": [float(T_perm_std)],
                    "p_value_test_global": [float(p_global)],
                    "divide_by_total_var": [bool(divide_by_total_var)],
                    "n_permutations": [int(n_permutations)],
                    "seed": [int(seed)],
                    "Kc_returned": [int(Kc)],
                }
            )
            results_global.append(df_global)

            p_frac = (1.0 + exceed_frac.astype(np.float64)) / (1.0 + n_permutations)
            V_null_std = np.sqrt(V_null_M2 / denom)

            df_frac = pd.DataFrame(
                {
                    "permuted_component": comp_name,
                    "tested_component": component_names[comp_idx],
                    "metabolite": met_names,
                    "varexp_real_test": V_real[:, comp_idx],
                    "varexp_null_mean_test": V_null_mean,
                    "varexp_null_std_test": V_null_std,
                    "p_value_test_component": p_frac,
                    "divide_by_total_var": bool(divide_by_total_var),
                    "n_permutations": int(n_permutations),
                    "seed": int(seed),
                    "Kc_returned": int(Kc),
                }
            )

            # Effect size: z-score vs null
            df_frac["effect_z"] = (
                (df_frac["varexp_real_test"] - df_frac["varexp_null_mean_test"])
                / (df_frac["varexp_null_std_test"] + 1e-12)
            )

            if do_fdr:
                df_frac["q_value_bh_fdr"] = bh_fdr(df_frac["p_value_test_component"].values)

            results_per_met.append(df_frac)

            # -----------------------------
            # Plots
            # -----------------------------
            if make_plots:
                # 1) Global null histogram with real line
                fig, ax = plt.subplots(figsize=(10, 7))
                ax.hist(T_perm_all, bins=40, alpha=0.7, color="skyblue", edgecolor="black")
                ax.axvline(
                    float(T_real[comp_idx]),
                    color="red",
                    linewidth=3,
                    label=f"Observed (p={p_global:.4f})",
                    linestyle="--",
                )

                if p_global < 0.05:
                    y_max = ax.get_ylim()[1]
                    ax.text(
                        float(T_real[comp_idx]),
                        y_max * 0.95,
                        f"Significant!\np={p_global:.4f}",
                        fontsize=11,
                        fontweight="bold",
                        color="red",
                        ha="left",
                        va="top",
                        bbox=dict(
                            boxstyle="round,pad=0.5",
                            facecolor="yellow",
                            alpha=0.8,
                            edgecolor="red",
                            linewidth=2,
                        ),
                    )

                ax.set_title(f"Permutation Test: {comp_name}", fontsize=18, fontweight="bold", pad=20)
                ax.set_xlabel("Mean Variance Fraction Across Metabolites", fontsize=14, fontweight="bold")
                ax.set_ylabel("Frequency", fontsize=14, fontweight="bold")
                ax.tick_params(axis="both", labelsize=12)
                ax.legend(fontsize=12, loc="upper right", frameon=True, shadow=True)
                ax.grid(True, alpha=0.3, linestyle="--")
                plt.tight_layout()
                pth = os.path.join(plot_dir, f"global_null_hist_{MODEL_NAME}_{run_id}_{comp_name}.png")
                plt.savefig(pth, dpi=300, bbox_inches="tight")
                plt.close()

                # 2) Real vs null mean scatter (per metabolite)
                fig, ax = plt.subplots(figsize=(12, 11))
                sig_mask = df_frac["p_value_test_component"].values < 0.05

                non_sig_idx = np.where(~sig_mask)[0]
                if len(non_sig_idx) > 0:
                    ax.scatter(
                        df_frac.iloc[non_sig_idx]["varexp_null_mean_test"].values,
                        df_frac.iloc[non_sig_idx]["varexp_real_test"].values,
                        s=50,
                        alpha=0.4,
                        c="steelblue",
                        edgecolors="black",
                        linewidth=0.3,
                        zorder=1,
                    )

                lim = max(df_frac["varexp_null_mean_test"].max(), df_frac["varexp_real_test"].max())
                ax.plot([0, lim], [0, lim], "gray", linestyle="--", linewidth=2, alpha=0.7, zorder=2)

                sig_idx = np.where(sig_mask)[0]
                if len(sig_idx) > 0:
                    for idx in sig_idx:
                        x_val = df_frac.iloc[idx]["varexp_null_mean_test"]
                        y_val = df_frac.iloc[idx]["varexp_real_test"]
                        met_name = str(df_frac.iloc[idx]["metabolite"])
                        ax.scatter(
                            x_val,
                            y_val,
                            s=70,
                            alpha=0.7,
                            c="red",
                            edgecolors="black",
                            linewidth=0.5,
                            zorder=3,
                        )
                        if y_val > x_val * 1.1:
                            ax.text(
                                x_val,
                                y_val,
                                f" {met_name}",
                                fontsize=7,
                                ha="left",
                                va="center",
                                fontweight="bold",
                                color="darkred",
                                zorder=4,
                                bbox=dict(
                                    boxstyle="round,pad=0.25",
                                    facecolor="yellow",
                                    alpha=0.75,
                                    edgecolor="black",
                                    linewidth=0.4,
                                ),
                            )

                ax.set_title(f"Real vs Null Variance: {comp_name}", fontsize=18, fontweight="bold", pad=20)
                ax.set_xlabel("Null Mean Variance Fraction", fontsize=14, fontweight="bold")
                ax.set_ylabel("Observed Variance Fraction", fontsize=14, fontweight="bold")
                ax.tick_params(axis="both", labelsize=12)

                from matplotlib.patches import Patch

                legend_elements = [
                    ax.plot([0, lim], [0, lim], "gray", linestyle="--", linewidth=2, alpha=0.7)[0],
                    Patch(facecolor="red", edgecolor="black", label="Significant (p<0.05)", alpha=0.7),
                    Patch(facecolor="steelblue", edgecolor="black", label="Not significant", alpha=0.4),
                ]
                ax.legend(
                    handles=legend_elements,
                    labels=["y=x (null)", "Significant (p<0.05)", "Not significant"],
                    fontsize=12,
                    loc="upper left",
                    frameon=True,
                    shadow=True,
                )
                ax.grid(True, alpha=0.3, linestyle="--", zorder=0)
                ax.set_aspect("equal", adjustable="box")
                plt.tight_layout()
                pth = os.path.join(plot_dir, f"real_vs_null_scatter_{MODEL_NAME}_{run_id}_{comp_name}.png")
                plt.savefig(pth, dpi=300, bbox_inches="tight")
                plt.close()

                # 3) Volcano: effect_z vs -log10(p)
                fig, ax = plt.subplots(figsize=(14, 11))
                x = df_frac["effect_z"].values
                y = -np.log10(df_frac["p_value_test_component"].values + 1e-300)

                non_sig_idx = np.where(~sig_mask)[0]
                if len(non_sig_idx) > 0:
                    ax.scatter(
                        x[non_sig_idx],
                        y[non_sig_idx],
                        s=60,
                        alpha=0.4,
                        c="gray",
                        edgecolors="black",
                        linewidth=0.3,
                        zorder=1,
                    )

                sig_idx = np.where(sig_mask)[0]
                if len(sig_idx) > 0:
                    for idx in sig_idx:
                        met_name = str(df_frac.iloc[idx]["metabolite"])
                        x_pos = x[idx]
                        y_pos = y[idx]
                        ax.scatter(
                            x_pos,
                            y_pos,
                            s=80,
                            alpha=0.7,
                            c="red",
                            edgecolors="black",
                            linewidth=0.5,
                            zorder=3,
                        )
                        ax.text(
                            x_pos,
                            y_pos,
                            f" {met_name}",
                            fontsize=8,
                            ha="left",
                            va="center",
                            fontweight="bold",
                            color="darkred",
                            zorder=4,
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="yellow",
                                alpha=0.75,
                                edgecolor="black",
                                linewidth=0.5,
                            ),
                        )

                ax.axhline(-np.log10(0.05), color="blue", linestyle="--", linewidth=2, alpha=0.7, zorder=2)

                sig_threshold = 0.0
                if do_fdr and "q_value_bh_fdr" in df_frac.columns:
                    if (df_frac["q_value_bh_fdr"] < 0.05).any():
                        sig_threshold = float(
                            df_frac.loc[df_frac["q_value_bh_fdr"] < 0.05, "p_value_test_component"].max()
                        )
                        ax.axhline(
                            -np.log10(sig_threshold),
                            color="green",
                            linestyle="--",
                            linewidth=2,
                            alpha=0.7,
                            zorder=2,
                        )

                ax.set_title(f"Volcano Plot: {comp_name}", fontsize=18, fontweight="bold", pad=20)
                ax.set_xlabel("Effect Size (z-score vs null)", fontsize=14, fontweight="bold")
                ax.set_ylabel(r"$-\log_{10}(p\mathrm{-value})$", fontsize=14, fontweight="bold")
                ax.tick_params(axis="both", labelsize=12)


                legend_elements = [
                    Line2D([0], [0], color="blue", linestyle="--", linewidth=2, alpha=0.7),
                    Patch(facecolor="red", edgecolor="black", label="Significant (p<0.05)", alpha=0.7),
                    Patch(facecolor="gray", edgecolor="black", label="Not significant", alpha=0.4),
                ]
                legend_labels = ["p=0.05 threshold", "Significant (p<0.05)", "Not significant"]
                if sig_threshold > 0:
                    legend_labels[0] = "p=0.05 / FDR=0.05"
                ax.legend(handles=legend_elements, labels=legend_labels, fontsize=12, loc="upper right", frameon=True, shadow=True)
                ax.grid(True, alpha=0.3, linestyle="--", zorder=0)
                plt.tight_layout()
                pth = os.path.join(plot_dir, f"volcano_{MODEL_NAME}_{run_id}_{comp_name}.png")
                plt.savefig(pth, dpi=300, bbox_inches="tight")
                plt.close()

                # 4) Effect Size vs Variance Fraction
                fig, ax = plt.subplots(figsize=(12, 10))

                non_sig_idx = np.where(~sig_mask)[0]
                if len(non_sig_idx) > 0:
                    ax.scatter(
                        df_frac.iloc[non_sig_idx]["varexp_real_test"].values * 100,
                        df_frac.iloc[non_sig_idx]["effect_z"].values,
                        s=60,
                        alpha=0.4,
                        c="gray",
                        edgecolors="black",
                        linewidth=0.3,
                        zorder=1,
                    )

                ax.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5, zorder=2)

                sig_idx = np.where(sig_mask)[0]
                if len(sig_idx) > 0:
                    sig_df = df_frac.iloc[sig_idx].copy()
                    sig_df = sig_df.nlargest(min(15, len(sig_df)), "effect_z")

                    for _, row in sig_df.iterrows():
                        x_val = float(row["varexp_real_test"]) * 100
                        y_val = float(row["effect_z"])
                        met_name = str(row["metabolite"])
                        ax.scatter(
                            x_val,
                            y_val,
                            s=80,
                            alpha=0.7,
                            c="red",
                            edgecolors="black",
                            linewidth=0.5,
                            zorder=3,
                        )
                        ax.text(
                            x_val,
                            y_val,
                            f" {met_name}",
                            fontsize=7,
                            ha="left",
                            va="center",
                            fontweight="bold",
                            color="darkred",
                            zorder=4,
                            bbox=dict(
                                boxstyle="round,pad=0.25",
                                facecolor="yellow",
                                alpha=0.75,
                                edgecolor="black",
                                linewidth=0.4,
                            ),
                        )

                    remaining_sig = df_frac.iloc[sig_idx][~df_frac.iloc[sig_idx].index.isin(sig_df.index)]
                    if len(remaining_sig) > 0:
                        ax.scatter(
                            remaining_sig["varexp_real_test"].values * 100,
                            remaining_sig["effect_z"].values,
                            s=70,
                            alpha=0.7,
                            c="red",
                            edgecolors="black",
                            linewidth=0.5,
                            zorder=3,
                        )

                ax.set_title(f"Effect Size vs Variance Explained: {comp_name}", fontsize=18, fontweight="bold", pad=20)
                ax.set_xlabel("Variance Explained (%)", fontsize=14, fontweight="bold")
                ax.set_ylabel("Effect Size (z-score)", fontsize=14, fontweight="bold")
                ax.tick_params(axis="both", labelsize=12)
                ax.legend(
                    handles=[
                        Patch(facecolor="red", edgecolor="black", label="Significant (p<0.05)", alpha=0.7),
                        Patch(facecolor="gray", edgecolor="black", label="Not significant", alpha=0.4),
                    ],
                    fontsize=12,
                    loc="upper right",
                    frameon=True,
                    shadow=True,
                )
                ax.grid(True, alpha=0.3, linestyle="--", zorder=0)
                plt.tight_layout()
                pth = os.path.join(plot_dir, f"effect_vs_variance_{MODEL_NAME}_{run_id}_{comp_name}.png")
                plt.savefig(pth, dpi=300, bbox_inches="tight")
                plt.close()

                # 5) P-value distribution histogram
                fig, ax = plt.subplots(figsize=(10, 7))
                ax.hist(df_frac["p_value_test_component"].values, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
                ax.axvline(0.05, color="red", linestyle="--", linewidth=2, label="α=0.05", alpha=0.7)
                n_sig = int(sig_mask.sum())
                n_total = int(len(df_frac))
                ax.set_title(
                    f"P-value Distribution: {comp_name}\n{n_sig}/{n_total} significant ({100*n_sig/n_total:.1f}%)",
                    fontsize=18,
                    fontweight="bold",
                    pad=20,
                )
                ax.set_xlabel("P-value", fontsize=14, fontweight="bold")
                ax.set_ylabel("Frequency", fontsize=14, fontweight="bold")
                ax.tick_params(axis="both", labelsize=12)
                ax.legend(fontsize=12, frameon=True, shadow=True)
                ax.grid(True, alpha=0.3, linestyle="--")
                plt.tight_layout()
                pth = os.path.join(plot_dir, f"pvalue_dist_{MODEL_NAME}_{run_id}_{comp_name}.png")
                plt.savefig(pth, dpi=300, bbox_inches="tight")
                plt.close()

                # 6) Top significant metabolites bar chart
                if sig_mask.any():
                    sig_df = df_frac[sig_mask].copy()
                    sig_df = sig_df.sort_values("varexp_real_test", ascending=True)
                    plot_df = sig_df.tail(min(20, len(sig_df))).reset_index(drop=True)

                    n_bars = len(plot_df)
                    fig_w = 14
                    fig_h = max(7, 0.55 * n_bars)
                    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

                    y_pos = np.arange(n_bars)
                    var_pct = plot_df["varexp_real_test"].values * 100
                    colors_bar = ["red" if p < 0.01 else "orange" for p in plot_df["p_value_test_component"].values]

                    ax.barh(y_pos, var_pct, color=colors_bar, edgecolor="black", alpha=0.8, zorder=3)

                    x_max = float(var_pct.max())
                    ax.set_xlim(0, x_max * 1.25)

                    ax.set_yticks(y_pos)
                    ax.set_yticklabels([])

                    ax.set_xlabel("Variance Explained (%)", fontsize=18)
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2f}%"))
                    ax.set_title(f"Top Significant Metabolites: {comp_name}", fontsize=18, pad=16)
                    ax.tick_params(axis="x", labelsize=13)

                    for i, row in plot_df.iterrows():
                        pval = float(row["p_value_test_component"])
                        met_name = str(row["metabolite"])
                        v = float(row["varexp_real_test"]) * 100

                        if pval < 0.001:
                            stars = "***"
                        elif pval < 0.01:
                            stars = "**"
                        elif pval < 0.05:
                            stars = "*"
                        else:
                            stars = ""

                        ax.text(
                            -0.02,
                            i,
                            f"{met_name} {stars}",
                            transform=ax.get_yaxis_transform(),
                            va="center",
                            ha="right",
                            fontsize=12,
                            color="black",
                            clip_on=False,
                        )

                        if v < 0.08 * x_max:
                            ax.text(v + 0.8, i, f"{v:.2f}%", va="center", ha="left", fontsize=11, color="black")
                        else:
                            ax.text(v - 0.8, i, f"{v:.2f}%", va="center", ha="right", fontsize=11, color="white")

                    ax.legend(
                        handles=[
                            Patch(facecolor="red", edgecolor="black", label="p<0.01 (**)", alpha=0.8),
                            Patch(facecolor="orange", edgecolor="black", label="0.01≤p<0.05 (*)", alpha=0.8),
                        ],
                        fontsize=12,
                        frameon=True,
                        shadow=True,
                        loc="lower right",
                    )
                    ax.grid(True, axis="x", alpha=0.25, linestyle="--", zorder=0)

                    plt.tight_layout()
                    plt.subplots_adjust(left=0.45)

                    pth = os.path.join(plot_dir, f"top_significant_{MODEL_NAME}_{run_id}_{comp_name}.png")
                    plt.savefig(pth, dpi=300, bbox_inches="tight")
                    plt.close()

        # Save CSVs
        df_global_all = pd.concat(results_global, ignore_index=True)
        global_csv_path = os.path.join(out_dir, f"permtest_GLOBAL_test_{MODEL_NAME}_{run_id}.csv")
        df_global_all.to_csv(global_csv_path, index=False)
        print("\nSaved:", global_csv_path)

        df_per_met_all = pd.concat(results_per_met, ignore_index=True)
        per_met_csv_path = os.path.join(out_dir, f"permtest_PER_MET_test_{MODEL_NAME}_{run_id}.csv")
        df_per_met_all.to_csv(per_met_csv_path, index=False)
        print("Saved:", per_met_csv_path)

        return df_global_all, df_per_met_all


    # Example caller (keep this exactly if you already have args/config defined)
    if hasattr(args, "run_permutation_test") and args.run_permutation_test:
        df_g, df_m = permutation_test_componentwise_val(
            decoder=decoder,
            Y_te=Y_test,
            zs1_te=zs1_te,
            zs2_te=zs2_te,
            zc1_te=zc1_te,
            out_dir=out_dir,
            MODEL_NAME=MODEL_NAME,
            run_id=run_id,
            metabolite_names=metabolite_names,
            n_permutations=int(getattr(args, "n_permutations", 1000)),
            divide_by_total_var=True,
            batch_size=int(getattr(args, "batch_size", config.batch_size)),
            device=device,
            do_fdr=True,
            seed=int(getattr(args, "seed", 0)),
            make_plots=True,
        )


        wandb.finish()
if __name__ == "__main__":
    main()