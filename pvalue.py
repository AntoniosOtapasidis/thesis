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
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from torch.utils.data import Subset, DataLoader
import seaborn as sns

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
        import torch, numpy as np, pandas as pd
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

    if hasattr(disen, "zsmodel"):
        disen.zsmodel.eval()
    # Verify encoder freezing before training
    verify_encoder_frozen(disen, model_name="DisenModel")

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
    parser.add_argument('--hsic_weight', type=float, default=0.01) #Statistical independence weight
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--kappa', type=float, default=50)
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


        # Index by sim
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

        # Drop 'rep', get numpy
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

        # Load GT variance shares from R
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

        # Data and targets
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


        # Index by sim
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

        # Drop 'rep', get numpy
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

        # Load GT variance shares from R
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

        # Data and targets
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


        # Index by sim
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

        # Drop 'rep', get numpy
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

        # Load GT variance shares from R
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

        # Data and targets
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
        df_X1 = _read_clean('/users/antonios/code/microbiome.CLR.LEAF.tsv')
        df_X2 = _read_clean('/users/antonios/code/virome.CLR.LEAF.tsv')
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
                generator=torch.Generator().manual_seed(1),
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
        # Standardize CLR features for X1 and X2 using train indices only
        # X1_full = modalities_raw[0]  # [N, dim_X1], CLR already
        #X2_full = modalities_raw[1]  # [N, dim_X2], CLR already

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

        # Multimodal dataset
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

        # Load and normalize Y using train indices only
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

        # Build dataset with normalized Y as single label block
        #dataset = MultimodalDataset(data, targets1, targets2, targets3)
        dataset = MultiomicDataset(total_data=modalities, total_labels1=Y_final)


        # Create train/val/test subsets

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

    # Train models
    train_mp(args.beta, train_loader, val_loader, train_dataset, val_dataset, args)  # val used as eval

    print("CALLING train_step2", flush=True)
    logs, disen = train_step2(args, train_loader, val_loader, train_dataset, val_dataset)
    print("RETURNED from train_step2", flush=True)


    # Gather embeddings
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


   # lim_val = 2.0
    #grid_z = (torch.rand(num_samples, z_dim_zs1, device=device) * 2 * lim_val) - lim_val
    #grid_cov = lambda x: (torch.rand(num_samples, x.shape[1], device=device) * 2 * lim_val) - lim_val
    #grid_c = [grid_cov(x) for x in (zs2, zc1)]

        # Use fixed prior grids (standard normal) to prevent overfitting
    # Grid size should be large enough for proper integration
    K = 1000  # Much larger grid for better marginalization

    def rand_grid_prior(dim, K):
        # Sample from standard normal prior (no training data dependency)
        return torch.randn(K, dim, device=device).detach()

    grid_z = rand_grid_prior(zs1_tr.shape[1], K)
    grid_c = [rand_grid_prior(zs2_tr.shape[1], K), rand_grid_prior(zc1_tr.shape[1], K)]

    assert Y_train.shape[0] == zs1_tr.shape[0] == zs2_tr.shape[0] == zc1_tr.shape[0]
    assert Y_val.shape[0]   == zs1_val.shape[0]
  #  assert Y_test.shape[0]  == zs1_te.shape[0]

    def bh_fdr(pvals: np.ndarray, q: float = 0.05):
        """
        Benjamini-Hochberg FDR control.
        Returns:
            qvals: adjusted p-values
            reject: boolean mask where qvals <= q
        """
        p = np.asarray(pvals, dtype=float)
        m = p.size
        order = np.argsort(p)
        ranked = p[order]

        qvals_ranked = ranked * m / (np.arange(1, m + 1))
        qvals_ranked = np.minimum.accumulate(qvals_ranked[::-1])[::-1]
        qvals_ranked = np.clip(qvals_ranked, 0.0, 1.0)

        qvals = np.empty_like(qvals_ranked)
        qvals[order] = qvals_ranked

        reject = qvals <= q
        return qvals, reject

    def make_decoder_and_model():
        decoder_z = nn.Sequential(
            nn.Linear(z_dim_zs1, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, data_dim)
        )

        encoders = nn.ModuleList([
            cEncoder(z_dim=z_dim_zs2, mapping=nn.Sequential()),
            cEncoder(z_dim=z_dim_zc1, mapping=nn.Sequential()),
        ])

        decoders_c = [
            nn.Sequential(
            nn.Linear(x.shape[1], hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, data_dim)
            )
            for x in (zs2_tr, zc1_tr)
        ]

        decoders_cz = []

        decoder = Decoder_multiple_latents(
            data_dim, n_covariates,
            grid_z, grid_c,
            decoder_z, decoders_c, decoders_cz,
            has_feature_level_sparsity=False, p1=0.1, p2=0.1, p3=0.1,
            lambda0=args.lambda0, penalty_type=args.penalty_type,
            device=device,
        ).to(device)

        model = CVAE_multiple_latent_spaces_with_covariates(
            encoders, decoder, lr=args.decoder_lr, device=device
        )

        return decoder, model

    def load_pool_from_arrays(*arrays_np_or_torch):
        pools = []
        for a in arrays_np_or_torch:
            if isinstance(a, torch.Tensor):
                pools.append(a.detach().cpu().numpy())
            else:
                pools.append(np.asarray(a))
        Z = np.concatenate(pools, axis=0)
        if Z.ndim != 2:
            Z = Z.reshape(Z.shape[0], -1)
        return Z.astype(np.float32, copy=False)

    def sample_empirical_pool(Z_pool: np.ndarray, N: int, seed: int):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, Z_pool.shape[0], size=N)
        return Z_pool[idx].copy()

    def make_loader(Y, z1, z2, c, batch_size, shuffle):
        ds = TensorDataset(Y, z1, z2, c)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    def train_decoder_on_loader(model, data_loader_train):
        loss_values, integrals, integrals_dict, overfit_history = model.optimize(
            data_loader_train,
            augmented_lagrangian_lr=0.01,
            n_iter=config.iters,
            verbose=False,
        )
        return loss_values, integrals

    def eval_varexp_from_loader(decoder, data_loader, z1_full, z2_full, c_full, divide_by_total_var: bool):
        decoder.eval()
        with torch.no_grad():
            Y_err_list = []
            for batch in data_loader:
                Y_b, z1_b, z2_b, c_b = batch
                Y_pred = decoder(z1_b, [z2_b, c_b])
                Y_err_list.append(Y_b - Y_pred)
            Y_err = torch.cat(Y_err_list, dim=0)

        V = decoder.fraction_of_variance_explained(
            z1_full,
            [z2_full, c_full],
            Y_error=Y_err,
            divide_by_total_var=divide_by_total_var,
        ).detach().cpu().numpy()

        return V  # shape (D, Kc)

    def retrain_empirical_null_test_global(
        tested_component: str,  # "z1" or "zs2" or "zc1"
        Y_tr, z1_tr, z2_tr, c_tr,
        Y_val, z1_val, z2_val, c_val,
        Y_te, z1_te, z2_te, c_te,
        Z_pool_np: np.ndarray = None,  # only for "pool"
        n_null: int = 500,
        seed: int = 0,
        divide_by_total_var: bool = False,
        null_mode: str = "null2",  # "null2" or "perm" or "pool"
        fdr_q: float = 0.05,
    ):
        comp_to_idx = {"z1": 0, "zs2": 1, "zc1": 2}
        if tested_component not in comp_to_idx:
            raise ValueError("tested_component must be 'z1', 'zs2', or 'zc1'")
        if null_mode not in {"null2", "perm", "pool"}:
            raise ValueError("null_mode must be one of: 'null2', 'perm', 'pool'")
        k = comp_to_idx[tested_component]

        # ---------------- Observed fit (real pipeline) ----------------
        torch.manual_seed(seed)
        decoder_obs, model_obs = make_decoder_and_model()

        loader_tr_obs = make_loader(Y_tr, z1_tr, z2_tr, c_tr, config.batch_size, shuffle=True)
        train_decoder_on_loader(model_obs, loader_tr_obs)

        # Hyperparameter tuning logic (if needed) goes here

        loader_te_obs = make_loader(Y_te, z1_te, z2_te, c_te, config.batch_size, shuffle=False)
        V_obs = eval_varexp_from_loader(decoder_obs, loader_te_obs, z1_te, z2_te, c_te, divide_by_total_var)

        V_obs_k = V_obs[:, k].astype(np.float64)
        T_obs = float(V_obs_k.mean())

        # ---------------- Null distribution ----------------
        T_null = np.zeros(n_null, dtype=np.float64)
        D = int(V_obs.shape[0])
        Vnull_k = np.zeros((n_null, D), dtype=np.float64)

        Ntr = int(Y_tr.shape[0])
        Nval = int(Y_val.shape[0])
        Nte = int(Y_te.shape[0])

        # pick which latent is under test
        if tested_component == "z1":
            Ztr_real, Zval_real, Zte_real = z1_tr, z1_val, z1_te
            d_expected = int(z1_tr.shape[1])
        elif tested_component == "zs2":
            Ztr_real, Zval_real, Zte_real = z2_tr, z2_val, z2_te
            d_expected = int(z2_tr.shape[1])
        else:
            Ztr_real, Zval_real, Zte_real = c_tr, c_val, c_te
            d_expected = int(c_tr.shape[1])

        if null_mode == "null2":
            mu = Ztr_real.mean(dim=0, keepdim=True)
            sd = Ztr_real.std(dim=0, unbiased=False, keepdim=True)
            sd = torch.clamp(sd, min=1e-8)

        if null_mode == "pool":
            if Z_pool_np is None:
                raise ValueError("Z_pool_np must be provided when null_mode='pool'")
            d_pool = int(Z_pool_np.shape[1])
            if d_pool != d_expected:
                raise ValueError(f"Pool dim {d_pool} != expected {d_expected} for {tested_component}")

        for b in tqdm(range(n_null), desc=f"{tested_component} | {null_mode} nulls", leave=True):

            if null_mode == "null2":
                torch.manual_seed(seed + 10000 + b)
                Ztr_b = (mu + sd * torch.randn(Ntr, d_expected, device=device)).detach()
                Zval_b = (mu + sd * torch.randn(Nval, d_expected, device=device)).detach()
                Zte_b = (mu + sd * torch.randn(Nte, d_expected, device=device)).detach()

            elif null_mode == "perm":
                # MiMeNet-style: permute within each split
                g_tr = torch.Generator(device=device).manual_seed(seed + 11000 + b)
                idx_tr = torch.randperm(Ntr, generator=g_tr, device=device)
                Ztr_b = Ztr_real[idx_tr].detach()

                g_val = torch.Generator(device=device).manual_seed(seed + 11500 + b)
                idx_val = torch.randperm(Nval, generator=g_val, device=device)
                Zval_b = Zval_real[idx_val].detach()

                g_te = torch.Generator(device=device).manual_seed(seed + 12000 + b)
                idx_te = torch.randperm(Nte, generator=g_te, device=device)
                Zte_b = Zte_real[idx_te].detach()

            else:  # pool
                Ztr_np = sample_empirical_pool(Z_pool_np, Ntr, seed=seed + 10000 + b)
                Zval_np = sample_empirical_pool(Z_pool_np, Nval, seed=seed + 15000 + b)
                Zte_np = sample_empirical_pool(Z_pool_np, Nte, seed=seed + 20000 + b)
                Ztr_b = torch.from_numpy(Ztr_np).to(device)
                Zval_b = torch.from_numpy(Zval_np).to(device)
                Zte_b = torch.from_numpy(Zte_np).to(device)

            # substitute tested component in each split
            if tested_component == "z1":
                z1_tr_b, z2_tr_b, c_tr_b = Ztr_b, z2_tr, c_tr
                z1_val_b, z2_val_b, c_val_b = Zval_b, z2_val, c_val
                z1_te_b, z2_te_b, c_te_b = Zte_b, z2_te, c_te
            elif tested_component == "zs2":
                z1_tr_b, z2_tr_b, c_tr_b = z1_tr, Ztr_b, c_tr
                z1_val_b, z2_val_b, c_val_b = z1_val, Zval_b, c_val
                z1_te_b, z2_te_b, c_te_b = z1_te, Zte_b, c_te
            else:  # zc1
                z1_tr_b, z2_tr_b, c_tr_b = z1_tr, z2_tr, Ztr_b
                z1_val_b, z2_val_b, c_val_b = z1_val, z2_val, Zval_b
                z1_te_b, z2_te_b, c_te_b = z1_te, z2_te, Zte_b

            # retrain decoder on null TRAIN
            torch.manual_seed(seed + 30000 + b)
            decoder_b, model_b = make_decoder_and_model()
            loader_tr_b = make_loader(Y_tr, z1_tr_b, z2_tr_b, c_tr_b, config.batch_size, shuffle=True)
            train_decoder_on_loader(model_b, loader_tr_b)

            # If you tune hyperparams in reality, do it here using loader_val_b.
            # This function currently does not implement tuning.

            # evaluate on null TEST
            loader_te_b = make_loader(Y_te, z1_te_b, z2_te_b, c_te_b, config.batch_size, shuffle=False)
            V_b = eval_varexp_from_loader(decoder_b, loader_te_b, z1_te_b, z2_te_b, c_te_b, divide_by_total_var)

            V_b_k = V_b[:, k].astype(np.float64)
            Vnull_k[b, :] = V_b_k
            T_null[b] = float(V_b_k.mean())

        p_global = (1.0 + float((T_null >= T_obs).sum())) / (1.0 + n_null)
        p_per_met = (1.0 + (Vnull_k >= V_obs_k[None, :]).sum(axis=0).astype(np.float64)) / (1.0 + n_null)
        qvals, reject = bh_fdr(p_per_met, q=fdr_q)

        return {
            "tested_component": tested_component,
            "null_mode": null_mode,
            "T_obs": T_obs,
            "p_global": float(p_global),
            "p_per_metabolite": p_per_met,
            "q_per_metabolite": qvals,
            "reject_fdr": reject,
            "n_null": int(n_null),
            "fdr_q": float(fdr_q),
            "V_obs_k": V_obs_k,
            "V_obs_all": V_obs,  # all components (D, 3) for z1, zs2, zc1
            "T_null": T_null,
            "Vnull_k": Vnull_k,
            "metabolite_names": metabolite_names,  # list of metabolite names
        }

    # Run permutation test for all three components: z1, zs2, zc1
    null_modes = ["perm"]
    tested_components = ["z1", "zs2", "zc1"]

    for comp in tested_components:
        for mode in null_modes:
            res = retrain_empirical_null_test_global(
                comp,
                Y_train, zs1_tr,  zs2_tr,  zc1_tr,
                Y_val,   zs1_val, zs2_val, zc1_val,
                Y_test,  zs1_te,  zs2_te,  zc1_te,
                n_null=args.n_permutations,
                seed=int(args.seed),
                divide_by_total_var=False,
                null_mode=mode,
                fdr_q=0.05,
            )

            print(f"Done {comp}_{mode}: p_global={res['p_global']:.6g}")

            out_prefix = f"/users/antonios/LEAF_revisit/LEAF/one_hold/pvalue/pval_runs_{comp}"

            np.savez_compressed(
                f"{out_prefix}_{mode}.npz",
                tested_component=res["tested_component"],
                null_mode=res["null_mode"],
                n_null=res["n_null"],
                fdr_q=res["fdr_q"],
                T_obs=res["T_obs"],
                p_global=res["p_global"],
                V_obs_k=res["V_obs_k"],
                V_obs_all=res["V_obs_all"],  # (D, 3) variance for all components: z1, zs2, zc1
                T_null=res["T_null"],
                Vnull_k=res["Vnull_k"],
                p_per_metabolite=res["p_per_metabolite"],
                q_per_metabolite=res["q_per_metabolite"],
                reject_fdr=res["reject_fdr"].astype(np.uint8),
                metabolite_names=np.array(res["metabolite_names"], dtype=object),
            )

            print(f"Saved {comp} {mode}: p_global={res['p_global']:.6g}")

            # ---------------- FDR summary ----------------
            q = res["q_per_metabolite"]
            p = res["p_per_metabolite"]
            rej = res["reject_fdr"].astype(bool)
            v = res["V_obs_k"]

            n_sig = int(rej.sum())
            print(f"[{comp}] FDR @ q={res['fdr_q']}: {n_sig}/{rej.size} metabolites significant")

            if n_sig > 0:
                order = np.lexsort((-v, q))
                top = order[:min(20, rej.size)]

                print("Top metabolites by q-value (showing up to 20):")
                print("idx\tp\tq\tV_obs_k\tsig")
                for j in top:
                    print(f"{j}\t{p[j]:.6g}\t{q[j]:.6g}\t{v[j]:.6g}\t{int(rej[j])}")

                if "metabolite_names" in globals() and metabolite_names is not None:
                    sig_idx = np.where(rej)[0]
                    sig_order = sig_idx[np.argsort(q[sig_idx])]
                    print(f"[{comp}] Significant metabolites (sorted by q):")
                    for j in sig_order[:50]:
                        print(f"{metabolite_names[j]}\tp={p[j]:.6g}\tq={q[j]:.6g}\tV_obs_k={v[j]:.6g}")
            else:
                print(f"[{comp}] No metabolites pass BH-FDR at this threshold.")


    decoder, model = make_decoder_and_model()
    train_decoder_on_loader(model, data_loader_train)

        # ------------------------------------------------------------------
    # Switch to evaluation mode BEFORE computing errors / variance
    # ------------------------------------------------------------------

    decoder.eval()
    model.eval()

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


    Y_error_tr, _ = eval_split(decoder, data_loader_train_eval)
    varexp_tr = decoder.fraction_of_variance_explained(
        zs1_tr, [zs2_tr, zc1_tr], Y_error=Y_error_tr, divide_by_total_var=False
    ).detach().cpu()

    # Val
    Y_error_val, _ = eval_split(decoder, data_loader_val)
    varexp_val = decoder.fraction_of_variance_explained(
        zs1_val, [zs2_val, zc1_val], Y_error=Y_error_val, divide_by_total_var=False
    ).detach().cpu()

    # Test
    Y_error_te, _ = eval_split(decoder, data_loader_test)
    varexp_te = decoder.fraction_of_variance_explained(
        zs1_te, [zs2_te, zc1_te], Y_error=Y_error_te, divide_by_total_var=False
    ).detach().cpu()

    # assert varexp_te.shape[1] == 4, (
    #     f"Expected 4 columns (z1, zs2, zc1, noise), got {varexp_te.shape[1]}"
    # )




    # Build variance tables
    df_train_long = build_variance_table(varexp_tr, Y_train,"train", metabolite_names=metabolite_names)
    df_val_long  = build_variance_table(varexp_val, Y_val, "val", metabolite_names=metabolite_names)
    df_test_long  = build_variance_table(varexp_te, Y_test, "test", metabolite_names=metabolite_names)

    df_both = pd.concat([df_train_long,df_val_long, df_test_long], ignore_index=True)
    df_piv  = df_both.pivot_table(
        index=["outcome","component"],
        columns="split",
        values="est_fraction",
        aggfunc="mean",
    ).reset_index()



    # assert varexp_te.shape[1] == 4, (
    #     f"Expected 4 columns (z1, zs2, zc1, noise), got {varexp_te.shape[1]}"
    # )









    print("Var(Y_train) min/mean/max:",
        float(Y_train.var(dim=0, unbiased=False).min()),
        float(Y_train.var(dim=0, unbiased=False).mean()),
        float(Y_train.var(dim=0, unbiased=False).max()))


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
    avg_corr_te, avg_cov_te, cov_stats_te = compute_avg_covcorr_from_decoder(
        decoder, zs1_te, zs2_te, zc1_te, Y_in=Y_test
    )

    print("Train cov share stats:", cov_stats_tr)
    print("Val   cov share stats:", cov_stats_val)
    print("Test  cov share stats:", cov_stats_te)
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

    avg_corr_te, avg_cov_te, cov_share_stats_te = compute_avg_covcorr_from_decoder(
        decoder,
        zs1_te.to(device),
        zs2_te.to(device),
        zc1_te.to(device),
        Y_test  # <-- important if you want covariance share relative to Var(Y)
    )

    print("\nTEST component correlation:\n", avg_corr_te)
    print("max |off-diag corr|:", float(np.max(np.abs(avg_corr_te[~np.eye(3, dtype=bool)]))))
    print("\nTEST component covariance:\n", avg_cov_te)
    print("max |off-diag cov|:", float(np.max(np.abs(avg_cov_te[~np.eye(3, dtype=bool)]))))
    print("\nTEST covariance share stats (2*sum cov / Var(y)):\n", cov_share_stats_te)


    # df_piv should have columns: outcome, component, train, val, test (depending on what you included)
    # After you build df_both = concat([df_train_long, df_val_long, df_test_long])

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
    out_dir = "/users/antonios/LEAF_revisit/LEAF/one_hold/pvalue"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"5000_variance_comparison_{MODEL_NAME}_{run_id}.csv")
    final_table.to_csv(out_path, index=False)

    print("Wrote:", out_path)
    print(final_table.head(12))


if __name__ == "__main__":
    main()