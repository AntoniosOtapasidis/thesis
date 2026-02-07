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


from ND.encoder import cEncoder
from ND.decoder import Decoder
from ND.CVAE import CVAE
from ND.helpers import expand_grid
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.uniform import Uniform

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
    cmib_optim = optim.Adam(cmib.parameters(), lr=args.lr_s1)
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

    disen_optim = optim.Adam(disen.parameters(), lr=args.lr_s2)
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


# def build_variance_table(varexp, Y_torch, split, metabolite_names=None):
#     comp_keys = ["z1", "zs2", "zc1"]
#     rows = []
#     varexp_np = varexp.numpy()
#     total_var = Y_torch.var(dim=0)

#     for i in range(Y_torch.shape[1]):
#         outcome = metabolite_names[i] if metabolite_names is not None else f"metabolite_{i}"
#         outcome_var = float(total_var[i].item())

#         # take only the first three components (z1, zs2, zc1)
#         est_abs = varexp_np[i][:len(comp_keys)]
#         est_sum = float(np.sum(est_abs))

#         # compute fractions for each component
#         est_fracs = [float(v)/outcome_var if outcome_var > 0 else 0.0 for v in est_abs]

#         # compute noise as leftover variance fraction
#         est_noise_abs = max(outcome_var - est_sum, 0.0)
#         est_noise_frac = est_noise_abs/outcome_var if outcome_var > 0 else 0.0

#         # add component rows
#         for key, est_f in zip(comp_keys, est_fracs):
#             rows.append({
#                 "outcome": outcome,
#                 "component": key,
#                 "split": split,
#                 "est_fraction": est_f,
#             })

#         # add noise row
#         rows.append({
#             "outcome": outcome,
#             "component": "noise",
#             "split": split,
#             "est_fraction": est_noise_frac,
#         })

#     return pd.DataFrame(rows)



def build_variance_table(varexp, Y_torch,gt, split):
    comp_keys = ["z1", "zs2", "zc1"]
    rows = []
    varexp_np = varexp.numpy()
    total_var = Y_torch.var(dim=0)
    gt_map = {"z1": "Z1", "zs2": "Z2", "zc1": "Zs"}

    for i in range(Y_torch.shape[1]):
        #outcome_name = str(metabolite_names[i]) if metabolite_names is not None else f"metabolite_{i}"
        outcome_name = f'outcome_{i +1}'
        outcome_var = float(total_var[i].item())
        est_abs = varexp_np[i][:len(comp_keys)]
        est_sum = float(np.sum(est_abs))

        # compute fractions for each component
        est_fracs = [float(v)/outcome_var if outcome_var > 0 else 0.0 for v in est_abs]
        est_noise_abs = max(outcome_var - est_sum, 0.0)
        est_noise_frac = est_noise_abs/outcome_var if outcome_var > 0 else 0.0
        # if callable(metabolite_names):
        #     raise TypeError("metabolite_names is a function; pass a list/array of names.")

        # # add component rows
        for key, est_f in zip(comp_keys, est_fracs):
            rows.append({
                "outcome": outcome_name,
                "component": key,
                "gt_fraction": float(gt["shares"][gt_map[key]][i]),
                "split": split,
                "est_fraction": est_f,
            })
        # add noise row
        rows.append({
            "outcome": outcome_name,
            "component": "noise",
            "gt_fraction": float(gt["shares"]["noise"][i]),
            "split": split,
            "est_fraction": est_noise_frac,
         })

    return pd.DataFrame(rows)



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
    parser.add_argument('--drop_scale', type=float, default=2)
    parser.add_argument('--debug_mode', type=str2bool, default=False)
    parser.add_argument('--device', type=str, default="0")

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
            "R/synthetic_generation/X1_bacteria_synthetic_CLR.csv"
        )
        df_X2 = pd.read_csv(
            "R/synthetic_generation/X2_viruses_synthetic_CLR.csv"
        )
        df_Y = pd.read_csv(
            "R/synthetic_generation/Y_metabolites_log_synthetic_complex_RA_COPSAC.csv"
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
            "R/synthetic_generation/GT_virome_variance_shares_complex_COPSAC.csv"
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
            "R/synthetic_generation/X1_bacteria_synthetic_ILR_final.csv"
        )
        df_X2 = pd.read_csv(
            "R/synthetic_generation/X2_viruses_synthetic_ILR_final.csv"
        )
        df_Y = pd.read_csv(
            "R/synthetic_generation/Y_metabolites_log_synthetic_complex_RA_COPSAC.csv"
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
            "R/synthetic_generation/GT_virome_variance_shares_complex_COPSAC.csv"
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
            df = pd.read_csv(path, sep='\t', index_col=0)
            # normalize index to comparable strings
            df.index = df.index.astype(str).str.strip()
            # drop accidental first data column if present
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
            return df

#       Y values need to be independent with its each other. That is why we are going with log trasnformation
        #df_Y  = pd.read_csv('/users/antonios/code/metabolome.CLR.LEAF.tsv', sep = "\t", index_col = 0)
        df_X1 = pd.read_csv('/users/antonios/code/microbiome.CLR.LEAF.tsv', sep = "\t", index_col = 0)
        df_X2 = pd.read_csv('/users/antonios/code/virome.CLR.LEAF.tsv', sep = "\t", index_col = 0)
        df_Y  = pd.read_csv('/users/antonios/data/metabolites.known.tsv', sep = "\t", index_col = 0)

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
            f"_seed{args.seed}_beta{args.beta}_hsic{args.hsic_weight}"
        )
        wandb.config.update({
                    "seed": args.seed,
                    "model_name": MODEL_NAME,
                    "dims": args.dim_info # Now you can safely pass the dictionary
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

        # 1) Pick train/test indices 
        train_tmp, test_tmp = torch.utils.data.random_split(
            tmp_dataset,
            [int(0.8 * num_data), num_data - int(0.8 * num_data)],
            generator=torch.Generator().manual_seed(0),
        )

        indices = np.array(train_tmp.indices)
        all_idx = np.arange(num_data)
        test_idx = np.setdiff1d(all_idx, indices)

        X1_full = X1_np.copy()
        X2_full = X2_np.copy()
        # === Standardize CLR features for X1 and X2 using train indices only ===
        X1_full = modalities_raw[0]  # [N, dim_X1], CLR already
        X2_full = modalities_raw[1]  # [N, dim_X2], CLR already

        # Bacteria
        X1_mean = X1_full[indices].mean(axis=0, keepdims=True)
        X1_std  = X1_full[indices].std(axis=0, ddof=0, keepdims=True)
        X1_std[X1_std == 0] = 1.0
        X1_full = (X1_full - X1_mean) / X1_std
        X1_full = X1_full.astype(np.float32)
        # Viruses
        X2_mean = X2_full[indices].mean(axis=0, keepdims=True)
        X2_std  = X2_full[indices].std(axis=0, ddof=0, keepdims=True)
        X2_std[X2_std == 0] = 1.0
        X2_full = (X2_full - X2_mean) / X2_std
        X2_full = X2_full.astype(np.float32)

        # THIS IS FOR THE MULTIMODAL DATASET
        # # Rebuild data with standardized CLR inputs
        # data = np.stack(
        #     [X1_full.astype(np.float32), X2_full.astype(np.float32)],
        #     axis=0
        # )

        #ΤHIS IS FOR THE MULTIOMICS DATASET
        modalities = (X1_full, X2_full)

        # right after MODEL_NAME is constructed and before Y_full = np.load(...)
        np.save(f"data/disentangled/all_targets_{MODEL_NAME}.npy", Y_np)

        # 2) Load and normalize Y using train indices only
        # AFTER:
        Y_full = np.load(f"data/disentangled/all_targets_{MODEL_NAME}.npy").astype(np.float32)

        # Only log-transform real COPSAC (raw metabolite concentrations)
        # Synthetic data is already in log-space from R generation
        if args.dataset == "microbiome":
            Y_log = np.log1p(Y_full)  # Real data: raw → log
        else:
            Y_log = Y_full  # Synthetic: already log-space

        Y_mean = Y_log[indices].mean(axis=0, keepdims=True)
        Y_std  = Y_log[indices].std(axis=0, ddof=0, keepdims=True)
        Y_std[Y_std == 0] = 1.0

        Y_final = (Y_log - Y_mean) / Y_std  # shape [num_samples, num_targets]


        # Optional saves
        np.save(f"data/disentangled/all_targets_log_{MODEL_NAME}.npy", Y_final)
        np.save(f"data/disentangled/targets_log_{MODEL_NAME}_train.npy", Y_final[indices])
        np.save(f"data/disentangled/targets_log_{MODEL_NAME}_test.npy",  Y_final[test_idx])

        # 3) Build the REAL dataset with normalized Y as single label block
        #dataset = MultimodalDataset(data, targets1, targets2, targets3)
        dataset = MultiomicDataset(total_data=modalities, total_labels1=Y_final)


        # 4) Now create subsets from THIS dataset
        from torch.utils.data import Subset, DataLoader

        train_dataset = Subset(dataset, indices.tolist())
        test_dataset  = Subset(dataset, test_idx.tolist())
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset}")

    # # Train test split with fixed random seed
    # #num_data = len(dataset) 
    # num_data = data.shape[1]
    # g = torch.Generator().manual_seed(0)
    # train_dataset, test_dataset = torch.utils.data.random_split(
    #     dataset,
    #     [int(0.8 * num_data), num_data - int(0.8 * num_data)],
    #     generator=torch.Generator().manual_seed(0)
    # )

    # indices = np.array(train_dataset.indices)
    # all_idx = np.arange(num_data)
    # test_idx = np.setdiff1d(all_idx, indices)

    # # #Log transform and standardize Y based on training set stats
    # # # 2) Load full Y, log1p-transform, and standardize using train stats
    # Y_full = np.load(f'data/disentangled/all_targets_{MODEL_NAME}.npy').astype(np.float32)
    # # # Y_log = np.log1p(Y_full)
    # Y_mean = Y_full[indices].mean(axis=0, keepdims=True)
    # Y_std  = Y_full[indices].std(axis=0, ddof=0, keepdims=True)
    # Y_std[Y_std == 0] = 1.0  
    # Y_logstd = (Y_full - Y_mean) / Y_std   
    # Y_final = Y_logstd 
    # np.save(f"data/disentangled/all_targets_log_{MODEL_NAME}_train.npy", Y_final)
    # np.save(f"data/disentangled/targets_log_{MODEL_NAME}_train.npy", Y_final[indices])
    # np.save(f"data/disentangled/targets_log_{MODEL_NAME}_test.npy",  Y_final[test_idx])
  
    # #np.save(f"data/disentangled/targets_{MODEL_NAME}_train.npy", Y[indices])
    # #np.save(f"data/disentangled/targets_{MODEL_NAME}_test.npy",  Y[test_idx])
    # #Y_final= np.load(f'data/disentangled/all_targets_{MODEL_NAME}.npy').astype(np.float32)

    # # # 3) Build dataset with log-standardized Y COPSAC Data
    # # modalities = [X1, X2]
    # # dataset = MultiomicDataset(total_data=modalities, total_labels1=Y_final)

    # # # 4) Create subsets using the fixed indices 
    # from torch.utils.data import Subset, DataLoader, TensorDataset

    # train_dataset = Subset(dataset, indices.tolist())
    # test_dataset  = Subset(dataset, test_idx.tolist())

    # # I do this because I loaded the Y data aas separate columns which I amalgamate afterwards because of linear probing (lnear regression per column instead of ridge regression)

    # def collate_modalities_and_labels(batch):
    #     # batch: list of tuples (X1, X2, y1, y2, ..., yK)
    #     num_modalities = 2
    #     num_items = len(batch[0])
    #     K = num_items - num_modalities

    #     x1 = torch.stack([b[0] for b in batch], dim=0)
    #     x2 = torch.stack([b[1] for b in batch], dim=0)

    #     if K > 0:
    #         Y = torch.stack(
    #             [torch.stack([b[num_modalities + j] for j in range(K)], dim=0) for b in batch],
    #             dim=0
    #         )  # [B, K]
    #         return x1, x2, Y
    #     else:
    #         return x1, x2

    # train_loader = DataLoader(
    #     train_dataset, shuffle=False, drop_last=False, batch_size=args.batch_size,
    #     collate_fn=collate_modalities_and_labels
    # )
    # test_loader  = DataLoader(
    #     test_dataset,  shuffle=False, drop_last=False, batch_size=args.batch_size,
    #     collate_fn=collate_modalities_and_labels
    # )





######################################################################## 
########################Synthetic Data Modeling ########################
########################################################################


    train_loader = DataLoader(
        train_dataset, shuffle=False, drop_last=False, batch_size=args.batch_size,
    )
    test_loader  = DataLoader(
        test_dataset,  shuffle=False, drop_last=False, batch_size=args.batch_size,
    )

    # if args.dataset == "microbiome":
    #     x1_shape = dataset[0][0].shape
    #     x2_shape = dataset[0][1].shape
    #     print(f"Number of modalities : 2")
    #     print(f"Modality 1 (X1) shape : {x1_shape}")
    #     print(f"Modality 2 (X2) shape : {x2_shape}")
    #     print(f"Target (Y_matrix) shape : {Y_final.shape}")
    # else:
    #     print(f"Number of modalities : 2")
    #     print(f"Modality 1 (X1) shape : {X1.shape}")
    #     print(f"Modality 2 (X2) shape : {X2.shape}")
    #     print(f"Target (Y_matrix) shape : {Y_final.shape}")


    # 5) Train models — these now see log-standardized Y via the dataset
    train_mp(args.beta, train_loader, test_loader, train_dataset, test_dataset, args)

    print("CALLING train_step2", flush=True)
    logs, disen = train_step2(args, train_loader, test_loader, train_dataset, test_dataset)
    print("RETURNED from train_step2", flush=True)


    # 6) Gather embeddings (unchanged)
    zs1_tr, zs2_tr, zc1_tr, zc2_tr = gather_embeddings(disen, train_loader, device=device)
    zs1_te, zs2_te, zc1_te, zc2_te = gather_embeddings(disen, test_loader, device=device)
    np.save(f'data/disentangled/zs1_{MODEL_NAME}.npy', zs1_tr)
    np.save(f'data/disentangled/zs2_{MODEL_NAME}.npy', zs2_tr)
    np.save(f'data/disentangled/zc1_{MODEL_NAME}.npy', zc1_tr)
    np.save(f'data/disentangled/zc2_{MODEL_NAME}.npy', zc2_tr)
    np.save(f'data/disentangled/zs1_test_{MODEL_NAME}.npy', zs1_te)
    np.save(f'data/disentangled/zs2_test_{MODEL_NAME}.npy', zs2_te)
    np.save(f'data/disentangled/zc1_test_{MODEL_NAME}.npy', zc1_te)
    np.save(f'data/disentangled/zc2_test_{MODEL_NAME}.npy', zc2_te)

    # # ========================================================================
    # # VISUALIZE EMBEDDING SPACES + SAVE FOR LATER (no rerun needed)
    # # ========================================================================
    # print(f"\n{'='*80}")
    # print(f"VISUALIZING EMBEDDING SPACES")
    # print(f"{'='*80}\n")




    # # Combine train + test
    # zs1_all = np.vstack([zs1_tr, zs1_te])
    # zs2_all = np.vstack([zs2_tr, zs2_te])
    # zc1_all = np.vstack([zc1_tr, zc1_te])

    # split_labels = np.array(["train"] * len(zs1_tr) + ["test"] * len(zs1_te))

    # # Y in the same order as zs*_all
    # Y_all_np = np.vstack([Y_final[indices], Y_final[test_idx]])  # [N, 32]


    # def reduce_umap(X, n_components=2, random_state=42, n_neighbors=30, min_dist=0.1):
    #     reducer = umap.UMAP(
    #         n_components=n_components,
    #         random_state=random_state,
    #         n_neighbors=n_neighbors,
    #         min_dist=min_dist,
    #         metric="euclidean",
    #     )
    #     return reducer.fit_transform(X)


    # def plot_latent_colored_by_all_metabolites(
    #     emb2d, Y_all_np, title_prefix, out_path,
    #     metabolite_names=None,
    #     ncols=8, point_size=12
    # ):
    #     """
    #     emb2d: [N, 2]
    #     Y_all_np: [N, n_met]
    #     metabolite_names: list[str] length = n_met (optional)
    #     """
    #     n_met = Y_all_np.shape[1]
    #     nrows = int(math.ceil(n_met / ncols))

    #     fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows), squeeze=False)

    #     for j in range(n_met):
    #         r, c = divmod(j, ncols)
    #         ax = axes[r, c]

    #         sc = ax.scatter(
    #             emb2d[:, 0], emb2d[:, 1],
    #             c=Y_all_np[:, j],
    #             cmap="viridis",
    #             s=point_size,
    #             alpha=0.8,
    #             edgecolors="none"
    #         )

    #         if metabolite_names is not None and j < len(metabolite_names):
    #             ax.set_title(str(metabolite_names[j]), fontsize=9)
    #         else:
    #             ax.set_title(f"met{j+1}", fontsize=10)

    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.01)

    #     for j in range(n_met, nrows * ncols):
    #         r, c = divmod(j, ncols)
    #         axes[r, c].axis("off")

    #     fig.suptitle(title_prefix, fontsize=14)
    #     plt.tight_layout()
    #     plt.savefig(out_path, dpi=300, bbox_inches="tight")
    #     plt.close()
    #     print(f"✓ Saved: {out_path}")

    # # ---- after you compute zs1_umap, zs2_umap, zc1_umap and Y_all_np ----
    # print("Computing UMAP reductions...")
    # zs1_umap = reduce_umap(zs1_all)
    # zs2_umap = reduce_umap(zs2_all)
    # zc1_umap = reduce_umap(zc1_all)

    # # ------------------------------------------------------------------------
    # # SAVE the projections so you can re-plot later without rerunning anything
    # # ------------------------------------------------------------------------
    # np.savez(
    #     os.path.join(out_dir, f"embeddings_2d_{MODEL_NAME}_{run_id}.npz"),
    #     zs1_umap=zs1_umap,
    #     zs2_umap=zs2_umap,
    #     zc1_umap=zc1_umap,
    #     split_labels=split_labels,
    #     Y_all_np=Y_all_np,
    # )
    # print("✓ Saved 2D projections + labels to NPZ")
    # # ------------------------------------------------------------------------
    # # 1) Plot train vs test colored (3 panels)
    # # ------------------------------------------------------------------------
    # fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # embeddings = [
    #     (zs1_umap, "Zs1", "UMAP1", "UMAP2"),
    #     (zs2_umap, "Zs2", "UMAP1", "UMAP2"),
    #     (zc1_umap, "Zc1", "UMAP1", "UMAP2"),
    # ]
    # for idx, (emb, title, xlabel, ylabel) in enumerate(embeddings):
    #     ax = axes[idx]
    #     for split, color in [("train", "steelblue"), ("test", "coral")]:
    #         mask = split_labels == split
    #         ax.scatter(emb[mask, 0], emb[mask, 1], c=color, label=split, alpha=0.6, s=20, edgecolors="none")
    #     ax.set_xlabel(xlabel, fontsize=11)
    #     ax.set_ylabel(ylabel, fontsize=11)
    #     ax.set_title(f"{title} (colored by split)", fontsize=12)
    #     ax.legend()
    #     ax.grid(alpha=0.3)

    # plt.tight_layout()
    # split_plot_path = os.path.join(out_dir, f"embedding_split_{MODEL_NAME}_{run_id}.png")
    # plt.savefig(split_plot_path, dpi=300, bbox_inches="tight")
    # plt.close()
    # print(f"✓ Saved: {split_plot_path}")

    # plot_latent_colored_by_all_metabolites(
    #     zs1_umap, Y_all_np,
    #     title_prefix="Zs1 colored by all metabolites",
    #     out_path=os.path.join(out_dir, f"zs1_all_metabolites_{MODEL_NAME}_{run_id}.png"),
    #     metabolite_names=metabolite_names,
    # )

    # plot_latent_colored_by_all_metabolites(
    #     zs2_umap, Y_all_np,
    #     title_prefix="Zs2 colored by all metabolites",
    #     out_path=os.path.join(out_dir, f"zs2_all_metabolites_{MODEL_NAME}_{run_id}.png"),
    #     metabolite_names=metabolite_names,
    # )

    # plot_latent_colored_by_all_metabolites(
    #     zc1_umap, Y_all_np,
    #     title_prefix="Zc1 colored by all metabolites",
    #     out_path=os.path.join(out_dir, f"zc1_all_metabolites_{MODEL_NAME}_{run_id}.png"),
    #     metabolite_names=metabolite_names,
    # )



    # # ------------------------------------------------------------------------
    # # 3) Train vs test separability AUC per latent, and SAVE it
    # # ------------------------------------------------------------------------


    # def split_separability_auc(Z, split_labels, n_splits=5, random_state=42):
    #     y = (split_labels == "test").astype(int)
    #     cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    #     clf = make_pipeline(
    #         StandardScaler(),
    #         LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=1)
    #     )

    #     aucs = []
    #     for tr_idx, te_idx in cv.split(Z, y):
    #         clf.fit(Z[tr_idx], y[tr_idx])
    #         p = clf.predict_proba(Z[te_idx])[:, 1]
    #         aucs.append(roc_auc_score(y[te_idx], p))

    #     aucs = np.array(aucs, dtype=float)
    #     return float(aucs.mean()), float(aucs.std(ddof=0))

    # rows = []
    # for name, Z in [("zs1", zs1_all), ("zs2", zs2_all), ("zc1", zc1_all)]:
    #     mean_auc, std_auc = split_separability_auc(Z, split_labels)
    #     rows.append({"latent": name, "auc_mean": mean_auc, "auc_std": std_auc})

    # df_auc = pd.DataFrame(rows)
    # print("\n[Split separability AUC (train vs test)]")
    # print(df_auc)

    # auc_csv_path = os.path.join(out_dir, f"split_separability_auc_{MODEL_NAME}_{run_id}.csv")
    # df_auc.to_csv(auc_csv_path, index=False)
    # print(f"✓ Saved: {auc_csv_path}")


    # print("\nTrain vs test split separability (AUC, higher is worse):")
    # for name, Z in [("Zs1", zs1_all), ("Zs2", zs2_all), ("Zc1", zc1_all)]:
    #     mean_auc, std_auc = split_separability_auc(Z, split_labels, n_splits=5, random_state=42)
    #     print(f"  {name}: AUC = {mean_auc:.3f} ± {std_auc:.3f}")

    # 7) For the variance-explained CVAE, directly use the same processed Y (avoid reloading originals)
    Y_final = np.load(f"data/disentangled/all_targets_log_{MODEL_NAME}.npy").astype(np.float32)
    Y_train = torch.from_numpy(Y_final[indices]).float().to(device)
    Y_test  = torch.from_numpy(Y_final[test_idx]).float().to(device)
    #c_train = torch.from_numpy(train_confounder).float().to(device)
    #c_test  = torch.from_numpy(test_confounder).float().to(device)
    #Y_all = torch.from_numpy(np.load(f'data/disentangled/all_targets_{MODEL_NAME}.npy')).float().to(device)
    #Y_train = Y_all[indices]
    #Y_test  = Y_all[test_idx]
    zs1 = torch.from_numpy(np.load(f'data/disentangled/zs1_{MODEL_NAME}.npy')).float().to(device)
    zs2 = torch.from_numpy(np.load(f'data/disentangled/zs2_{MODEL_NAME}.npy')).float().to(device)
    zc1 = torch.from_numpy(np.load(f'data/disentangled/zc1_{MODEL_NAME}.npy')).float().to(device)
    zc2 = torch.from_numpy(np.load(f'data/disentangled/zc2_{MODEL_NAME}.npy')).float().to(device)
    zs1_test = torch.from_numpy(np.load(f'data/disentangled/zs1_test_{MODEL_NAME}.npy')).float().to(device)
    zs2_test = torch.from_numpy(np.load(f'data/disentangled/zs2_test_{MODEL_NAME}.npy')).float().to(device)
    zc1_test = torch.from_numpy(np.load(f'data/disentangled/zc1_test_{MODEL_NAME}.npy')).float().to(device)
    zc2_test = torch.from_numpy(np.load(f'data/disentangled/zc2_test_{MODEL_NAME}.npy')).float().to(device)

    # Residualize confounder latents (optional; here we DO residualize zc1/zc2; keep z1/zs2 as-is)
   # zc1_res_train, zc2_res_train = residualize_data(c_train, *[zc1, zc2], config=config, device=device)
    #zs1_res_train, zs2_res_train = zs1, zs2
   # zc1_res_test, zc2_res_test = residualize_data(c_test, *[zc1_test, zc2_test], config=config, device=device)
    #zs1_res_test, zs2_res_test = zs1_test, zs2_test

    # Build loaders for the variance explained CVAE
    dataset_train = TensorDataset(
        Y_train,
        zs1.to(device),
        zs2.to(device),
        zc1.to(device),
        #c_train.to(device),
    )
    data_loader_train = DataLoader(dataset_train, shuffle=False, batch_size=config.batch_size)
    dataset_test = TensorDataset(
        Y_test,
        zs1_test.to(device),
        zs2_test.to(device),
        zc1_test.to(device),
        #c_test.to(device),
    )
    data_loader_test = DataLoader(dataset_test, shuffle=False, batch_size=config.batch_size)

    # Fit variance explained model
    data_dim = Y_train.shape[1]
    hidden_dim = 50  # Reduced from 200 to prevent overfitting
    #z_dim = zs1.shape[1]
    n_covariates = 2
    num_samples = 177  # This is just dataset size, not grid size

    z_dim_zs1 = zs1.shape[1]
    z_dim_zs2 = zs2.shape[1]
    z_dim_zc1 = zc1.shape[1]


   # lim_val = 2.0
    #grid_z = (torch.rand(num_samples, z_dim_zs1, device=device) * 2 * lim_val) - lim_val
    #grid_cov = lambda x: (torch.rand(num_samples, x.shape[1], device=device) * 2 * lim_val) - lim_val
    #grid_c = [grid_cov(x) for x in (zs2, zc1)]

        # Use fixed prior grids (standard normal) to prevent overfitting
    # Grid size should be large enough for proper integration
    K = 177  # Much larger grid for better marginalization

    def rand_grid_prior(dim, K):
        # Sample from standard normal prior (no training data dependency)
        return torch.randn(K, dim, device=device).detach()

    grid_z = rand_grid_prior(zs1.shape[1], K)
    grid_c = [rand_grid_prior(zs2.shape[1], K), rand_grid_prior(zc1.shape[1], K)]



    decoder_z = nn.Sequential(
        nn.Linear(z_dim_zs1, hidden_dim),
        nn.Tanh(),
      #  nn.Dropout(p=0.1),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
      #  nn.Dropout(p=0.1),
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
      #  nn.Dropout(p=0.1),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
      #  nn.Dropout(p=0.1),
        nn.Linear(hidden_dim, data_dim)
    ) for x in (zs2, zc1)]

    decoders_cz = []
    decoder = Decoder_multiple_latents(
        data_dim, n_covariates,
        grid_z, grid_c,
        decoder_z, decoders_c, decoders_cz,
        has_feature_level_sparsity=False, p1=0.1, p2=0.1, p3=0.1,
        lambda0=1e3, penalty_type="MDMM",
        device=device
    )
    model = CVAE_multiple_latent_spaces_with_covariates(encoders, decoder, lr=0.0001, device=device)
    loss, integrals, integrals_dict, overfit_history = model.optimize(data_loader_train, n_iter=config.iters,augmented_lagrangian_lr=0.1) 
    cz_dx2_traces = None
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

    Y_error_tr, Y_pred_tr = eval_split(decoder, data_loader_train, n_covariates=2)
    confounders_tr = [zs2.to(device), zc1.to(device)]
    varexp_tr = decoder.fraction_of_variance_explained(
        zs1.to(device), confounders_tr, Y_error=Y_error_tr.to(device), divide_by_total_var=False
    ).cpu()
    Y_error_te, Y_pred_te = eval_split(decoder, data_loader_test, n_covariates=2)
    confounders_te = [zs2_test.to(device), zc1_test.to(device)]
    varexp_te = decoder.fraction_of_variance_explained(
        zs1_test.to(device), confounders_te, Y_error=Y_error_te.to(device), divide_by_total_var=False
    ).cpu()

    # --- Overfitting check (add this right after varexp_te) ---
    # 1) Reconstruction error gap
    train_mse = (Y_error_tr ** 2).mean().item()
    test_mse  = (Y_error_te ** 2).mean().item()
    print(f"Reconstruction MSE  train={train_mse:.4f}  test={test_mse:.4f}")

    # 2) Variance explained gap
    # If varexp_* is a tensor per-feature, summarise it
    train_varexp_mean = varexp_tr.mean().item()
    test_varexp_mean  = varexp_te.mean().item()
    print(f"VarExpl mean train={train_varexp_mean:.4f}  test={test_varexp_mean:.4f}")


    print("[DEBUG] Y_train shape:", Y_train.shape)  # (N_train, 100)
    print("[DEBUG] varexp_tr shape:", varexp_tr.shape)  # (100, 4) if 4 components

    #print("[DEBUG] first 5 GT shares for Z1/Z2/Zs/noise:")
    #print(df_gt.head())

    # Build variance tables
    df_train_long = build_variance_table(varexp_tr, Y_train,gt,"train")
    df_test_long  = build_variance_table(varexp_te, Y_test, gt,"test")
    df_both = pd.concat([df_train_long, df_test_long], ignore_index=True)
    df_piv  = df_both.pivot_table(
        index=["outcome","component"],
        columns="split",
        values="est_fraction",
        aggfunc="mean",
    ).reset_index()


    print("Var(Y_pred_tr) min/mean/max:", 
        float(Y_train.var(dim=0, unbiased=False).min()),
        float(Y_train.var(dim=0, unbiased=False).mean()),
        float(Y_train.var(dim=0, unbiased=False).max()))
    print("varexp_tr row-sum (first 10):", varexp_tr[:, :3].sum(1)[:10])
    print("any NaN in varexp_tr?", np.isnan(varexp_tr.detach().cpu().numpy()).any())


#Check for overfitting by correlating train and test variance explained
    corrs = []
    for i in range(varexp_tr.shape[1]):  # each component
        corr = np.corrcoef(varexp_tr[:, i], varexp_te[:, i])[0, 1]
        corrs.append(corr)

    df_corr = pd.DataFrame({
        'component': [f'comp{i+1}' for i in range(len(corrs))],
        'train_mean': varexp_tr.mean(0).numpy(),
        'test_mean': varexp_te.mean(0).numpy(),
        'corr_train_test': corrs
    })
    print(df_corr)



    if "train" in df_piv and "test" in df_piv:
        df_piv["est_fraction_mean"] = df_piv[["train","test"]].mean(axis=1)
    else:
        df_piv["est_fraction_mean"] = df_piv.get("train", df_piv.get("test"))

    #cols = ["outcome","component","train","test","est_fraction_mean"]
   
    #final_table = df_piv.reindex(columns=cols)


    gt_unique = df_train_long[["outcome","component","gt_fraction"]].drop_duplicates()
    final_table = df_piv.merge(gt_unique, on=["outcome","component"], how="left")

    cols = ["outcome","component","gt_fraction","train","test","est_fraction_mean"]
    #cols = ["outcome","component","train","test","est_fraction_mean"]

    final_table = final_table.reindex(columns=cols)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = "/users/antonios/LEAF_revisit/LEAF/COPSAC_clone/k_50/1000_kappa_/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"5000_CLR_final_variance_comparison_{MODEL_NAME}_{run_id}.csv"
    )
    final_table.to_csv(out_path, index=False)
    print("Wrote:", out_path)
    print(final_table.head(12))
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
            # Get the device from the decoder model
            device = next(decoder_obj.parameters()).device

            # Convert numpy arrays to tensors if needed and move to correct device
            if isinstance(zs1_in, np.ndarray):
                zs1_in = torch.from_numpy(zs1_in).float().to(device)
            if isinstance(zs2_in, np.ndarray):
                zs2_in = torch.from_numpy(zs2_in).float().to(device)
            if isinstance(zc1_in, np.ndarray):
                zc1_in = torch.from_numpy(zc1_in).float().to(device)

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
    # avg_corr_val, avg_cov_val, cov_stats_val = compute_avg_covcorr_from_decoder(
    #     decoder, zs1_val, zs2_val, zc1_val, Y_in=Y_val
    # )
    avg_corr_te, avg_cov_te, cov_stats_te = compute_avg_covcorr_from_decoder(
        decoder, zs1_te, zs2_te, zc1_te, Y_in=Y_test
    )

    print("Train cov share stats:", cov_stats_tr)
 #   print("Val   cov share stats:", cov_stats_val)
    print("Test  cov share stats:", cov_stats_te)
    print("Avg corr (train):\n", avg_corr_tr)


#Check for overfitting by correlating train and test variance explained
    # Note: correlation calculation is commented out, so we skip creating df_corr
    # corrs = []
    # for i in range(varexp_tr.shape[1]):  # each component
    #     corr = np.corrcoef(varexp_tr[:, i], varexp_val[:, i])[0, 1]
    #     corrs.append(corr)

    # df_corr = pd.DataFrame({
    #     'component': [f'comp{i+1}' for i in range(len(corrs))],
    #     'train_mean': varexp_tr.mean(0).numpy(),
    #   #  'val_mean': varexp_val.mean(0).numpy(),
    #     'corr_train_val': corrs
    # })
    # print(df_corr)

    avg_corr_te, avg_cov_te, cov_share_stats_te = compute_avg_covcorr_from_decoder(
        decoder,
        zs1_te,
        zs2_te,
        zc1_te,
        Y_test  # <-- important if you want covariance share relative to Var(Y)
    )

    print("\nTEST component correlation:\n", avg_corr_te)
    print("max |off-diag corr|:", float(np.max(np.abs(avg_corr_te[~np.eye(3, dtype=bool)]))))
    print("\nTEST component covariance:\n", avg_cov_te)
    print("max |off-diag cov|:", float(np.max(np.abs(avg_cov_te[~np.eye(3, dtype=bool)]))))
    print("\nTEST covariance share stats (2*sum cov / Var(y)):\n", cov_share_stats_te)





    # # Map components to desired labels
    # component_mapping = {'z1': 'Z1', 'zc1': 'Zc', 'zs2': 'Z2', 'noise': 'noise'}
    # final_table['component_label'] = final_table['component'].map(component_mapping)

    # # 1) Ground truth
    # gt_data = final_table[['outcome', 'component_label', 'gt_fraction']].copy()
    # gt_pivot = gt_data.pivot(index='outcome', columns='component_label', values='gt_fraction')

    # est_data  = final_table[['outcome', 'component_label', 'est_fraction_mean']].copy()
    # est_pivot = est_data.pivot(index='outcome', columns='component_label', values='est_fraction_mean')

    # # 2) Train and test estimates
    # train_data = final_table[['outcome', 'component_label', 'train']].copy()
    # test_data  = final_table[['outcome', 'component_label', 'test']].copy()

    # train_pivot = train_data.pivot(index='outcome', columns='component_label', values='train')
    # test_pivot  = test_data.pivot(index='outcome', columns='component_label', values='test')

    # # 3) Plot GT vs train vs test side by side
    # fig, axes = plt.subplots(2, 2, figsize=(18, 6), sharey=True)

    # # Unpack axes correctly
    # (ax0, ax1), (ax2, ax3) = axes

    # # Ground truth
    # gt_pivot.plot(kind='barh', stacked=True, ax=ax0, colormap='tab10')
    # ax0.set_title('Ground truth')
    # ax0.set_xlabel('Proportion')
    # ax0.legend(title='Components')
    # ax0.set_ylabel('Metabolites')

    # # Train
    # train_pivot.plot(kind='barh', stacked=True, ax=ax1, colormap='tab10')
    # ax2.set_title('Train measurements')
    # ax2.set_xlabel('Proportion')
    # ax2.legend(title='Components')
    # ax2.set_ylabel('Metabolites')

    # # Test
    # test_pivot.plot(kind='barh', stacked=True, ax=ax2, colormap='tab10')
    # ax3.set_title('Test measurements')
    # ax3.set_xlabel('Proportion')
    # ax3.legend(title='Components')
    # ax3.set_ylabel('Metabolites')

    # # Estimated
    # est_pivot.plot(kind='barh', stacked=True, ax=ax3, colormap='tab10')
    # ax1.set_title('Estimated measurements')
    # ax1.set_xlabel('Proportion')
    # ax1.legend(title='Components')
    # ax1.set_ylabel('Metabolites')

    # plt.tight_layout()
    # png_path = f"data/disentangled/variance_explained_{MODEL_NAME}_{run_id}_gt_train_test.png"
    # validate_unique_filename(png_path)
    # plt.savefig(png_path)
    # print(f"Saved: {png_path}")
    # plt.close()





#     df = pd.DataFrame(final_table)

#     #Auto-scale to percentages if values look like fractions
#     def to_percent(series):
#         m = series.max()
#         return series * 100 if m <= 1.5 else series

#     df_plot = df.copy()
#     df_plot["gt_pct"]  = to_percent(df_plot["gt_fraction"])
#     df_plot["est_pct"] = to_percent(df_plot["est_fraction_mean"])

#     components = sorted(df_plot["component"].unique())
#     n = len(components)

#     fig, axes = plt.subplots(1, n, figsize=(4*n, 4), squeeze=False)
#     axes = axes[0]

#     for ax, comp in zip(axes, components):
#         g = df_plot[df_plot["component"] == comp]

#         x = g["gt_pct"].to_numpy()
#         y = g["est_pct"].to_numpy()

#         # Metrics
#         rmse = np.sqrt(np.mean((y - x) ** 2))
#         r2   = r2_score(x, y) if len(g) > 1 else np.nan

#         # Scatter
#         ax.scatter(x, y, s=20)

#         # Identity line
#         lim_min = 0
#         lim_max = max(x.max(), y.max()) * 1.05
#         ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", linewidth=1)

#         ax.set_xlim(lim_min, lim_max)
#         ax.set_ylim(lim_min, lim_max)
#         ax.set_title(f"{comp}")
#         ax.set_xlabel("Ground truth (%)")
#         ax.set_ylabel("Estimate (%)")

#         # Annotation (top-left)
#         txt = f"$R^2$={r2:.3f}\nRMSE={rmse:.2f}"
#         ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top")

#     plt.tight_layout()
#     scatter_path = f'data/disentangled/scatterplot_explained_{MODEL_NAME}_{run_id}_train_test.png'
#     validate_unique_filename(scatter_path)
#     plt.savefig(scatter_path)
#     print(f"Saved: {scatter_path}")
#     plt.close()

#    # Permutation sanity check per component
#     print("Permutation sanity check on variance shares")
#     for comp in components:
#         g = df_plot[df_plot["component"] == comp]
#         x = g["gt_pct"].to_numpy()
#         y = g["est_pct"].to_numpy()

#         if len(g) <= 1:
#             print(f"{comp}: not enough points for R2")
#             continue

#         r2_real = r2_score(x, y)
#         y_perm = np.random.permutation(y)
#         r2_perm = r2_score(x, y_perm)

#         print(f"{comp}: R2 real = {r2_real:.3f}, R2 permuted = {r2_perm:.3f}")



    # ========================================================================
    # PERMUTATION TESTING (if requested)
    # ========================================================================
    if hasattr(args, 'run_permutation_test') and args.run_permutation_test:
        print(f"\n{'='*80}")
        print(f"VARIANCE FRACTION PERMUTATION TEST")
        print(f"{'='*80}")
        print(f"Number of permutations: {args.n_permutations}")
        print(f"Testing if variance fractions (Z1, Z2, Zs, noise) are significant...")
        print(f"Null hypothesis: Variance fractions are spurious (model memorizes noise)")
        print(f"Alternative: Variance fractions reflect real biological signal")
        print(f"{'='*80}\n")

        # Run permutations
        print(f"Running {args.n_permutations} permutations (this may take a while)...")
        perm_varexp_train = []
        perm_varexp_test = []
        perm_train_mse_list = []
        perm_test_mse_list  = []

        for perm_idx in range(args.n_permutations):
            if (perm_idx + 1) % 10 == 0:
                print(f"  Progress: {perm_idx + 1}/{args.n_permutations}")

            # permute TRAIN labels only
            perm_idx_train = torch.randperm(Y_train.shape[0], device=Y_train.device)
            Y_train_perm = Y_train[perm_idx_train]

            # latents fixed
            dataset_train_perm = TensorDataset(Y_train_perm, zs1, zs2, zc1)
            loader_train_perm = DataLoader(dataset_train_perm, shuffle=False, batch_size=config.batch_size)

            loader_test_real = data_loader_test

            # Train new decoder on permuted data (reuse same architecture)
            decoder_perm = Decoder_multiple_latents(
                data_dim, n_covariates, grid_z, grid_c,
                nn.Sequential(nn.Linear(z_dim_zs1, hidden_dim), nn.Tanh(), nn.Dropout(p=0.3),
                             nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Dropout(p=0.3),
                             nn.Linear(hidden_dim, data_dim)),
                [nn.Sequential(nn.Linear(zs2.shape[1], hidden_dim), nn.Tanh(), nn.Dropout(p=0.3),
                              nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Dropout(p=0.3),
                              nn.Linear(hidden_dim, data_dim)),
                 nn.Sequential(nn.Linear(zc1.shape[1], hidden_dim), nn.Tanh(), nn.Dropout(p=0.3),
                              nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Dropout(p=0.3),
                              nn.Linear(hidden_dim, data_dim))],
                [], has_feature_level_sparsity=False, p1=0.1, p2=0.1, p3=0.1,
                lambda0=1e3, penalty_type="MDMM", device=device
            )

            encoders_perm = nn.ModuleList([
                cEncoder(z_dim=z_dim_zs2, mapping=nn.Sequential()),
                cEncoder(z_dim=z_dim_zc1, mapping=nn.Sequential())
            ])

            model_perm = CVAE_multiple_latent_spaces_with_covariates(encoders_perm, decoder_perm, lr=0.0001, device=device)
            loss_perm, _, _ = model_perm.optimize(loader_train_perm, n_iter=config.iters, augmented_lagrangian_lr=0.01)

            # Evaluate on permuted data
            Y_error_tr_perm, _ = eval_split(decoder_perm, loader_train_perm, n_covariates=2)
            Y_error_te_perm, _ = eval_split(decoder_perm, loader_test_real,  n_covariates=2)
            perm_train_mse_list.append((Y_error_tr_perm ** 2).mean().item())
            perm_test_mse_list.append((Y_error_te_perm ** 2).mean().item())

            varexp_tr_perm = decoder_perm.fraction_of_variance_explained(
                zs1.to(device), [zs2.to(device), zc1.to(device)],
                Y_error=Y_error_tr_perm.to(device), divide_by_total_var=False
            ).cpu()

            varexp_te_perm = decoder_perm.fraction_of_variance_explained(
                zs1_test.to(device), [zs2_test.to(device), zc1_test.to(device)],
                Y_error=Y_error_te_perm.to(device), divide_by_total_var=False
            ).cpu()

            perm_varexp_train.append(varexp_tr_perm.numpy())
            perm_varexp_test.append(varexp_te_perm.numpy())

        perm_varexp_train = np.array(perm_varexp_train)
        perm_varexp_test  = np.array(perm_varexp_test)
        n_perm = perm_varexp_train.shape[0]

        print("\n✓ Permutations complete!")
        # ============================================================
        # GLOBAL (component-level) permutation test on TEST split
        # ============================================================
        varexp_real_test = varexp_te.numpy()          # [n_metabolites, n_components]
        perm_test = perm_varexp_test                 # [n_perm, n_metabolites, n_components]

        component_names = ['z1', 'zs2', 'zc1', 'noise']  # keep only if ordering is verified

        global_results = []
        for k, name in enumerate(component_names):
            # Real statistic: mean across metabolites (test split)
            T_real = float(varexp_real_test[:, k].mean())

            # Permutation null distribution: one mean per permutation
            T_perm = perm_test[:, :, k].mean(axis=1)  # shape [n_perm]

            # One-sided p-value (real larger than permuted)
            p_val = float((1.0 + np.sum(T_perm >= T_real)) / (1.0 + n_perm))

            global_results.append({
                "component": name,
                "T_real_test_mean": T_real,
                "T_perm_test_mean": float(T_perm.mean()),
                "T_perm_test_std": float(T_perm.std(ddof=0)),
                "p_value_test_global": p_val
            })

        df_global = pd.DataFrame(global_results)
        print("\n[GLOBAL TEST] Component-level permutation test (TEST split):")
        print(df_global)

        # Optional save next to your other outputs
        global_csv_path = os.path.join(out_dir, f"variance_fraction_global_permutation_{MODEL_NAME}_{run_id}.csv")
        df_global.to_csv(global_csv_path, index=False)
        print(f"✓ Global permutation results saved to: {global_csv_path}")


        # ========================================================================
        # VARIANCE FRACTION ANALYSIS
        # ========================================================================
        print(f"\n{'='*80}")
        print(f"VARIANCE FRACTION SIGNIFICANCE TEST")
        print(f"{'='*80}\n")
        print("n_perm =", n_perm)

        print("Real train MSE:", train_mse)
        print("Perm train MSE mean ± std:",
            float(np.mean(perm_train_mse_list)), "±", float(np.std(perm_train_mse_list)))

        print("Real test MSE:", test_mse)
        print("Perm test MSE mean ± std:",
            float(np.mean(perm_test_mse_list)), "±", float(np.std(perm_test_mse_list)))
        print("perm_varexp_test shape:", perm_varexp_test.shape)
        print("varexp_te col means:", varexp_te.mean(dim=0).cpu().numpy())
        print("varexp_te col mins :", varexp_te.min(dim=0).values.cpu().numpy())
        print("varexp_te col maxs :", varexp_te.max(dim=0).values.cpu().numpy())

        # also check last column explicitly (should be residual if you passed Y_error)
        print("residual column mean (assume last):", varexp_te[:, -1].mean().item())


        component_names = ['z1', 'zs2', 'zc1', 'noise']
        varexp_real_train = varexp_tr.numpy()
        varexp_real_test = varexp_te.numpy()
        n_metabolites = Y_train.shape[1]
        detailed_results = []
        for comp_idx, comp_name in enumerate(component_names):
            print(f"\n--- Component: {comp_name} ---")

            real_train_comp = varexp_real_train[:, comp_idx]
            real_test_comp  = varexp_real_test[:, comp_idx]
            perm_train_comp = perm_varexp_train[:, :, comp_idx]
            perm_test_comp  = perm_varexp_test[:, :, comp_idx]

            # Finite-sample corrected permutation p-values
            p_values_train = (1.0 + np.sum(perm_train_comp >= real_train_comp[np.newaxis, :], axis=0)) / (1.0 + n_perm)
            p_values_test  = (1.0 + np.sum(perm_test_comp  >= real_test_comp[np.newaxis, :],  axis=0)) / (1.0 + n_perm)

            # Effect sizes (z-score vs permutation null)
            effect_sizes_train = (real_train_comp - perm_train_comp.mean(axis=0)) / (perm_train_comp.std(axis=0) + 1e-10)
            effect_sizes_test  = (real_test_comp  - perm_test_comp.mean(axis=0))  / (perm_test_comp.std(axis=0)  + 1e-10)

            # BH-FDR correction within this component across metabolites
            _, q_values_train, _, _ = multipletests(p_values_train, alpha=0.05, method="fdr_bh")
            _, q_values_test,  _, _ = multipletests(p_values_test,  alpha=0.05, method="fdr_bh")

            for met_idx in range(n_metabolites):
                if metabolite_names is not None and met_idx < len(metabolite_names):
                    met_name = str(metabolite_names[met_idx])
                else:
                    met_name = f"metabolite_{met_idx+1}"
                detailed_results.append({
                    'metabolite': met_name,
                    'component': comp_name,
                    'real_train': real_train_comp[met_idx],
                    'real_test': real_test_comp[met_idx],
                    'perm_mean_train': perm_train_comp[:, met_idx].mean(),
                    'perm_mean_test': perm_test_comp[:, met_idx].mean(),
                    'perm_std_train': perm_train_comp[:, met_idx].std(),
                    'perm_std_test': perm_test_comp[:, met_idx].std(),
                    'p_value_train': float(p_values_train[met_idx]),
                    'p_value_test': float(p_values_test[met_idx]),
                    'q_value_train': float(q_values_train[met_idx]),
                    'q_value_test': float(q_values_test[met_idx]),
                    'effect_size_train': float(effect_sizes_train[met_idx]),
                    'effect_size_test': float(effect_sizes_test[met_idx]),
                    'significant_train_fdr05': bool(q_values_train[met_idx] < 0.05),
                    'significant_test_fdr05': bool(q_values_test[met_idx] < 0.05),
                })

            n_sig_train = (q_values_train < 0.05).sum()
            n_sig_test  = (q_values_test  < 0.05).sum()

            print(f"  Real variance (mean ± std): Train={real_train_comp.mean():.4f}±{real_train_comp.std():.4f}, Test={real_test_comp.mean():.4f}±{real_test_comp.std():.4f}")
            print(f"  Perm variance (mean ± std): Train={perm_train_comp.mean():.4f}±{perm_train_comp.std():.4f}, Test={perm_test_comp.mean():.4f}±{perm_test_comp.std():.4f}")
            print(f"  Significant metabolites (BH-FDR<0.05): Train={n_sig_train}/{n_metabolites} ({100*n_sig_train/n_metabolites:.1f}%), Test={n_sig_test}/{n_metabolites} ({100*n_sig_test/n_metabolites:.1f}%)")
            print(f"  Median p-value: Train={np.median(p_values_train):.4f}, Test={np.median(p_values_test):.4f}")
            print(f"  Median q-value: Train={np.median(q_values_train):.4f}, Test={np.median(q_values_test):.4f}")
            print(f"  Mean effect size: Train={effect_sizes_train.mean():.2f}, Test={effect_sizes_test.mean():.2f}")

        df_varfrac_results = pd.DataFrame(detailed_results)
        varfrac_csv_path = os.path.join(out_dir, f"variance_fraction_permutation_{MODEL_NAME}_{run_id}.csv")
        df_varfrac_results.to_csv(varfrac_csv_path, index=False)
        print(f"\n✓ Variance fraction results saved to: {varfrac_csv_path}")

        # Variance fraction heatmap
        fig_heat, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        pval_train_var = df_varfrac_results.pivot(index='metabolite', columns='component', values='q_value_train')[component_names]
        pval_test_var = df_varfrac_results.pivot(index='metabolite', columns='component', values='q_value_test')[component_names]

        im1 = ax1.imshow(pval_train_var.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
        ax1.set_xticks(range(len(component_names)))
        ax1.set_xticklabels(component_names)
        ax1.set_yticks(range(len(pval_train_var)))
        ax1.set_yticklabels(pval_train_var.index, fontsize=7)
        ax1.set_title('Variance Fraction p-values (Train)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Component')
        ax1.set_ylabel('Metabolite')
        plt.colorbar(im1, ax=ax1, label='p-value')

        for i in range(len(pval_train_var)):
            for j in range(len(component_names)):
                if pval_train_var.iloc[i, j] < 0.05:
                    ax1.text(j, i, '*', ha='center', va='center', color='black', fontsize=14, fontweight='bold')

        im2 = ax2.imshow(pval_test_var.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
        ax2.set_xticks(range(len(component_names)))
        ax2.set_xticklabels(component_names)
        ax2.set_yticks(range(len(pval_test_var)))
        ax2.set_yticklabels(pval_test_var.index, fontsize=7)
        ax2.set_title('Variance Fraction p-values (Test)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Component')
        ax2.set_ylabel('Metabolite')
        plt.colorbar(im2, ax=ax2, label='p-value')

        for i in range(len(pval_test_var)):
            for j in range(len(component_names)):
                if pval_test_var.iloc[i, j] < 0.05:
                    ax2.text(j, i, '*', ha='center', va='center', color='black', fontsize=14, fontweight='bold')

        plt.tight_layout()
        heatmap_path = os.path.join(out_dir, f"variance_fraction_pvalues_{MODEL_NAME}_{run_id}.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"✓ Variance fraction heatmap saved to: {heatmap_path}")
        plt.close()

        print(f"\n{'='*80}")
        print(f"PERMUTATION TEST COMPLETE")
        print(f"{'='*80}\n")

    wandb.finish()
if __name__ == "__main__":
    main()