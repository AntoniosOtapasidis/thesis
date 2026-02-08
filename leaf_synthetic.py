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

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def step1_filename(args):
    return f'cib_seed{args.seed}_{args.beta}_kappa{int(args.kappa)}_epoch{args.num_epoch_s1}_dim{args.embed_dim}.tar'

def train_mp(beta, train_loader, test_loader, train_dataset, test_dataset, args):
    step1_name = step1_filename(args)
    out_dir = f'./results_synthetic_task/{args.data_mode}/models/'
    os.makedirs(out_dir, exist_ok=True)
    step1_path = os.path.join(out_dir, step1_name)
    print(f'Training step 1: CIB model for beta:', beta, 'seed:', args.seed)
    cmib = MVInfoMaxModel(
        args.dim_info['X'], args.dim_info['Y'],
        args.hidden_dim, args.embed_dim,
        initialization='normal', distribution='vmf', vmfkappa=args.kappa,
        layers=args.layers, beta_start_value=beta, beta_end_value=beta,
        beta_n_iterations=8000, beta_start_iteration=0,
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
    step1_name = step1_filename(args)
    step1_path = os.path.join(f'./results_synthetic_task/{args.data_mode}/models/', step1_name)
    cmib_step1 = MVInfoMaxModel(
        args.dim_info['X'], args.dim_info['Y'],
        args.hidden_dim, args.embed_dim,
        initialization='normal', distribution='vmf', vmfkappa=args.kappa,
        layers=args.layers, beta_start_value=args.beta, beta_end_value=args.beta,
        beta_n_iterations=8000, beta_start_iteration=0
    )
    if args.device != "cpu":
        cmib_step1 = cmib_step1.cuda()
    cmib_step1.load_state_dict(torch.load(step1_path))
    print('Training step 2: Disen model')
    disen = DisenModel(
        cmib_step1, args.dim_info['X'], args.dim_info['Y'],
        args.hidden_dim, args.embed_dim, zs_dim=args.embed_dim,
        initialization='normal',
        layers=args.layers, lmd_start_value=args.lmd_start, lmd_end_value=args.lmd_end,
        lmd_n_iterations=8000, lmd_start_iteration=0,
        ortho_norm=args.ortho_norm, condzs=args.condzs, proj=args.proj,
        usezsx=args.usezsx, apdzs=args.apdzs, hsic_weight=args.hsic_weight
    )
    if args.device != "cpu":
        disen = disen.cuda()
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

def residualize_data(c, *data_tensors, config=None, device="cuda"):
    Y_combined = torch.cat(data_tensors, dim=1)
    N = c.shape[0]
    z1 = Uniform(-2.0, 2.0).sample((N, 1))
    z = z1
    dataset = TensorDataset(Y_combined.to(device), z.to(device), c.to(device))
    data_loader = DataLoader(dataset, shuffle=False, batch_size=config.batch_size)
    z_dim = z.shape[1]
    lim_val = 2.0
    num_samples = 10000
    data_dim = Y_combined.shape[1]
    hidden_dim = 32
    grid_z = (torch.rand(num_samples, z_dim, device=device) * 2 * lim_val) - lim_val
    # grid_cov = lambda x: (torch.rand(num_samples, x.shape[1], device=device) * 2 * lim_val) - lim_val
    # grid_c = [grid_cov(x) for x in [c]]
    grid_cov = lambda x: (torch.rand(num_samples, x.shape[1], device=device) * 2 * lim_val) - lim_val
    grid_c = [grid_cov(x) for x in [c]]
    print("RESIDUALIZE: len(grid_c)=", len(grid_c), "c.shape=", tuple(c.shape))

    encoder_mapping = nn.Sequential()

    decoder_z = nn.Sequential(
        nn.Linear(z_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, data_dim)
    )
    encoder = cEncoder(z_dim=z_dim, mapping=encoder_mapping)
    decoders_c = [nn.Sequential(
        nn.Linear(x.shape[1], hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, data_dim)
    ) for x in [c]]
    decoders_cz = []
    decoder = Decoder_multiple_latents(
        data_dim, 1,
        grid_z, grid_c,
        decoder_z, decoders_c, decoders_cz,
        has_feature_level_sparsity=False, p1=0.1, p2=0.1, p3=0.1,
        lambda0=1e3, penalty_type="MDMM", device=device
    )
    model = CVAE_multiple_latent_spaces_with_covariates(encoder, decoder, lr=1e-4, device=device)
    loss, integrals, _, _ = model.optimize(data_loader, n_iter=config.iters_res, augmented_lagrangian_lr=0.1)
    with torch.no_grad():
        Y_error_list, Y_pred_list = [], []
        for batch in data_loader:
            Y_batch, z_batch, c_batch = batch
            confounders = [c_batch.to(device)]
            Y_pred = decoder.forward_c(confounders)[0]
            Y_error = Y_batch - Y_pred
            Y_error_list.append(Y_error)
            Y_pred_list.append(Y_pred)
        Y_error = torch.cat(Y_error_list, dim=0)
    dims = [d.shape[1] for d in data_tensors]
    Y_parts = torch.split(Y_error, dims, dim=1)
    return Y_parts

def build_variance_table(varexp, Y_torch, gt, split):
    print("INSIDE build_variance_table")
    print("gt id inside:", id(gt))
    print("gt shares keys inside:", list(gt["shares"].keys()))
    print("gt_map expects:", {"c": "C"})

    comp_keys = ["z1", "zs2", "zc1", "C"]
    rows = []
    varexp_np = varexp.numpy()
    total_var = Y_torch.var(dim=0)
    gt_map = {"z1": "Z1", "zs2": "Z2", "zc1": "Zs", "c": "C", "C": "C"}
    for i in range(Y_torch.shape[1]):
        outcome = f"outcome{i+1}"
        outcome_var = float(total_var[i].item())
        est_abs = varexp_np[i][:len(comp_keys)]
        est_sum = float(np.sum(est_abs))
        est_noise_abs = max(outcome_var - est_sum, 0.0)
        est_fracs = [float(v)/outcome_var if outcome_var > 0 else 0.0 for v in est_abs]
        est_noise_frac = est_noise_abs/outcome_var if outcome_var > 0 else 0.0
        for key, est_f in zip(comp_keys, est_fracs):
            rows.append({
                "outcome": outcome,
                "component": key,
                "gt_fraction": float(gt["shares"][gt_map[key]][i]),
                "split": split,
                "est_fraction": est_f,
            })
        rows.append({
            "outcome": outcome,
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
    if c_var is None:
        c_var = 0.25  # default for Bernoulli(0.5)

    shares = {
        "Z1":    np.zeros(dim_y, dtype=np.float32),
        "Z2":    np.zeros(dim_y, dtype=np.float32),
        "Zs":    np.zeros(dim_y, dtype=np.float32),  # unused in this GT
        "C":     np.zeros(dim_y, dtype=np.float32),
        "noise": np.zeros(dim_y, dtype=np.float32),
    }

    # Digits (deterministic assignments)
    shares["Z1"][0] = 1.0  # digit of image 1
    shares["Z2"][1] = 1.0  # digit of image 2

    def set_pve(idx, w1_, ws_, wc_, sig_):
        """Fill shares for column idx using weights and component variances."""
        num_Z1 = (w1_**2) * z1p_var
        num_Z2 = (ws_**2) * s_var
        num_C  = (wc_**2) * c_var
        num_eps = (sig_**2)
        denom = num_Z1 + num_Z2 + num_C + num_eps
        # Guard against zero denom (shouldn't happen with noise present)
        if denom <= 0:
            return
        shares["Z1"][idx]    = num_Z1 / denom
        shares["Zs"][idx]    = num_Z2 / denom
        shares["C"][idx]     = num_C  / denom
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


def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['synthetic_linear',"microbiome_synthetic", 'synthetic_nonlinear', 'mnist'], default='synthetic')
    parser.add_argument('--data_mode', type=str, default='entangle')
    parser.add_argument('--num_data', type=int, default=10000)
    parser.add_argument('--dim_info', type=dict, default={'Y': 100, 'Z1': 50, 'Zs': 50, 'X': 100, 'Z2': 50})
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iters_pred', type=int, default=10000)
    parser.add_argument('--iters_res', type=int, default=5000)
    parser.add_argument('--hsic_weight', type=float, default=1)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--kappa', type=float, default=50)
    parser.add_argument('--ortho_norm', type=str2bool, default=True)
    parser.add_argument('--condzs', type=str2bool, default=True)
    parser.add_argument('--proj', type=str2bool, default=False)
    parser.add_argument('--apdzs', type=str2bool, default=True)
    parser.add_argument('--usezsx', type=str2bool, default=False)
    parser.add_argument('--simclr', type=str2bool, default=False)
    parser.add_argument('--head', type=str, default='none')
    parser.add_argument('--lmd_start', type=float, default=0.5)
    parser.add_argument('--lmd_end', type=float, default=1)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epoch_s1', type=int, default=30)
    parser.add_argument('--num_epoch_s2', type=int, default=30)
    parser.add_argument('--lr_s1', type=float, default=1e-4)
    parser.add_argument('--lr_s2', type=float, default=1e-4)
    parser.add_argument('--noise_scale', type=float, default=0.01)
    parser.add_argument('--drop_scale', type=float, default=2)
    parser.add_argument('--debug_mode', type=str2bool, default=False)
    parser.add_argument('--device', type=str, default="0")
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
        "iters_res": args.iters_res,
    })
    config = wandb.config

    if args.dataset in ["synthetic_linear", "synthetic_nonlinear"]:
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
        Wc_vec = np.array([w[3] for w in grid], dtype=float)
        MODEL_NAME = f'{args.dataset}_{args.num_epoch_s1}_{args.dim_info["Zs"]}_{args.dim_info["Z1"]}_{args.dim_info["Z2"]}_{args.dim_info["X"]}_{dim_Y}_3_{args.num_data}_seed{args.seed}_beta{args.beta}_hsic{args.hsic_weight}'
        if args.dataset == "synthetic_linear":
            X1, X2, Y, C, gt = make_model(
                W1_vec, W2_vec, Ws_vec, Wc_vec,
                n=args.num_data,
                dim_Zs=args.dim_info["Zs"], dim_Z1=args.dim_info["Z1"], dim_Z2=args.dim_info["Z2"],
                dim_X=args.dim_info["X"], dim_Y=dim_Y, dim_C=3,
                seed=args.seed,
            )
        else:
            X1, X2, Y, C, gt = make_model_non_linear(
                W1_vec, W2_vec, Ws_vec, Wc_vec,
                n=args.num_data,
                dim_Zs=args.dim_info["Zs"], dim_Z1=args.dim_info["Z1"], dim_Z2=args.dim_info["Z2"],
                dim_X=args.dim_info["X"], dim_Y=dim_Y, dim_C=3,
                seed=args.seed,
            )
        data = np.array([X1, X2])
        targets1 = Y[:, 0]
        targets2 = Y[:, 1]
        targets3 = Y[:, 2]
        wandb.config.update({
            "seed": args.seed,
            "model_name": MODEL_NAME,
            "dims": {"Zs": args.dim_info["Zs"], "Z1": args.dim_info["Z1"], "Z2": args.dim_info["Z2"], "X": args.dim_info["X"], "Y": dim_Y, "C": 3},
            "gt_weights_table_logged": True,
        }, allow_val_change=True)

    elif args.dataset == "microbiome_synthetic":
        _conf_dir = os.path.join(os.path.dirname(__file__), "data", "synthetic_confounders")
        df_X1 = pd.read_csv(
            os.path.join(_conf_dir, "X1_bacteria_synthetic_CLR_confounders.csv")
        )
        df_X2 = pd.read_csv(
            os.path.join(_conf_dir, "X2_viruses_synthetic_CLR_confounders.csv")
        )
        df_Y = pd.read_csv(
            os.path.join(_conf_dir, "Y_metabolites_log_synthetic_complex_sparse_confounders.csv")
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
        # 2b) load confounders (LEAF-style: C is a dim_C vector per sample)
        df_C = pd.read_csv(
            os.path.join(_conf_dir, "confounders_vector_complex_sparse.csv")
        ).set_index("sim")

        df_C = df_C.loc[common_idx]

        # Extract confounder columns (C_1, C_2, ..., C_dim_C)
        c_cols = [col for col in df_C.columns if col.startswith("C_")]
        dim_C = len(c_cols)
        print(f"Found {dim_C} confounder dimensions: {c_cols}")

        assert df_C.index.equals(common_idx), "df_C index not aligned to common_idx"

        # C is the full confounder matrix (N, dim_C)
        C = df_C[c_cols].to_numpy(dtype=np.float32)  # (N, dim_C)
        c = C  # alias for residualization (same as C now)

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
            os.path.join(_conf_dir, "GT_virome_variance_shares_complex_sparse_confounders.csv")
        )

        # ensure metabolite ordering in GT matches Y columns
        # df_gt$met is like "met_1", "met_2", ...
        # df_Y_feats.columns should be same strings
        assert list(df_gt["met"]) == list(df_Y_feats.columns), \
            "GT metabolite order does not match Y columns"
        assert C.shape[0] == len(common_idx), "Confounder length mismatch vs common_idx"

        Z1_share    = df_gt["share_Z1"].to_numpy(dtype=np.float32)
        Z2_share    = df_gt["share_Z2"].to_numpy(dtype=np.float32)
        Zs_share    = df_gt["share_Zs"].to_numpy(dtype=np.float32)
        noise_share = df_gt["share_noise"].to_numpy(dtype=np.float32)

        # Add confounder share
        if "share_C" in df_gt.columns:
            C_share = df_gt["share_C"].to_numpy(dtype=np.float32)
        elif "share_c" in df_gt.columns:
            C_share = df_gt["share_c"].to_numpy(dtype=np.float32)
        else:
            # If GT file doesn't explicitly store confounder share, infer it as leftover
            C_share = 1.0 - (Z1_share + Z2_share + Zs_share + noise_share)
            C_share = np.clip(C_share, 0.0, 1.0).astype(np.float32)

        shares = {
            "Z1":    Z1_share,
            "Z2":    Z2_share,
            "Zs":    Zs_share,
            "C":     C_share,          # <-- THIS fixes the KeyError
            "noise": noise_share,
        }

        gt = {"shares": shares}

        print("GT shares keys:", list(gt["shares"].keys()))

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
                    "C":  dim_C,  # LEAF-style: multi-dimensional confounder
                },
                "gt_weights_table_logged": True,
            },
            allow_val_change=True,
        )

    else:
        # MNIST data generation
        args.dim_info = {'Y': 784, 'Z1': 50, 'Zs': 50, 'X': 784, 'Z2': 50}
        MODEL_NAME = (
            f"mnist_pairs_sharedConf_same{args.prob_same:.2f}"
            f"_alphaS{args.alpha_s}_alphaC{args.alpha_c}_p{args.p_high}-{args.p_low}"
            f"_N{args.num_data}_seed{args.seed}_beta{args.beta}_hsic{args.hsic_weight}"
        )
        data, targets1, targets2, targets3, C, Y, S_shared = build_mnist_pairs_shared_conf(
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
            "dims": {"X1": 784, "X2": 784, "Y_out": int(Y.shape[1]), "C": 1},
            "dataset": "MNIST paired (shared style + confounder)",
        }, allow_val_change=True)

    # Save confounders and targets
    os.makedirs('data/disentangled', exist_ok=True)
    np.save(f'data/disentangled/confounder_{MODEL_NAME}.npy', C)
    np.save(f'data/disentangled/all_targets_{MODEL_NAME}.npy', Y_np)

    # Dataset + split
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

        # --- Confounder split (must align with the SAME indices/test_idx) ---
        confounder = C  # shape [num_samples, ...] (must match num_data on axis 0)

        train_confounder = confounder[indices]
        test_confounder  = confounder[test_idx]

        np.save(f"data/disentangled/train_confounder_{MODEL_NAME}.npy", train_confounder)
        np.save(f"data/disentangled/test_confounder_{MODEL_NAME}.npy",  test_confounder)

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

    # Loaders
    train_loader = DataLoader(train_dataset, shuffle=False, drop_last=False, batch_size=args.batch_size)
    test_loader  = DataLoader(test_dataset,  shuffle=False, drop_last=False, batch_size=args.batch_size)

    # Train models
    train_mp(args.beta, train_loader, test_loader, train_dataset, test_dataset, args)
    logs, disen = train_step2(args, train_loader, test_loader, train_dataset, test_dataset)

    # Gather embeddings
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

    # Prepare tensors for variance explained model
    Y_all = torch.from_numpy(np.load(f'data/disentangled/all_targets_{MODEL_NAME}.npy')).float().to(device)
    Y_train = Y_all[indices]
    Y_test  = Y_all[test_idx]
    c_train = torch.from_numpy(train_confounder).float().to(device)
    c_test  = torch.from_numpy(test_confounder).float().to(device)
    zs1 = torch.from_numpy(np.load(f'data/disentangled/zs1_{MODEL_NAME}.npy')).float().to(device)
    zs2 = torch.from_numpy(np.load(f'data/disentangled/zs2_{MODEL_NAME}.npy')).float().to(device)
    zc1 = torch.from_numpy(np.load(f'data/disentangled/zc1_{MODEL_NAME}.npy')).float().to(device)
    zc2 = torch.from_numpy(np.load(f'data/disentangled/zc2_{MODEL_NAME}.npy')).float().to(device)
    zs1_test = torch.from_numpy(np.load(f'data/disentangled/zs1_test_{MODEL_NAME}.npy')).float().to(device)
    zs2_test = torch.from_numpy(np.load(f'data/disentangled/zs2_test_{MODEL_NAME}.npy')).float().to(device)
    zc1_test = torch.from_numpy(np.load(f'data/disentangled/zc1_test_{MODEL_NAME}.npy')).float().to(device)
    zc2_test = torch.from_numpy(np.load(f'data/disentangled/zc2_test_{MODEL_NAME}.npy')).float().to(device)

    # Residualize confounder latents (optional; here we DO residualize zc1/zc2; keep z1/zs2 as-is)
    zc1_res_train, zc2_res_train = residualize_data(c_train, *[zc1, zc2], config=config, device=device)
    zs1_res_train, zs2_res_train = zs1, zs2
    zc1_res_test, zc2_res_test = residualize_data(c_test, *[zc1_test, zc2_test], config=config, device=device)
    zs1_res_test, zs2_res_test = zs1_test, zs2_test

    # Build loaders for the variance explained CVAE
    dataset_train = TensorDataset(
        Y_train,
        zs1_res_train.to(device),
        zs2_res_train.to(device),
        zc1_res_train.to(device),
        c_train.to(device),
    )
    data_loader_train = DataLoader(dataset_train, shuffle=False, batch_size=config.batch_size)
    dataset_test = TensorDataset(
        Y_test,
        zs1_res_test.to(device),
        zs2_res_test.to(device),
        zc1_res_test.to(device),
        c_test.to(device),
    )
    data_loader_test = DataLoader(dataset_test, shuffle=False, batch_size=config.batch_size)

    # Fit variance explained model
    data_dim = Y_train.shape[1]
    hidden_dim = 100
    z_dim = zs1.shape[1]
    n_covariates = 3
    lim_val = 2.0
    num_samples = 1000




    K = 1000  # Much larger grid for better marginalization

    def rand_grid_prior(dim, K):
        # Sample from standard normal prior (no training data dependency)
        return torch.randn(K, dim, device=device).detach()

    grid_z = rand_grid_prior(zs1_tr.shape[1], K)
    grid_c = [
        rand_grid_prior(zs2_tr.shape[1], K),
        rand_grid_prior(zc1_tr.shape[1], K),
        rand_grid_prior(c_train.shape[1], K),
    ]
    assert Y_train.shape[0] == zs1_tr.shape[0] == zs2_tr.shape[0] == zc1_tr.shape[0]



    encoder_mapping = nn.Sequential()
    decoder_z = nn.Sequential(
        nn.Linear(z_dim, hidden_dim),
        nn.Tanh(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_dim, data_dim)
    )
    encoder = cEncoder(z_dim=z_dim, mapping=encoder_mapping)
    decoders_c = [nn.Sequential(
        nn.Linear(x.shape[1], hidden_dim),
        nn.Tanh(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_dim, data_dim)
    ) for x in (zs2, zc1, c_train)]
    decoders_cz = []
    decoder = Decoder_multiple_latents(
        data_dim, n_covariates,
        grid_z, grid_c,
        decoder_z, decoders_c, decoders_cz,
        has_feature_level_sparsity=False, p1=0.1, p2=0.1, p3=0.1,
        lambda0=1e3, penalty_type="MDMM",
        device=device
    )
    model = CVAE_multiple_latent_spaces_with_covariates(encoder, decoder, lr=1e-4, device=device)
    loss, integrals, _, _ = model.optimize(data_loader_train, n_iter=config.iters, augmented_lagrangian_lr=0.1)

    def eval_split(decoder, data_loader, n_covariates=3):
        with torch.no_grad():
            Y_err_list, Y_pred_list = [], []
            for batch in data_loader:
                Y_b, zs1_b, zs2_b, zc1_b, c_b = batch
                confs = [zs2_b, zc1_b, c_b]
                Y_pred = decoder(zs1_b, confs)
                Y_err  = Y_b - Y_pred
                Y_err_list.append(Y_err)
                Y_pred_list.append(Y_pred)
            Y_err  = torch.cat(Y_err_list,  dim=0)
            Y_pred = torch.cat(Y_pred_list, dim=0)
        return Y_err, Y_pred

    Y_error_tr, Y_pred_tr = eval_split(decoder, data_loader_train, n_covariates=3)
    confounders_tr = [zs2_res_train.to(device), zc1_res_train.to(device), c_train.to(device)]
    varexp_tr = decoder.fraction_of_variance_explained(
        zs1_res_train.to(device), confounders_tr, Y_error=Y_error_tr.to(device), divide_by_total_var=False
    ).cpu()
    Y_error_te, Y_pred_te = eval_split(decoder, data_loader_test, n_covariates=3)
    confounders_te = [zs2_res_test.to(device), zc1_res_test.to(device), c_test.to(device)]
    varexp_te = decoder.fraction_of_variance_explained(
        zs1_res_test.to(device), confounders_te, Y_error=Y_error_te.to(device), divide_by_total_var=False
    ).cpu()

    out_dir = os.path.join(os.path.dirname(__file__), "results_synthetic_task")

        # ---- SAVE PREDICTIONS FOR PLOTTING + COMPUTE TEST R2 ----
    Y_test_np = Y_test.detach().cpu().numpy()
    Y_pred_te_np = Y_pred_te.detach().cpu().numpy()

    # SSE-based R2 (same definition as sklearn)
    r2_test_per_met = r2_score(Y_test_np, Y_pred_te_np, multioutput="raw_values")
    r2_test_mean = float(r2_score(Y_test_np, Y_pred_te_np, multioutput="uniform_average"))

    print("LEAF SSE-R2 test mean:", r2_test_mean)
    print("LEAF SSE-R2 test median per-met:", float(np.median(r2_test_per_met)))

    # variance-based R2 (matches your variance decomposition idea)
    Vy = np.var(Y_test_np, axis=0, ddof=0)
    Vres = np.var(Y_test_np - Y_pred_te_np, axis=0, ddof=0)
    eps = 1e-12
    Vy_safe = np.where(Vy <= eps, eps, Vy)
    r2_var_per_met = 1.0 - (Vres / Vy_safe)
    print("LEAF VAR-R2 test mean:", float(np.mean(r2_var_per_met)))
    pred_dir = os.path.join(out_dir, "predictions")

    # Save for later parity/residual plots
    os.makedirs(pred_dir, exist_ok=True)
    np.save(os.path.join(pred_dir, f"Y_test_{MODEL_NAME}_seed{args.seed}.npy"), Y_test_np)
    np.save(os.path.join(pred_dir, f"Y_hat_test_{MODEL_NAME}_seed{args.seed}.npy"), Y_pred_te_np)

    np.save(os.path.join(pred_dir, f"R2_sse_test_per_met_{MODEL_NAME}_seed{args.seed}.npy"), r2_test_per_met)
    np.save(os.path.join(pred_dir, f"R2_var_test_per_met_{MODEL_NAME}_seed{args.seed}.npy"), r2_var_per_met)



    print("ABOUT TO CALL build_variance_table")
    print("gt id:", id(gt))
    print("gt shares keys right before call:", list(gt["shares"].keys()))
    # Build variance tables
    df_train_long = build_variance_table(varexp_tr, Y_train, gt, split="train")
    df_test_long  = build_variance_table(varexp_te, Y_test,  gt, split="test")


    df_both = pd.concat([df_train_long, df_test_long], ignore_index=True)
    df_piv  = df_both.pivot_table(
        index=["outcome","component"],
        columns="split",
        values="est_fraction",
        aggfunc="mean",
    ).reset_index()
    if "train" in df_piv and "test" in df_piv:
        df_piv["est_fraction_mean"] = df_piv[["train","test"]].mean(axis=1)
    else:
        df_piv["est_fraction_mean"] = df_piv.get("train", df_piv.get("test"))
    gt_unique = df_train_long[["outcome","component","gt_fraction"]].drop_duplicates()
    final_table = df_piv.merge(gt_unique, on=["outcome","component"], how="left")
    cols = ["outcome","component","gt_fraction","train","test","est_fraction_mean"]
    final_table = final_table.reindex(columns=cols)
    out_path = os.path.join(out_dir, f"variance_comparison_{MODEL_NAME}_train_test.csv")
    os.makedirs("data/disentangled", exist_ok=True)
    final_table.to_csv(out_path, index=False)
    print("Wrote:", out_path)
    print(final_table.head(12))
    wandb.finish()

if __name__ == "__main__":
    main()