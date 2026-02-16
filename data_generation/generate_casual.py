import numpy as np
rng = np.random.default_rng


def to_len_Y(x, dim_Y):
    """Allow scalar or 1D array; return (dim_Y,) float np.array"""
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return np.full(dim_Y, float(x))
    assert x.shape == (dim_Y,), f"Expected shape ({dim_Y},), got {x.shape}"
    return x


def outcome(Zs, Z1, Z2, C, W1_c, W2_c, Ws_c, Wc_c, dim_Y=5, seed=0):
    """
    Y[:,j] = Z1 @ W1[:,j] + Z2 @ W2[:,j] + Zs @ Ws[:,j] + C @ Wc[:,j] + eps[:,j]
    - W*_c can be scalar or length-dim_Y vector of per-outcome coefficients
    - Noise per outcome j is set to 1 - (W1_c[j]^2 + W2_c[j]^2 + Ws_c[j]^2 + Wc_c[j]^2)
    Returns:
      Y: (n, dim_Y)
      gt: dict with weights, per-source variance parts, totals, and fraction shares
    """
    n, dZs = Zs.shape
    dZ1, dZ2, dC = Z1.shape[1], Z2.shape[1], C.shape[1]
    # Normalize features to unit variance
    Zs = (Zs - Zs.mean(0)) / Zs.std(0)
    Z1 = (Z1 - Z1.mean(0)) / Z1.std(0)
    Z2 = (Z2 - Z2.mean(0)) / Z2.std(0)
    C  = (C  - C.mean(0))  / C.std(0)

    # Per-outcome coefficients
    W1_c = to_len_Y(W1_c, dim_Y)
    W2_c = to_len_Y(W2_c, dim_Y)
    Ws_c = to_len_Y(Ws_c, dim_Y)
    Wc_c = to_len_Y(Wc_c, dim_Y)

    r = rng(seed)
    # Weight matrices: each latent dim uses the same coeff per outcome (as in your original)
    W1 = np.tile(W1_c, (dZ1, 1))
    W2 = np.tile(W2_c, (dZ2, 1))
    Ws = np.tile(Ws_c, (dZs, 1))
    Wc = np.tile(Wc_c, (dC,  1))

    noise_var = 1.0 - (W1_c**2 + W2_c**2 + Ws_c**2 + Wc_c**2)  # (dim_Y,)
    assert np.all(noise_var > 0), "All outcomes must satisfy sum(W^2) < 1 for positive noise variance."
    eps = r.normal(0, np.sqrt(noise_var), size=(n, dim_Y))

    Y = Z1 @ W1 + Z2 @ W2 + Zs @ Ws + C @ Wc + eps

    # Theory variance pieces (given unit-variance latents and shared coeffs)
    V1 = np.sum(W1**2, axis=0)        # (dim_Y,)
    V2 = np.sum(W2**2, axis=0)
    Vs = np.sum(Ws**2, axis=0)
    Vc = np.sum(Wc**2, axis=0)
    Vn = noise_var.copy()
    Vtot = V1 + V2 + Vs + Vc + Vn

    shares = {
        "Z1": V1 / Vtot,
        "Z2": V2 / Vtot,
        "Zs": Vs / Vtot,
        "C" : Vc / Vtot,
        "noise": Vn / Vtot,
    }

    gt = dict(
        weights=dict(W1=W1_c, W2=W2_c, Ws=Ws_c, Wc=Wc_c),
        var_parts=dict(Z1=V1, Z2=V2, Zs=Vs, C=Vc, noise=Vn, total=Vtot),
        shares=shares,
        noise_var=noise_var,
    )
    return Y, gt


def independent_latents(n=1000, dim_Zs=10, dim_Z1=10, dim_Z2=10, seed=0):
    r = rng(seed)
    Zs = r.normal(0, 1.0, size=(n, dim_Zs))
    Z1 = r.normal(0, 1.0, size=(n, dim_Z1))
    Z2 = r.normal(0, 1.0, size=(n, dim_Z2))
    return Zs, Z1, Z2

def confounders(n=1000, dim_c=1, seed=0):
    r = rng(10_000 + seed)
    return r.normal(0, 1.0, size=(n, dim_c))


def confound_latent_linear(Z, C, strength=0.6, seed=0):
    r = rng(20_000 + seed)
    dZ, dC = Z.shape[1], C.shape[1]
    G = r.normal(0, strength/np.sqrt(dC), size=(dC, dZ))
    U = r.normal(0, 0.1, size=Z.shape)
    return Z + C @ G + U
    # G = r.uniform(low=-1, high=1, size=(dC, dZ))
    # return Z + C @ G


def confound_latent_mlp(Z, C, strength=0.6, seed=0):
    r = rng(21_000 + seed)
    n, dZ = Z.shape
    dC = C.shape[1]
    h_dim = max(8, (dZ + dC)//2)
    W1 = r.normal(0, 1.0/np.sqrt(dC), size=(dC, h_dim))
    b1 = r.normal(0, 0.1, size=(h_dim,))
    W2 = r.normal(0, 1.0/np.sqrt(h_dim), size=(h_dim, dZ))
    b2 = r.normal(0, 0.1, size=(dZ,))
    h = np.tanh(C @ W1 + b1)
    mlp_out = h @ W2 + b2
    U = r.normal(0, 0.1, size=Z.shape)
    return Z + mlp_out + U


def make_X_linear(Zs_tilde, Zx_tilde, dim_X=20, noise_var=0.5, seed=0):
    r = rng(30_000 + seed)
    n = Zs_tilde.shape[0]
    Zcat = np.concatenate([Zx_tilde, Zs_tilde], axis=1)
    d = Zcat.shape[1]
    T = r.uniform(low=-1, high=1, size=(d, dim_X))
    return Zcat @ T


def make_X_mlp(Zs_tilde, Zx_tilde, dim_X=20, noise_var=0.5, seed=0):
    r = rng(31_000 + seed)
    n = Zs_tilde.shape[0]
    Zcat = np.concatenate([Zx_tilde, Zs_tilde], axis=1)
    d = Zcat.shape[1]
    T = r.uniform(low=-1, high=1, size=(d, dim_X))
    mlp_out = np.tanh(Zcat @ T)
    T2 = r.uniform(low=-1, high=1, size=(dim_X, dim_X))
    return mlp_out @ T2


def make_weight_grid(
    values=(0.0, 0.1, 0.2, 0.3, 0.4),
    max_sum_sq=0.95,
    target_n=100,
    seed=0,
):
    """
    Build a list of tuples (W1, W2, Ws, Wc) with sum of squares < 1.
    Returns up to target_n tuples, shuffled reproducibly.
    """
    vals = np.asarray(values, dtype=float)
    combos = []
    for w1 in vals:
        for w2 in vals:
            for ws in vals:
                for wc in vals:
                    if (w1*w1 + w2*w2 + ws*ws + wc*wc) < max_sum_sq and (w1, w2, ws, wc) != (0,0,0,0):
                        combos.append((w1, w2, ws, wc))
    r = rng(seed)
    r.shuffle(combos)
    if target_n is not None:
        combos = combos[:target_n]
    return combos


def make_model(W1_c, W2_c, Ws_c, Wc_c, n=1000, dim_Zs=10, dim_Z1=10, dim_Z2=10,
               dim_X=20, dim_Y=20, dim_C=1,
               confound_latent_fn=confound_latent_linear,
               make_X_fn=make_X_linear,
               outcome_fn=outcome,
               seed=0):
    Zs, Z1, Z2 = independent_latents(n, dim_Zs, dim_Z1, dim_Z2, seed=seed)
    C = confounders(n, dim_C, seed=seed)

    Z1_tilde = confound_latent_fn(Z1, C, strength=0.6, seed=seed)
    Z2_tilde = confound_latent_fn(Z2, C, strength=0.6, seed=seed)
    Zs_tilde = confound_latent_fn(Zs, C, strength=0.6, seed=seed)

    X1 = make_X_fn(Zs_tilde, Z1_tilde, dim_X=dim_X, seed=seed)
    X2 = make_X_fn(Zs_tilde, Z2_tilde, dim_X=dim_X, seed=seed)

    Y, gt = outcome_fn(Zs, Z1, Z2, C, W1_c, W2_c, Ws_c, Wc_c, dim_Y=dim_Y, seed=seed)
    return X1, X2, Y, C, gt


def make_model_non_linear(W1_c, W2_c, Ws_c, Wc_c, n=1000, dim_Zs=10, dim_Z1=10, dim_Z2=10,
               dim_X=20, dim_Y=20, dim_C=1,
               seed=0):
    return make_model(W1_c, W2_c, Ws_c, Wc_c, n=n, dim_Zs=dim_Zs, dim_Z1=dim_Z1, dim_Z2=dim_Z2,
               dim_X=dim_X, dim_Y=dim_Y, dim_C=dim_C,
               confound_latent_fn=confound_latent_mlp,
               make_X_fn=make_X_mlp,
               outcome_fn=outcome,
               seed=seed)
