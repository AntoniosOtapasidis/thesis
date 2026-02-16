
import numpy as np
from torchvision import datasets, transforms


def apply_shared_style(img_flat, s_scalar, alpha_s=0.75):
    # brightness scaling: img * (1 + alpha_s * s)
    return np.clip(img_flat * (1.0 + alpha_s * s_scalar), 0.0, 1.0)


def apply_confounder_mark(img_flat, c_scalar, alpha_c=0.4, patch=4):
    if c_scalar > 0.5:
        img = img_flat.reshape(28, 28).copy()
        img[:patch, :patch] = np.clip(img[:patch, :patch] + alpha_c, 0.0, 1.0)
        return img.reshape(-1)
    else:
        return img_flat


def build_mnist_pairs_shared_conf(seed, max_N, prob_same, alpha_s, alpha_c, patch,
                                  p_high, p_low, S_high_digits,
                                  add_style_target, w1, ws, wc, sigma_y):
    """
    - Pairs MNIST digits with ~prob_same same-digit pairs
    - Adds a shared brightness style s ~ N(0,1) to both images
    - Adds a confounder c ~ Bernoulli(p_c) independent of labels
    - Injects a shared latent driven 50% by c and 50% by noise into BOTH images
    - Returns two scalar digit targets (d1, d2) + optional style + confounder-only outcome

    If add_style_target:
      Y columns are:
        [0] y_style_base  (original ws, wc)
        [1] y_style_01_09 (signal split s:c = 0.1:0.9, same total signal power as base)
        [2] y_style_03_07 (0.3:0.7)
        [3] y_style_06_04 (0.6:0.4)
        [4] y_confonly
      Else:
        Y columns are:
        [0] y_confonly
    """

    rng = np.random.default_rng(seed)

    tf = transforms.Compose([transforms.ToTensor()])
    mnist_tr = datasets.MNIST(root="./data", train=True, download=True, transform=tf)
    mnist_te = datasets.MNIST(root="./data", train=False, download=True, transform=tf)

    # images [0,1], flatten
    imgs = np.concatenate([mnist_tr.data.numpy(), mnist_te.data.numpy()], axis=0).astype(np.float32) / 255.0
    imgs = imgs.reshape(len(imgs), -1)  # (70000, 784)
    labs = np.concatenate([mnist_tr.targets.numpy(), mnist_te.targets.numpy()], axis=0).astype(int)

    # Digit pools
    pools = {d: np.where(labs == d)[0] for d in range(10)}

    N = min(max_N, len(imgs))  # we'll sample with replacement anyway

    # sample digits
    d1 = rng.integers(0, 10, size=N)
    same_mask = rng.random(N) < prob_same
    d2 = d1.copy()
    diff_idx = np.where(~same_mask)[0]
    if len(diff_idx) > 0:
        d2[diff_idx] = rng.integers(0, 10, size=len(diff_idx))
        clash = diff_idx[d2[diff_idx] == d1[diff_idx]]
        while len(clash) > 0:
            d2[clash] = rng.integers(0, 10, size=len(clash))
            clash = clash[d2[clash] == d1[clash]]

    # pick actual images
    idx1 = np.array([rng.choice(pools[int(k)]) for k in d1], dtype=int)
    idx2 = np.array([rng.choice(pools[int(k)]) for k in d2], dtype=int)

    X1_raw = imgs[idx1].astype(np.float32)  # (N,784)
    X2_raw = imgs[idx2].astype(np.float32)  # (N,784)

    # shared style s ~ N(0,1)
    s = rng.normal(0, 1, size=(N, 1)).astype(np.float32)

    # independent confounder (no dependence on digits)
    p_c = float((p_high + p_low) / 2.0)   # or set to 0.5 if you prefer
    c = (rng.random(N) < p_c).astype(np.float32).reshape(-1, 1)  # (N,1)

    # --- build a standardized confounder signal and mix 50/50 with noise ---
    u = 2.0 * c - 1.0                      # {-1, +1}, shape (N,1)
    u = (u - u.mean()) / (u.std() + 1e-8)  # zero-mean, unit-variance
    eps = rng.normal(0, 1, size=(N,1)).astype(np.float32)
    mix = (u + eps) / np.sqrt(2.0)         # Var(u)=Var(eps)=1 -> each contributes 50%

    # fixed direction in pixel space to embed the latent into BOTH images
    rng_dir = np.random.default_rng(2024)
    v_dir = rng_dir.normal(0, 1, size=(784,)).astype(np.float32)
    v_dir /= (np.linalg.norm(v_dir) + 1e-8)

    # inject additively; alpha controls pixel-level strength (not the 50/50 composition)
    alpha_c_latent = 0.4
    delta = alpha_c_latent * mix * v_dir[None, :]   # (N,784) via broadcasting
    X1_raw = np.clip(X1_raw + delta, 0.0, 1.0)
    X2_raw = np.clip(X2_raw + delta, 0.0, 1.0)

    # apply shared style to BOTH images (Variant A)
    X1 = np.empty_like(X1_raw)
    X2 = np.empty_like(X2_raw)
    for i in range(N):
        X1[i] = apply_shared_style(X1_raw[i], s[i, 0], alpha_s)
        X2[i] = apply_shared_style(X2_raw[i], s[i, 0], alpha_s)

    # apply confounder mark (corner patch) to BOTH images
    for i in range(N):
        X1[i] = apply_confounder_mark(X1[i], c[i, 0], alpha_c, patch)
        X2[i] = apply_confounder_mark(X2[i], c[i, 0], alpha_c, patch)

    # targets (digits)
    targets1 = d1.astype(np.int64)
    targets2 = d2.astype(np.int64)
    targets3 = targets1.copy()  # kept for API compatibility

    # --- Y targets: two digits (img1, img2) + optional style + confounder-only ----
    Y = np.stack([d1.astype(np.float32), d2.astype(np.float32)], axis=1)  # (N,2)

    # Helper: make y given weights
    def make_y(ws_, wc_):
        # NOTE: we keep your current design without z1p in y_style (w1 unused here).
        return (ws_ * s.squeeze(1) + wc_ * c.squeeze(1) +
                rng.normal(0, sigma_y, size=N).astype(np.float32)).reshape(-1, 1)

    if add_style_target:

        # Redistribute total signal power P = ws^2 + wc^2 to hit desired splits
        P = max(ws**2 + wc**2, 1e-12)

        splits = [
            (0.1, 0.9),  # y_style_01_09
            (0.3, 0.7),  # y_style_03_07
            (0.6, 0.4),  # y_style_06_04
            (1.0, 0.0),  # y_style_10_00
        ]
        ys_list = []
        for ps, pc in splits:
            ws_r = np.sqrt(ps * P)
            wc_r = np.sqrt(pc * P)
            ys_list.append(make_y(ws_r, wc_r))

        y_style_block = np.concatenate(ys_list, axis=1)  # (N, 4)
        Y = np.concatenate([Y, y_style_block], axis=1)   # -> (N, 6)

    # Confounder-only outcome (depends ONLY on c + noise)
    y_confonly = (wc * c.squeeze(1) + rng.normal(0, sigma_y, size=N).astype(np.float32)).reshape(-1, 1)
    Y = np.concatenate([Y, y_confonly], axis=1)          # -> (N, 3) if no style, else (N, 7)

    # modalities array (2, N, 784)
    data = np.array([X1, X2], dtype=np.float32)

    return data, targets1, targets2, targets3, c.astype(np.float32), Y.astype(np.float32), s.astype(np.float32)

