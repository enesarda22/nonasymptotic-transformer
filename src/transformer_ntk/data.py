import math
from typing import Tuple, Optional, Sequence

import torch
import torch.nn.functional as F

from transformer_ntk.utils import get_activation


def make_relative_pos_channels(
    n: int,
    T: int,
    *,
    K_pos: int = 16,
    pos_kind: str = "rbf",  # 'rbf' | 'cheb' | 'exp'
    rbf_sigma: float = None,  # if None, set from spacing
    pos_scale: float = 1.0,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Return fixed (non-learnable) positional channels of shape (n, K_pos, T),
    encoding the *distance-to-last* token Δ = T-1-t.

    - 'rbf': Gaussian bumps over Δ (best for picking a specific lag)
    - 'cheb': Chebyshev polynomials T_k on Δ ∈ [0, T-1] mapped to [-1, 1]
    - 'exp': a bank of exponentials 0.5^(Δ / half_life) with log-spaced half-lives
    """
    dev = device or torch.device("cpu")
    tt = torch.arange(T, device=dev, dtype=dtype)  # (T,)
    delta = (T - 1 - tt).unsqueeze(0)  # (1, T), Δ=0 at last token

    if pos_kind.lower() == "rbf":
        # centers evenly spaced over [0, T-1]
        centers = torch.linspace(
            0, max(1, T - 1), K_pos, device=dev, dtype=dtype
        )  # (K_pos,)
        spacing = max(1.0, (T - 1) / max(1, K_pos - 1))
        sigma = torch.tensor(
            rbf_sigma if rbf_sigma is not None else spacing / 2.0,
            device=dev,
            dtype=dtype,
        )
        # (1,T,K_pos)
        P = torch.exp(-0.5 * ((delta.unsqueeze(-1) - centers) / sigma) ** 2)

    elif pos_kind.lower() == "cheb":
        # map Δ ∈ [0, T-1] to x ∈ [-1, 1]
        denom = max(1.0, float(T - 1))
        x = (2.0 * delta / denom) - 1.0  # (1, T)
        # Chebyshev T_0=1, T_1=x, T_k=2x T_{k-1} - T_{k-2}
        T0 = torch.ones_like(x).unsqueeze(-1)  # (1, T, 1)
        if K_pos == 1:
            P = T0
        else:
            T1 = x.unsqueeze(-1)  # (1, T, 1)
            polys = [T0, T1]
            for _ in range(2, K_pos):
                polys.append(2.0 * x.unsqueeze(-1) * polys[-1] - polys[-2])
            P = torch.cat(polys[:K_pos], dim=-1)  # (1, T, K_pos)

    elif pos_kind.lower() == "exp":
        # half-lives log-spaced between ~1 and ~T
        # channel k is 0.5^(Δ / half_life_k), monotone but with varied decay rates
        min_hl = 1.5
        max_hl = max(min_hl * 1.1, float(T))
        half_lives = torch.logspace(
            math.log10(min_hl), math.log10(max_hl), steps=K_pos, device=dev, dtype=dtype
        )  # (K_pos,)
        P = torch.pow(0.5, delta.unsqueeze(-1) / half_lives)  # (1, T, K_pos)

    else:
        raise ValueError("pos_kind must be 'rbf', 'cheb', or 'exp'.")

    # shape -> (n, K_pos, T)
    P = P.transpose(1, 2)  # (1, K_pos, T)
    # per-channel norming over time for stable scale (keeps them comparable to data)
    ch_std = P.std(dim=2, keepdim=True, unbiased=False).clamp_min(1e-6)
    P = pos_scale * (P / ch_std)
    P = P.expand(n, -1, -1)
    return P


@torch.no_grad()
def make_ar_forecasting_data(
    n: int,  # number of windows
    T: int,  # window length
    L: int,  # AR lag (L=1 -> AR(1), L=96 -> AR(96))
    alpha: float,  # x_t = alpha * x_{t-L} + eps_t (|alpha|<1 for stationarity)
    *,
    sigma_eps: float = 1.0,  # innovation std
    burn_in: int = 800,  # steps to reach stationarity
    # --- positional encodings (fixed/non-learnable) ---
    add_pos: bool = True,  # append fixed positional channels
    K_pos: int = 8,  # number of sin/cos frequency pairs (d_pos = 2*K_pos)
    tau0: float = 32.0,  # base period for the lowest frequency
    pos_scale: float = 1.0,  # scale for positional channels before any normalization
    # --- preprocessing ---
    normalize_columns: bool = True,  # enforce per-token ‖·‖₂ ≤ 1 if True
    standardize_y: bool = True,  # z-score y over this set
    # --- misc ---
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Natural AR(L) generator for next-step forecasting (sliding windows).

    • Two regimes (choose by L at run-time):
        - Short-term:  small L (e.g., L=1 or 4)
        - Long-term:   large L (e.g., L=96)   with the same alpha
      Keep T >= L so the informative token is inside the window.

    • Appends fixed sin/cos positional channels so attention can resolve a specific lag.
      These channels are non-learnable and keep your theory intact.

    Returns:
      X_all: (n, d, T),  d = 1 (+ 2*K_pos if add_pos)
      y_all: (n,)
    """
    dev = device or torch.device("cpu")
    if seed is not None:
        torch.manual_seed(seed)

    assert isinstance(L, int) and L >= 1, "L must be an integer ≥ 1."
    assert -1.0 < alpha < 1.0, "|alpha| must be < 1 for stationarity."
    assert T >= L, f"Need T >= L; got T={T}, L={L}."

    # Total length: burn-in + n windows + T steps + L to allow lagged recursion
    need = burn_in + n + T + L
    x = torch.empty(need, device=dev, dtype=dtype)

    # Initialize and generate AR(L): each residue class mod L is AR(1) with coeff alpha
    x[:L].normal_(0.0, 1.0)
    eps = torch.randn(need - L, device=dev, dtype=dtype) * sigma_eps
    for t in range(L, need):
        x[t] = alpha * x[t - L] + eps[t - L]

    # Sliding windows and next-step targets
    starts = torch.arange(burn_in, burn_in + n, device=dev)
    Xc = torch.stack([x[s : s + T] for s in starts], dim=0).unsqueeze(1)  # (n,1,T)
    y_all = x[starts + T]  # (n,)

    # Fixed (non-learnable) positional channels: multi-frequency sin/cos
    if add_pos:
        t = torch.arange(T, device=dev, dtype=dtype).unsqueeze(0)  # (1,T)
        omegas = (
            2.0
            * math.pi
            / (tau0 * (2.0 ** torch.arange(K_pos, device=dev, dtype=dtype)))
        )
        S = torch.sin(t.unsqueeze(-1) * omegas)  # (1,T,K_pos)
        C = torch.cos(t.unsqueeze(-1) * omegas)  # (1,T,K_pos)
        P = torch.cat([S, C], dim=-1).transpose(1, 2)  # (1, 2*K_pos, T)
        P = (pos_scale * P).expand(n, -1, -1)  # (n, 2*K_pos, T)
        X_all = torch.cat([Xc, P], dim=1)  # (n, 1+2*K_pos, T)
    else:
        X_all = Xc  # (n, 1, T)

    # Optional: enforce per-token ‖·‖₂ ≤ 1 (matches the theory assumption if enabled)
    if normalize_columns:
        # token-wise ℓ2 norms: (n, 1, T)
        norms = X_all.norm(dim=1, keepdim=True).clamp_min(1.0)
        X_all = X_all / norms

    # Optional: standardize targets over this set
    if standardize_y:
        mu = y_all.mean()
        sd = y_all.std(unbiased=False).clamp_min(1e-12)
        y_all = (y_all - mu) / sd

    return X_all.to(dev), y_all.to(dev)


# @torch.no_grad()
# def make_ar_forecasting_data(
#     n: int,  # number of windows
#     T: int,  # window length
#     L: int,  # AR lag (L=1 -> AR(1), L=96 -> AR(96))
#     alpha: float,  # coefficient in x_t = alpha * x_{t-L} + eps_t
#     *,
#     sigma_eps: float = 1.0,  # innovation std
#     burn_in: int = 500,  # steps to reach stationarity
#     normalize_columns: bool = True,  # enforce per-token ‖·‖₂ ≤ 1
#     standardize_y: bool = True,  # z-score y over this set
#     seed: Optional[int] = None,
#     device: Optional[torch.device] = None,
#     dtype: torch.dtype = torch.float32,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Builds a *single* AR(L) series and sliding windows for next-step prediction.
#     For each start s, window is x[s : s+T] and target is x[s+T].
#     Returns:
#       X_all: (n, d, T)  with d = 1 (+1 if add_pos)
#       y_all: (n,)
#     Notes:
#       • Requires |alpha|<1 for stationarity. Recommend T >= L so the window covers the informative lag.
#       • Positional channel is fixed (non-learnable), so the model is unchanged (theory-preserving).
#     """
#     dev = device or torch.device("cpu")
#     if seed is not None:
#         torch.manual_seed(seed)
#
#     assert L >= 1, "L must be >= 1."
#     assert -1.0 < alpha < 1.0, "|alpha| must be < 1 for stationarity."
#     # Total length needed: burn-in + n windows + T steps for each window
#     need = burn_in + n + T
#
#     # Generate AR(L): each residue class mod L is an AR(1) with coeff alpha
#     x = torch.empty(need, device=dev, dtype=dtype)
#     std0 = sigma_eps / math.sqrt(max(1e-12, 1.0 - alpha * alpha))
#     if L == 1:
#         x[0].normal_(0.0, std0)
#         eps = torch.randn(need - 1, device=dev, dtype=dtype) * sigma_eps
#         for t in range(1, need):
#             x[t] = alpha * x[t - 1] + eps[t - 1]
#     else:
#         # Initialize first L entries from stationary variance of the L-step subsequences
#         x[:L].normal_(0.0, std0)
#         eps = torch.randn(need - L, device=dev, dtype=dtype) * sigma_eps
#         for t in range(L, need):
#             x[t] = alpha * x[t - L] + eps[t - L]
#
#     # Sliding windows and next-step targets
#     starts = torch.arange(burn_in, burn_in + n, device=dev)
#     Xc = torch.stack([x[s : s + T] for s in starts], dim=0).unsqueeze(1)  # (n,1,T)
#     y_all = x[starts + T]  # (n,)
#     X_all = Xc  # (n,1,T)
#
#     # Enforce per-token ‖·‖₂ ≤ 1 (matches your assumption)
#     if normalize_columns:
#         s = X_all.abs().amax(dim=(1, 2), keepdim=True)
#         s = torch.clamp(s, min=1.0)
#         X_all = X_all / s
#
#     # Standardize y over this set (use train-only stats if you split later)
#     if standardize_y:
#         mu = y_all.mean()
#         sd = y_all.std(unbiased=False).clamp_min(1e-12)
#         y_all = (y_all - mu) / sd
#
#     return X_all.to(dev), y_all.to(dev)


@torch.no_grad()
def make_pointer_dataset(
    n: int,
    T: int,
    *,
    d_content: int = 1,  # content channels (the label depends only on channel 0)
    code_dim: int = 64,  # positional code channels
    code_kind: str = "gauss",  # "gauss" or "fourier"
    lags: Optional[
        Sequence[int]
    ] = None,  # choose lag L from this set per sample; if None, sample uniform 1..T-1
    k: int = 1,  # number of pointers to sum (k=1 reduces to k-th last)
    noise_std: float = 0.0,
    standardize_y: bool = True,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      X: (n, d_content + code_dim, T)
      y: (n,) where y = sum_{j=1}^k x_{t*_j}  and  t*_j = T - L_j
    Notes:
      * The last token carries the *query* (its code equals the code of the target index/indices).
      * We zero the last token's *content* so there is no content shortcut at t=T.
      * For stable optimization across L, y is standardized (optional).
    """
    dev = device or torch.device("cpu")
    if seed is not None:
        torch.manual_seed(seed)

    # --- content ---
    Xc = torch.randn(n, d_content, T, device=dev, dtype=dtype)  # i.i.d. content

    # --- position codes C: (code_dim, T) ---
    if code_kind == "gauss":
        C = torch.randn(code_dim, T, device=dev, dtype=dtype)
    elif code_kind == "fourier":
        K = max(1, code_dim // 2)
        t = torch.linspace(0.0, 1.0, T, device=dev, dtype=dtype)
        freqs = torch.arange(1, K + 1, device=dev, dtype=dtype).view(-1, 1)
        S = torch.sin(2 * math.pi * freqs * t)  # (K,T)
        Cc = torch.cos(2 * math.pi * freqs * t)  # (K,T)
        C = torch.cat([S, Cc], dim=0)
        if C.shape[0] < code_dim:
            C = torch.cat([C, t.view(1, -1)], dim=0)
        C = C[:code_dim]
    else:
        raise ValueError("code_kind must be 'gauss' or 'fourier'")
    C = C / C.norm(dim=0, keepdim=True).clamp_min(1e-12)  # per-token unit-norm

    # batch broadcast
    P = C.unsqueeze(0).expand(n, -1, -1)  # (n, code_dim, T)

    # --- choose target indices ---
    if lags is None:
        L = torch.randint(1, T, (n, k), device=dev)  # lags in {1,...,T-1}
    else:
        L_choices = torch.tensor(list(lags), device=dev)
        idx = torch.randint(0, len(L_choices), (n, k), device=dev)
        L = L_choices[idx]
    t_star = (T - L).clamp_min(0)  # (n, k)

    # label: sum of selected contents (channel 0)
    b = torch.arange(n, device=dev).unsqueeze(1).expand_as(t_star)
    y = Xc[b, 0, t_star].sum(dim=1)  # (n,)

    # --- build X and write the query into the last token's *codes* ---
    X = torch.cat([Xc, P], dim=1)  # (n, d_content+code_dim, T)
    # last token content off to prevent a trivial shortcut
    X[:, :d_content, -1] = 0.0
    # query: set last token's code to sum of the target codes (unit-normalize per sample)
    q = torch.zeros(n, code_dim, device=dev, dtype=dtype)
    for j in range(k):
        q += P[torch.arange(n, device=dev), :, t_star[:, j]]
    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
    X[:, d_content:, -1] = q

    # optional label noise + standardization
    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)
    if standardize_y:
        y = (y - y.mean()) / (y.std(unbiased=False) + 1e-12)
    return X, y


def add_pos_channel(
    X: torch.Tensor,
    mode: str = "ramp",
    gamma: float = None,
    *,
    # New options for high-dim codes:
    code_dim: int = 16,  # number of positional code channels to add
    code_kind: str = "gauss",  # "gauss" or "fourier"
    codes: torch.Tensor = None,  # optional precomputed codes (code_dim, T)
    last_query: torch.Tensor = None,  # optional query vector to overwrite last token code (code_dim,)
    zero_last_content: bool = False,  # set content channels at t=T to 0 (helps pointer setups)
    normalize_columns: bool = True,  # keep per-token ||·||_2 ≤ 1
) -> torch.Tensor:
    """
    X: (B, d, T)  ->  X2: (B, d + extra, T)
      - mode="ramp":   append 1D ramp p_t=(t+1)/T
      - mode="geom":   append 1D geometric p_t=gamma^(T-1-t)  (gamma defaults to 0.95 if None)
      - mode="codes":  append code_dim-D codes P[:, :, t] ~ near-orthonormal position codes
                       (gaussian or fourier). Optionally set the last token code to `last_query`
                       and zero the last token's content to remove content shortcuts.
    """
    assert X.dim() == 3, "X must be (B, d, T)"
    B, d, T = X.shape
    device, dtype = X.device, X.dtype

    if mode == "ramp":
        t = torch.arange(T, device=device, dtype=dtype)
        p = (t + 1) / T
        P = p.view(1, 1, T).expand(B, 1, T)  # (B,1,T)
        X2 = torch.cat([X, P], dim=1)

    elif mode == "geom":
        t = torch.arange(T, device=device, dtype=dtype)
        g = float(gamma) if gamma is not None else 0.95
        p = g ** (T - 1 - t)
        P = p.view(1, 1, T).expand(B, 1, T)  # (B,1,T)
        X2 = torch.cat([X, P], dim=1)

    elif mode == "codes":
        # Build (code_dim, T) codes matrix C where each column is the code for timestep t
        if codes is not None:
            C = codes.to(device=device, dtype=dtype)
            assert C.shape == (
                code_dim,
                T,
            ), f"codes must be (code_dim, T), got {C.shape}"
        else:
            if code_kind == "gauss":
                C = torch.randn(code_dim, T, device=device, dtype=dtype)
            elif code_kind == "fourier":
                # Build sin/cos features up to K=floor(code_dim/2)
                K = max(1, code_dim // 2)
                t = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype)
                freqs = torch.arange(1, K + 1, device=device, dtype=dtype).view(-1, 1)
                S = torch.sin(2 * math.pi * freqs * t)  # (K,T)
                Cc = torch.cos(2 * math.pi * freqs * t)  # (K,T)
                C = torch.cat([S, Cc], dim=0)  # (2K,T)
                if C.shape[0] < code_dim:
                    # pad with a ramp if we need one more channel
                    ramp = t.view(1, -1)
                    C = torch.cat([C, ramp], dim=0)
                C = C[:code_dim]
            else:
                raise ValueError("code_kind must be 'gauss' or 'fourier'")

        # Normalize each column to unit norm (so dot-products are comparable across t)
        C = C / C.norm(dim=0, keepdim=True).clamp_min(1e-12)  # (code_dim, T)

        # Broadcast to batch
        P = C.unsqueeze(0).expand(B, -1, -1)  # (B, code_dim, T)

        # Optional: overwrite last token's code with a provided query vector
        if last_query is not None:
            q = last_query.to(device=device, dtype=dtype).view(
                1, -1, 1
            )  # (1,code_dim,1)
            assert q.shape[1] == code_dim, "last_query must have shape (code_dim,)"
            q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
            P[:, :, -1] = q.expand(B, -1, 1).squeeze(-1)

        X2 = torch.cat([X, P], dim=1)  # (B, d+code_dim, T)

        # Optional: zero out content channels at the last token (helps avoid content shortcuts)
        if zero_last_content:
            X2[:, :d, -1] = 0

    else:
        raise ValueError("mode must be 'ramp', 'geom', or 'codes'")

    if normalize_columns:
        # Enforce per-token ||·||_2 ≤ 1 across all channels
        col_norm = X2.norm(dim=1, keepdim=True).clamp_min(1e-12)  # (B,1,T)
        X2 = X2 / torch.maximum(col_norm, torch.ones_like(col_norm))

    return X2


@torch.no_grad()
def make_ar1_dataset(
    n: int,
    T: int,
    r: float,  # AR(1) coefficient in (-1,1); larger -> longer memory
    *,
    sigma_eps: float = 1.0,  # std of AR innovation ε_t
    burn_in: int = 200,  # steps to reach stationarity before recording
    target: str = "geom",  # "geom" (geometric sum), "last" (x_T), or "linear" (custom weights)
    gamma: Optional[float] = None,  # decay for "geom"; defaults to r
    weights: Optional[torch.Tensor] = None,  # (T,) for "linear" (most-recent last)
    noise_std: float = 0.0,  # optional observation noise on y
    standardize_y: bool = True,  # z-score y for stable scales across settings
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      X: (n, 1, T) sequences from AR(1): x_t = r x_{t-1} + ε_t
      y: (n,) scalar targets per sequence

    Targets:
      - "geom": y = sum_{k=0}^{T-1} gamma^k * x_{T-k}
      - "last": y = x_T
      - "linear": y = sum_{t=1}^{T} weights[t-1] * x_t   (weights shape (T,))
    Notes:
      * Stationary init: x_0 ~ N(0, sigma_eps^2 / (1 - r^2)).
      * For "geom", if gamma is None, uses gamma = r.
      * X’s time dimension is last (matches your models): (batch, d=1, T).
    """
    dev = device or torch.device("cpu")
    if seed is not None:
        torch.manual_seed(seed)

    assert -1.0 < r < 1.0, "AR(1) coefficient r must be in (-1,1)."
    n_total_steps = burn_in + T

    # Vectorized simulation for n sequences (d=1)
    # x0 ~ stationary N(0, sigma_eps^2 / (1 - r^2))
    std0 = sigma_eps / math.sqrt(max(1e-12, 1.0 - r * r))
    x = torch.empty(n, n_total_steps, device=dev, dtype=dtype)
    x[:, 0].normal_(mean=0.0, std=std0)
    eps = torch.randn(n, n_total_steps - 1, device=dev, dtype=dtype) * sigma_eps
    for t in range(1, n_total_steps):
        x[:, t] = r * x[:, t - 1] + eps[:, t - 1]

    # Keep the last T points
    xT = x[:, -T:]  # (n, T)
    X = xT.unsqueeze(1)  # (n, 1, T)  (d=1)

    # Targets
    if target == "last":
        y = xT[:, -1]  # x_T

    elif target == "geom":
        g = r if gamma is None else float(gamma)
        # weights w_t = g^(T-1-t) so that y = sum_{k=0}^{T-1} g^k x_{T-k}
        powers = torch.arange(T, device=dev, dtype=dtype)
        w = g ** (T - 1 - powers)  # (T,)
        y = (xT * w).sum(dim=1)  # (n,)

    elif target == "linear":
        assert weights is not None and weights.numel() == T, (
            "weights must be (T,) for target='linear'."
        )
        w = weights.to(device=dev, dtype=dtype).view(1, T)
        y = (xT * w).sum(dim=1)

    else:
        raise ValueError("target must be one of {'geom','last','linear'}.")

    # Optional observation noise on y
    if noise_std and noise_std > 0:
        y = y + torch.randn_like(y) * noise_std

    # Standardize y for stable loss scales (does not change relative comparisons)
    if standardize_y:
        y = (y - y.mean()) / (y.std(unbiased=False) + 1e-12)

    return X, y


def _normalize_columns(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize each column (token) to have L2 <= 1. X: (..., d, T)
    """
    norms = torch.linalg.vector_norm(X, ord=2, dim=-2, keepdim=True).clamp_min(
        eps
    )  # (..., 1, T)
    scale = torch.clamp(1.0 / norms, max=1.0)  # do not blow up small vectors
    return X * scale


def _ar1_cov_T(T: int, rho: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """AR(1) Toeplitz covariance across tokens: C[i,j] = rho**|i-j|."""
    idx = torch.arange(T, device=device)
    C = rho ** (idx[:, None] - idx[None, :]).abs()
    return C.to(dtype)


def make_inputs(
    n: int,
    d: int,
    T: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    *,
    token_corr: str | None = None,  # None | "ar1" | "custom"
    rho: float = 0.8,  # used when token_corr=="ar1"
    cov_T: torch.Tensor | None = None,  # (T,T) used when token_corr=="custom"
    normalize: bool = True,
) -> torch.Tensor:
    """
    Sample X ~ N(0, I) and (optionally) correlate tokens (columns) via a T×T covariance.
    Then enforce column-wise L2 <= 1.
      token_corr=None: i.i.d. tokens
      token_corr="ar1": AR(1) with parameter rho in [0,1)
      token_corr="custom": use provided cov_T (T×T, PSD)
    Returns: X ∈ R^{n×d×T}
    """
    X = torch.randn(n, d, T, device=device, dtype=dtype)

    if token_corr is not None:
        if token_corr == "ar1":
            C = _ar1_cov_T(T, rho=rho, device=device, dtype=dtype)
        elif token_corr == "custom":
            assert cov_T is not None and cov_T.shape == (T, T), "cov_T must be (T,T)"
            C = cov_T.to(device=device, dtype=dtype)
        else:
            raise ValueError(f"unknown token_corr={token_corr}")

        # Cholesky (jitter for numerical stability)
        C_jit = C + 1e-6 * torch.eye(T, device=device, dtype=dtype)
        L = torch.linalg.cholesky(C_jit)  # C = L L^T

        # Correlate along the last (token) dimension: (...,T) @ (T,T) -> (...,T)
        X = torch.matmul(X, L.T)

    if normalize:
        X = _normalize_columns(X)
    return X


def make_dataset(
    n: int,
    d: int,
    T: int,
    activation="tanh",
    R=16,
    num_mc=4096,
    nu=3.0,
    noise_std: float = 0.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    method="teacher",
    seed=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build (X, y) using the limit-model teacher \tilde f with Monte Carlo.
    """
    X = make_inputs(n, d, T, device=device, dtype=dtype)

    if method == "teacher":
        with torch.no_grad():
            _, y, _, _ = lin_and_tilde_outputs_multianchor(
                X=X,
                teacher_m=num_mc,
                m=1,
                num_anchors=R,
                nu_c=nu,
                nu_u=nu,
                nu_w=nu,
                activation=activation,
                seed=seed,
                device=device,
            )
    elif method == "kernel":
        idx = torch.randperm(n)[:R]
        anchors = X[idx].detach().clone()  # (R,d,T)

        # Gram on anchors
        K_RR = _kernel_section_mc(anchors, anchors, num_mc=num_mc, activation="tanh")

        # Sample alpha with target RKHS norm ||f*||_H = nu
        alpha0 = torch.randn(R)
        norm2 = alpha0 @ (K_RR @ alpha0)
        alpha = (nu / math.sqrt(float(norm2))) * alpha0

        # Labels for all X
        K_XR = _kernel_section_mc(X, anchors, num_mc=num_mc, activation="tanh")
        y = K_XR @ alpha
    else:
        raise ValueError(f"Unsupported method: {method}")

    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)

    return X, y


@torch.no_grad()
def lin_and_tilde_outputs_multianchor(
    X: torch.Tensor,  # (B,d,T) or (d,T)
    *,
    teacher_m: int,  # big MC for ~ground-truth \tilde f
    m: int,  # smaller MC for f_lin
    num_anchors: int = 8,
    anchors: Optional[torch.Tensor] = None,  # (R,d,T); if None, pick subset of X
    alpha: Optional[torch.Tensor] = None,  # (R,); L1 will be normalized to 1
    nu_c: float = 1.0,
    nu_u: float = 1.0,
    nu_w: float = 1.0,
    activation: str = "tanh",
    temperature: float = 1.0,
    chunk: int = 2048,  # chunk over MC samples to save memory
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (y_lin, y_tilde, anchors, alpha) for the same X.

    Transport map v is a multi-anchor mixture:
      v(φ) = Σ_r α_r * v^{(r)}(φ),
    where each v^{(r)} is the "representer at anchor X^{(r)}", sup-bounded by (nu_c,nu_u,nu_w):
      v_c^{(r)}(φ)   = (nu_c/σ0) * φ_c(X^{(r)}; φ)
      v_u^{(r)}(φ)   = (nu_u/σ1) * φ_u(X^{(r)}; φ)
      v_w^{(r)}(φ)   = clip_F( φ_w(X^{(r)}; φ), nu_w )   # Frobenius clip
    with Σ_r |α_r| ≤ 1 (we enforce Σ|α| = 1 by normalization if alpha is None).

    Then:
      f_lin(X; \bar φ_m)  = (1/m) Σ_{k=1}^m ⟨ φ(X; φ_k), v(φ_k) ⟩
      \tilde f(X; v)      ≈ (1/teacher_m) Σ_{k=1}^{teacher_m} ⟨ φ(X; φ_k), v(φ_k) ⟩
    where φ are the per-branch features (c, u, w).
    """
    # -------------- shapes & devices
    if X.dim() == 2:
        X = X.unsqueeze(0)
    B, d, T = X.shape
    dev = device or X.device
    X = X.to(dev, dtype)

    # activations
    sigma, sigma_p = get_activation(activation)
    inv_tau = 1.0 / max(1e-12, float(temperature))
    sigma0, sigma1 = (1.0, 1.0)  # activation bounds

    # -------------- anchors & coefficients
    if anchors is None:
        # pick anchors from X (no grad), or if B < R, tile/loop
        R = num_anchors
        idx = torch.randperm(B, device=dev)[: min(B, R)]
        anchors_sel = X[idx].detach().clone()
        if anchors_sel.shape[0] < R:
            # repeat to reach R (won't matter statistically)
            reps = (R + anchors_sel.shape[0] - 1) // anchors_sel.shape[0]
            anchors = anchors_sel.repeat(reps, 1, 1)[:R]
        else:
            anchors = anchors_sel
    else:
        anchors = anchors.to(dev, dtype)
        R = anchors.shape[0]

    if alpha is None:
        a = torch.randn(R, device=dev, dtype=dtype)
        # enforce sum |α| = 1 (so mixture respects sup bounds)
        alpha = a / (a.abs().sum().clamp_min(1e-12))
    else:
        alpha = alpha.to(dev, dtype)
        s = alpha.abs().sum().clamp_min(1e-12)
        alpha = alpha / s

    # precompute last-token norms/dots for anchors and X
    X_T = X[:, :, -1]  # (B,d)
    A_T = anchors[:, :, -1]  # (R,d)
    A_T_norm = A_T.norm(dim=1)  # (R,)

    # -------------- helpers: sample φ and compute features φ_c, φ_u, vec_w (rank-1 part)
    def _randn(shape, g=None):
        return torch.randn(
            shape, device="cpu" if g is not None else dev, dtype=dtype, generator=g
        )

    def _mc_estimate(num_mc: int) -> torch.Tensor:
        """
        Compute y(X) = E_φ [ Σ_r α_r (nu_c/σ0) φ_c(X)φ_c(Xr)
                            + Σ_r α_r (nu_u/σ1) <φ_u(X),φ_u(Xr)>
                            + Σ_r α_r <φ_w(X), clip_w(φ_w(Xr))>_F ]
        via MC with 'num_mc' samples.
        """
        # RNG on CPU for determinism if seed is set
        g = None
        if seed is not None:
            g = torch.Generator(device="cpu")
            g.manual_seed(seed + num_mc)  # different streams for m vs teacher_m

        total = torch.zeros(B, device=dev, dtype=dtype)
        done = 0
        count = 0
        while done < num_mc:
            k = min(chunk, num_mc - done)
            done += k
            count += k

            # φ samples
            U = _randn((k, d), g).to(dev)
            W = _randn((k, d, d), g).to(dev)

            # ------ features at X  (B,k,•)
            # WX_T: (k,d,d) @ (B,d) -> (B,k,d)
            WX_T = torch.einsum("kdd,bd->bkd", W, X_T)
            # z = X^T W X_T: (B,T,d)·(B,k,d)^T -> (B,k,T)
            z = torch.einsum("btd,bkd->bkt", X.transpose(1, 2), WX_T) * inv_tau
            αX = F.softmax(z, dim=-1)  # (B,k,T)
            aX = torch.einsum("bdt,bkt->bkd", X, αX)  # (B,k,d)
            sX = torch.einsum("kd,bkd->bk", U, aX)  # (B,k)
            φcX = sigma(sX)  # (B,k)
            σpX = sigma_p(sX)  # (B,k)

            # φ_u(X): σ'(s) * a(X)     (B,k,d)
            φuX = σpX.unsqueeze(-1) * aX

            # For φ_w: use rank-1 factorization φ_w = (σ' M U) X_T^T.
            # Compute M U via covariance trick:
            firstX = torch.einsum("bdt,bkt,bDt->bkdD", X, αX, X)  # (B,k,d,d)
            covX = firstX - torch.einsum("bkd,bkD->bkdD", aX, aX)
            MU_X = torch.einsum("bkdD,kD->bkd", covX, U)  # (B,k,d)
            vecX = σpX.unsqueeze(-1) * MU_X  # (B,k,d)

            # ------ features at anchors (R,k,•)
            WX_Ta = torch.einsum("kdd,Rd->Rkd", W, A_T)  # (R,k,d)
            za = torch.einsum("Rtd,Rkd->Rkt", anchors.transpose(1, 2), WX_Ta) * inv_tau
            αA = F.softmax(za, dim=-1)  # (R,k,T)
            aA = torch.einsum("Rdt,Rkt->Rkd", anchors, αA)  # (R,k,d)
            sA = torch.einsum("kd,Rkd->Rk", U, aA)  # (R,k)
            φcA = sigma(sA)  # (R,k)
            σpA = sigma_p(sA)  # (R,k)
            φuA = σpA.unsqueeze(-1) * aA  # (R,k,d)

            firstA = torch.einsum("Rdt,Rkt,RDt->RkdD", anchors, αA, anchors)
            covA = firstA - torch.einsum("Rkd,RkD->RkdD", aA, aA)
            MU_A = torch.einsum("RkdD,kD->Rkd", covA, U)  # (R,k,d)
            vecA = σpA.unsqueeze(-1) * MU_A  # (R,k,d)

            # Frobenius norm of φ_w(Anchor): ||vecA|| * ||A_T||
            vecA_norm = vecA.norm(dim=2)  # (R,k)
            # scaling for clipping to ||·||_F <= nu_w
            # s_rk = min(1, nu_w / (||vecA_rk|| * ||A_T_r||))
            denom = (vecA_norm * A_T_norm.view(-1, 1)).clamp_min(1e-12)  # (R,k)
            s_clip = torch.clamp(nu_w / denom, max=1.0)  # (R,k)

            # ------ accumulate mixture over anchors r
            # c-branch: (nu_c/σ0) * Σ_r α_r * [ φ_c(X) ⊙ φ_c(Xr) ] averaged over k
            term_c = torch.einsum("r, bk, rk -> b", alpha, φcX, φcA) * (nu_c / sigma0)

            # u-branch: (nu_u/σ1) * Σ_r α_r * [ <φ_u(X), φ_u(Xr)> ] averaged over k
            # inner_u: (B,k) with dot over d, then mix over r
            inner_u = torch.einsum("bkd, rkd -> brk", φuX, φuA)  # (B,R,k)
            term_u = torch.einsum("r, brk -> b", alpha, inner_u) * (nu_u / sigma1)

            # w-branch: Σ_r α_r * <φ_w(X), clip(φ_w(Xr))>_F
            # <φ_w(X), φ_w(Xr)>_F = (vecX·vecA_r) * (X_T·A_T_r)
            xT_dot = torch.einsum("bd, rd -> br", X_T, A_T)  # (B,R)
            inner_vec = torch.einsum("bkd, rkd -> brk", vecX, vecA)  # (B,R,k)
            term_w = torch.einsum(
                "r, brk, rk, br -> b", alpha, inner_vec, s_clip, xT_dot
            )

            # average over k MC samples
            total += term_c + term_u + term_w

        total /= float(count)
        return total  # (B,)

    # y_lin: MC with width = m
    y_lin = _mc_estimate(m)

    # y_tilde: MC with width = teacher_m (independent stream)
    y_tilde = _mc_estimate(teacher_m)

    return y_lin, y_tilde, anchors, alpha


def sample_phi(num: int, d: int, device=None, dtype=torch.float32):
    """
    Sample φ=(c,U,W) from the init distribution (NO pairing for teacher).
    c ~ Rad(±1), U ~ N(0, I_d), W_{kℓ} ~ N(0,1) i.i.d.
    """
    c = (torch.randint(0, 2, (num,), device=device) * 2 - 1).to(dtype)
    U = torch.randn(num, d, device=device, dtype=dtype)
    W = torch.randn(num, d, d, device=device, dtype=dtype)
    return c, U, W


def _kernel_section_mc(
    X: torch.Tensor,  # (B,d,T)
    X_anchors: torch.Tensor,  # (R,d,T)
    *,
    num_mc: int,
    activation: str = "tanh",
    temperature: float = 1.0,
    chunk: int = 1024,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Monte-Carlo estimate of the NTK section(s):
      K(X, X') = E[ φ_c(X)φ_c(X') + <φ_u(X),φ_u(X')> + <φ_w(X),φ_w(X')>_F ].
    Returns K ∈ R^{B×R}. No clipping; intended as a teacher primitive.
    """
    if X.dim() == 2:
        X = X.unsqueeze(0)
    if X_anchors.dim() == 2:
        X_anchors = X_anchors.unsqueeze(0)
    B, d, T = X.shape
    R = X_anchors.shape[0]
    dev = device or X.device
    dt = dtype or X.dtype
    inv_tau = 1.0 / max(1e-12, float(temperature))
    X = X.to(dev, dt)
    Xa = X_anchors.to(dev, dt)
    X_T = X[:, :, -1]  # (B,d)
    Xa_T = Xa[:, :, -1]  # (R,d)
    xT_dot = torch.einsum("bd,Rd->bR", X_T, Xa_T)  # (B,R)
    sigma, sigma_p = get_activation(activation)
    K_sum = torch.zeros(B, R, device=dev, dtype=dt)
    done = 0
    while done < num_mc:
        k = min(chunk, num_mc - done)
        done += k
        # φ samples
        _, U, W = sample_phi(k, d, device=dev, dtype=dt)  # c not needed
        # ----- Features at X -----
        WX_T = torch.einsum("mdk,bk->bmd", W, X_T)  # (B,k,d)
        z = torch.einsum("btd,bmd->bmt", X.transpose(1, 2), WX_T)  # (B,k,T)
        z = z * inv_tau
        alpha = torch.softmax(z, dim=-1)  # (B,k,T)
        a = torch.einsum("bdt,bmt->bmd", X, alpha)  # (B,k,d)
        s = torch.einsum("md,bmd->bm", U, a)  # (B,k)
        phi_cX = sigma(s)  # (B,k)
        sigpX = sigma_p(s)  # (B,k)
        first = torch.einsum("bdt,bmt,bDt->bmdD", X, alpha, X)  # (B,k,d,d)
        cov = first - torch.einsum("bmd,bmD->bmdD", a, a)  # (B,k,d,d)
        MU = torch.einsum("bmdD,mD->bmd", cov, U)  # (B,k,d)
        vecX = sigpX.unsqueeze(-1) * MU  # (B,k,d)
        # ----- Features at anchors -----
        WXa_T = torch.einsum("mdk,Rk->Rmd", W, Xa_T)  # (R,k,d)
        za = torch.einsum("Rtd,Rmd->Rmt", Xa.transpose(1, 2), WXa_T)  # (R,k,T)
        za = za * inv_tau
        alphaa = torch.softmax(za, dim=-1)  # (R,k,T)
        aa = torch.einsum("Rdt,Rmt->Rmd", Xa, alphaa)  # (R,k,d)
        sa = torch.einsum("md,Rmd->Rm", U, aa)  # (R,k)
        phi_cA = sigma(sa)  # (R,k)
        sigpA = sigma_p(sa)  # (R,k)
        firsta = torch.einsum("Rdt,Rmt,RDt->RmdD", Xa, alphaa, Xa)  # (R,k,d,d)
        cova = firsta - torch.einsum("Rmd,RmD->RmdD", aa, aa)  # (R,k,d,d)
        MUA = torch.einsum("RmdD,mD->Rmd", cova, U)  # (R,k,d)
        vecA = sigpA.unsqueeze(-1) * MUA  # (R,k,d)
        # ----- Accumulate terms -----
        # c-term:   mean_k [ φ_c(X) φ_c(X') ]
        term_c = torch.einsum("bk,rk->br", phi_cX, phi_cA)
        # u-term:   mean_k [ < σ'(s) a , σ'(s') a' > ]
        term_u = torch.einsum(
            "bkd,rkd->br", sigpX.unsqueeze(-1) * a, sigpA.unsqueeze(-1) * aa
        )
        # w-term:   mean_k [ (vecX · vecA) * (X_T · X_T') ]
        inner_vec = torch.einsum("bkd,rkd->br", vecX, vecA)  # (B,R)
        term_w = inner_vec * xT_dot
        K_sum += term_c + term_u + term_w
    K = K_sum / float(num_mc)
    return K
