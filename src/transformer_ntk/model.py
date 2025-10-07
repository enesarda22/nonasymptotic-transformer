import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    """
    PyTorch module for
      a(X; W_i) = X softmax(X^T W_i X_T)           in R^d
      h_i       = sigma(U_i^T a(X; W_i))           in R
      f(X)      = (1/sqrt(m)) sum_i c_i * h_i      in R

    Shapes:
      X: (B, d, T)  with last column X_T = X[:, :, -1]
      W: (m, d, d), U: (m, d), c: (m,)
    """

    def __init__(
        self,
        d: int,
        T: int,
        m: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
        symmetric_init: bool = True,  # use the symmetric init (requires m even)
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        mask_last_key=False,
    ):
        super().__init__()
        self.d, self.T, self.m = d, T, m
        self.activation = activation
        self.mask_last_key = mask_last_key

        # Parameters
        self.W = nn.Parameter(torch.empty(m, d, d, device=device, dtype=dtype))
        self.U = nn.Parameter(torch.empty(m, d, device=device, dtype=dtype))
        self.c = nn.Parameter(torch.empty(m, device=device, dtype=dtype))

        # Initialization
        self._reset_parameters(paired_init=symmetric_init)

        # Keep copies of init for PGD-style projection (centers of balls)
        self.register_buffer("W0", self.W.detach().clone())
        self.register_buffer("U0", self.U.detach().clone())
        self.register_buffer("c0", self.c.detach().clone())

        self.register_buffer(
            "sqrt_m", torch.tensor(math.sqrt(m), dtype=dtype, device=device)
        )

    def _reset_parameters(self, paired_init: bool = True):
        """
        Pairwise-symmetric init (as in the theory):
          - For i=1..m/2: W_i ~ N(0,1) iid, U_i ~ N(0,I_d), c_i ~ Rad(±1)
          - For i=m/2+1..m: W_i = W_{i-m/2}, U_i = U_{i-m/2}, c_i = -c_{i-m/2}
        Falls back to i.i.d. init if m is odd or paired_init=False.
        """
        m, d = self.m, self.d
        if paired_init and (m % 2 == 0):
            half = m // 2
            with torch.no_grad():
                W_half = torch.randn(
                    half, d, d, dtype=self.W.dtype, device=self.W.device
                )
                U_half = torch.randn(half, d, dtype=self.U.dtype, device=self.U.device)
                c_half = (
                    torch.randint(0, 2, (half,), device=self.c.device, dtype=torch.long)
                    * 2
                    - 1
                ).to(self.c.dtype)

                self.W[:half].copy_(W_half)
                self.W[half:].copy_(W_half)

                self.U[:half].copy_(U_half)
                self.U[half:].copy_(U_half)

                self.c[:half].copy_(c_half)
                self.c[half:].copy_(-c_half)
        else:
            # i.i.d. fallback
            with torch.no_grad():
                self.W.normal_(mean=0.0, std=1.0)
                self.U.normal_(mean=0.0, std=1.0)
                self.c.copy_(
                    (
                        torch.randint(
                            0, 2, (m,), device=self.c.device, dtype=torch.long
                        )
                        * 2
                        - 1
                    ).to(self.c.dtype)
                )

    @torch.no_grad()
    def project_(self, rho_c: float, rho_u: float, rho_w: float):
        """
        Project (W, U, c) onto product balls centered at (W0, U0, c0):
          ||W_i - W0_i||_F <= rho_w / sqrt(m)
          ||U_i - U0_i||_2 <= rho_u / sqrt(m)
          |c_i  - c0_i|    <= rho_c / sqrt(m)
        In-place.
        """
        scale = 1.0 / float(self.sqrt_m.item())
        # W
        diff = (self.W - self.W0).reshape(self.m, -1)
        norms = diff.norm(dim=1).clamp(min=1e-12)
        lim = rho_w * scale
        factors = torch.clamp(lim / norms, max=1.0).unsqueeze(1)
        self.W.data.copy_(self.W0 + (diff * factors).view_as(self.W))
        # U
        diff = self.U - self.U0
        norms = diff.norm(dim=1).clamp(min=1e-12)
        lim = rho_u * scale
        factors = torch.clamp(lim / norms, max=1.0).unsqueeze(1)
        self.U.data.copy_(self.U0 + diff * factors)
        # c
        diff = self.c - self.c0
        norms = diff.abs().clamp(min=1e-12)
        lim = rho_c * scale
        factors = torch.clamp(lim / norms, max=1.0)
        self.c.data.copy_(self.c0 + diff * factors)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, d, T) or (d, T). Returns (B,) predictions.
        """
        if X.dim() == 2:
            X = X.unsqueeze(0)  # (1, d, T)
        assert X.shape[1:] == (
            self.d,
            self.T,
        ), f"Expected X shape (B,{self.d},{self.T}), got {tuple(X.shape)}"

        # X_T: last column (B, d)
        X_T = X[:, :, -1]  # (B, d)

        # Compute z_{b,i,t} = x_{b,t}^T W_i X_T^{(b)}
        # First W_i X_T: (m,d,d) x (B,d) -> (B,m,d)
        WX_T = torch.einsum("mdk,bk->bmd", self.W, X_T)
        # Then X^T (B,T,d) times (B,m,d) -> (B,m,T)
        z = torch.einsum("btd,bmd->bmt", X.transpose(1, 2), WX_T)
        # z = z / math.sqrt(self.d)
        if self.mask_last_key:
            z[:, :, -1] = z[:, :, -1] - 1e9
        alpha = F.softmax(z, dim=-1)  # (B, m, T)

        # a_{b,i} = X_b * alpha_{b,i} over T: (B,d,T) x (B,m,T) -> (B,m,d)
        a = torch.einsum("bdt,bmt->bmd", X, alpha)

        # s_{b,i} = U_i^T a_{b,i}: (m,d) x (B,m,d) -> (B,m)
        s = torch.einsum("md,bmd->bm", self.U, a)

        h = self.activation(s)  # (B,m)

        # f_b = (1/sqrt(m)) sum_i c_i * h_{b,i}
        out = torch.einsum("m,bm->b", self.c, h) / self.sqrt_m  # (B,)
        return out


class IndiRNN(nn.Module):
    """
    Independent RNN (IndRNN) with scalar output y = (1/sqrt(m)) * sum_i c_i * h_{T,i}.
    Update:
        h_t = sigma( W_in x_t + u ⊙ h_{t-1} ),   h_0 = 0  (elementwise ⊙)
    Shapes:
        - input X: (B, d, T) or (d, T)
        - W_in: (m, d)
        - u:    (m,)        (diagonal recurrent weights)
        - c:    (m,)        (readout)
    Initialization (symmetric if symmetric_init=True):
        - sample first half i=0..m/2-1:
            W_in[i,:] ~ N(0,1),  u[i] ~ N(0,1),  c[i] ~ Rad(±1)
        - copy to second half:  W_in[i+m/2,:]=W_in[i,:], u[i+m/2]=u[i]
          and flip readout:     c[i+m/2] = -c[i]
        ⇒ For any X, f(X; φ^(0)) = 0 by cancellation.
    Projection:
        project_(rho_c, rho_u, rho_w) enforces per-unit balls of radius ρ/√m
        around the stored center (W0, u0, c0).
    """

    def __init__(
        self,
        d: int,
        T: int,
        m: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
        *,
        symmetric_init: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert m % 2 == 0, "Width m must be even for paired/symmetric init."
        self.d, self.T, self.m = d, T, m
        self.activation = activation

        # Parameters
        self.W_in = nn.Parameter(torch.empty(m, d, device=device, dtype=dtype))
        self.u = nn.Parameter(torch.empty(m, device=device, dtype=dtype))
        self.c = nn.Parameter(torch.empty(m, device=device, dtype=dtype))

        # Buffers to remember the symmetric center (for projection)
        self.register_buffer("W0", torch.empty_like(self.W_in))
        self.register_buffer("u0", torch.empty_like(self.u))
        self.register_buffer("c0", torch.empty_like(self.c))

        # Initialize
        self._reset_parameters(symmetric_init=symmetric_init)

    @torch.no_grad()
    def _reset_parameters(self, symmetric_init: bool = True):
        m = self.m
        half = m // 2

        # First half: random
        self.W_in[:half].normal_(mean=0.0, std=1.0)
        # self.u[:half].normal_(mean=0.0, std=1.0)
        # Rademacher {+1, -1}
        self.u[:half].bernoulli_(0.5).mul_(3.0).sub_(1.5)
        self.c[:half].bernoulli_(0.5).mul_(2.0).sub_(1.0)

        if symmetric_init:
            # Copy weights to second half and flip readout
            self.W_in[half:] = self.W_in[:half]
            self.u[half:] = self.u[:half]
            self.c[half:] = -self.c[:half]
        else:
            # Independent second half
            self.W_in[half:].normal_(mean=0.0, std=1.0)
            self.u[half:].normal_(mean=0.0, std=1.0)
            self.c[half:].bernoulli_(0.5).mul_(2.0).sub_(1.0)

        # Save symmetric center (for projection)
        self.W0.copy_(self.W_in.detach())
        self.u0.copy_(self.u.detach())
        self.c0.copy_(self.c.detach())

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, d, T) or (d, T). Returns y: (B,)
        """
        if X.dim() == 2:
            X = X.unsqueeze(0)  # (1,d,T)
        B, d, T = X.shape
        assert d == self.d and T == self.T, (
            f"Expected input shape (*,{self.d},{self.T}), got {X.shape}"
        )

        # Compute recurrently. h: (B, m)
        h = torch.zeros(B, self.m, device=X.device, dtype=X.dtype)
        # Precompute W_in x_t efficiently: (B,m) each step via einsum
        # For each t: pre = (W_in @ x_t) + u ⊙ h
        for t in range(T):
            x_t = X[:, :, t]  # (B,d)
            pre = torch.einsum("md,bd->bm", self.W_in, x_t)  # (B,m)
            pre = pre + (self.u * h)  # elementwise ⊙
            h = self.activation(pre)

        # Scalar output from final state h_T
        y = (self.c * h).sum(dim=1) / math.sqrt(self.m)  # (B,)
        return y

    @torch.no_grad()
    def project_(self, rho_c: float, rho_u: float, rho_w: float):
        """
        Project parameters into Ω_ρ around the stored center (W0,u0,c0):
          |c_i - c0_i|        ≤ ρ_c / √m,
          |u_i - u0_i|        ≤ ρ_u / √m,
          ||W_in[i]-W0[i]||₂  ≤ ρ_w / √m,   for all i.
        """
        m = self.m
        # scalar limits
        dc_max = float(rho_c) / math.sqrt(m)
        du_max = float(rho_u) / math.sqrt(m)
        dW_max = float(rho_w) / math.sqrt(m)

        # c: per-unit clamp
        dc = self.c - self.c0
        self.c.copy_(self.c0 + dc.clamp(min=-dc_max, max=dc_max))

        # u: per-unit clamp
        du = self.u - self.u0
        self.u.copy_(self.u0 + du.clamp(min=-du_max, max=du_max))

        # W_in: row-wise ℓ2 projection to radius dW_max
        dW = self.W_in - self.W0  # (m,d)
        norms = dW.norm(dim=1, keepdim=True).clamp_min(1e-12)
        scale = torch.clamp(dW_max / norms, max=1.0)  # (m,1)
        self.W_in.copy_(self.W0 + dW * scale)

    @torch.no_grad()
    def in_omega_rho(
        self, rho_c: float, rho_u: float, rho_w: float, tol: float = 1e-8
    ) -> bool:
        """
        Check membership in Ω_ρ (per-unit radii ρ/√m around center).
        """
        m = self.m
        dc_max = float(rho_c) / math.sqrt(m) + tol
        du_max = float(rho_u) / math.sqrt(m) + tol
        dW_max = float(rho_w) / math.sqrt(m) + tol

        ok_c = torch.all((self.c - self.c0).abs() <= dc_max)
        ok_u = torch.all((self.u - self.u0).abs() <= du_max)
        row_norms = (self.W_in - self.W0).norm(dim=1)
        ok_w = torch.all(row_norms <= dW_max)
        return bool(ok_c and ok_u and ok_w)
