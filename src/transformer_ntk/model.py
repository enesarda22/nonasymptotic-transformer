import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerNet(nn.Module):
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
    ):
        super().__init__()
        self.d, self.T, self.m = d, T, m
        self.activation = activation

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
        alpha = F.softmax(z, dim=-1)  # (B, m, T)

        # a_{b,i} = X_b * alpha_{b,i} over T: (B,d,T) x (B,m,T) -> (B,m,d)
        a = torch.einsum("bdt,bmt->bmd", X, alpha)

        # s_{b,i} = U_i^T a_{b,i}: (m,d) x (B,m,d) -> (B,m)
        s = torch.einsum("md,bmd->bm", self.U, a)

        # h = sigma(s)
        h = self.activation(s)  # (B,m)

        # f_b = (1/sqrt(m)) sum_i c_i * h_{b,i}
        out = torch.einsum("m,bm->b", self.c, h) / self.sqrt_m  # (B,)

        return out
