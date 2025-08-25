
import torch
import torch.nn as nn


from .mlp_utils import MLP


class DeepONet(nn.Module):
    """
    Single-branch DeepONet :
      branch(u) -> R^{p[+1]}, trunk(x) -> R^{p[+1]},
      G(u)(x) = <branch(u), trunk(x)>  (inner product over latent dim)
    """
    def __init__(self, m_sensors: int, x_dim: int, p: int = 128, bias: bool = True,
                 branch_hidden=4*[256], trunk_hidden=4*[256], activation=torch.nn.GELU()):
        super().__init__()
        out_dim = p + (1 if bias else 0)
        self.branch = MLP(m_sensors, out_dim, branch_hidden, activation)
        self.trunk  = MLP(x_dim,     out_dim, trunk_hidden,  activation)

    def forward(self, u_samples: torch.Tensor, x_query: torch.Tensor) -> torch.Tensor:
        """
        u_samples: (B, m)    samples of input function at m sensors
        x_query  : (M, d) or (B, M, d) query locations
        returns  : (B, M)
        """
        b = self.branch(u_samples)  # (B, P)
        t = self.trunk(x_query)     # (M, P) or (B, M, P)
        if t.dim() == 2:            # shared grid for all batch items
            return torch.einsum('bp,mp->bm', b, t)
        if t.dim() == 3:            # per-sample grids
            return torch.einsum('bp,bmp->bm', b, t)
        raise ValueError("x_query must be (M,d) or (B,M,d)")



# ---- DeepONet for nodal FE coefficients (single branch) ----
class DeepONetNodal(nn.Module):
    """
    Inputs:  m_samples ∈ R^{B × dM}   (FE nodal coeffs on input mesh)
    Outputs: y         ∈ R^{B × dU}  (FE nodal coeffs on output mesh)
    b(m) = branch(m) ∈ R^{rU [+ 1 if bias]}
    Trunk table T ∈ R^{n_out × rU}; if bias, we append a ones column to T.
    y = b(u) @ T^T
    """
    def __init__(self, dM: int, dU: int, rU: int = 100, bias: bool = True,
                 branch_hidden = 4*[256], activation = torch.nn.GELU(),
                 trunk_init: torch.Tensor | None = None, trainable_trunk: bool = True):
        super().__init__()
        branch_out_dim = rU + (1 if bias else 0)
        self.bias = bias
        self.branch = MLP(dM, branch_out_dim, branch_hidden, activation)

        # trunk features per output node: (n_out, p)
        if trunk_init is None:
            T = torch.empty(dU, rU)
            nn.init.kaiming_uniform_(T, a=5**0.5)
        else:
            assert trunk_init.shape == (n_out, p)
            T = trunk_init
        if trainable_trunk:
            self.T = nn.Parameter(T)
        else:
            self.register_buffer("T", T)

    def forward(self, m_samples: torch.Tensor) -> torch.Tensor:
        B = m_samples.shape[0]
        branch_prediction = self.branch(m_samples)                        # (B, p[+1])
        T = self.T
        if self.bias:
            ones = torch.ones(T.size(0), 1, device=T.device, dtype=T.dtype)
            T = torch.cat([T, ones], dim=1)              # (n_out, p+1)
        return branch_prediction @ T.t()                                  # (B, n_out)



# class MultiBranchDeepONet(nn.Module):
#     """
#     branches:  [u^(i)(sensors_i)]_i  -> features in R^{p[+1]}
#     trunk:     x (coords)             -> basis in R^{p[+1]}
#     output:    G(u)(x) = < fuse_i b_i(u^(i)),  t(x) >
#     """
#     def __init__(self,
#                  branch_input_sizes,          # list[int], sensors per branch
#                  trunk_input_size,            # int, coord dim (e.g., 1,2,3,...)
#                  p=128,
#                  bias=True,                   # add extra channel for bias basis
#                  fuse="concat_linear",        # "concat_linear" or "sum"
#                  hidden_branch=4*[256],
#                  hidden_trunk=4*[256],
#                  activation=torch.nn.GELU()):
#         super().__init__()
#         out_dim = p + (1 if bias else 0)
#         # branches
#         self.branches = nn.ModuleList(
#             [MLP(m, out_dim, hidden_branch, activation) for m in branch_input_sizes]
#         )
#         # optional fusion if concatenating
#         self.fuse = fuse
#         if fuse == "concat_linear":
#             self.fuser = nn.Linear(out_dim * len(self.branches), out_dim)
#         elif fuse != "sum":
#             raise ValueError("fuse must be 'concat_linear' or 'sum'")
#         # trunk
#         self.trunk = MLP(trunk_input_size, out_dim, hidden_trunk, activation)

#     def forward(self, u_list, x_query):
#         """
#         u_list : list of tensors [(B, m_i), ...] matching branch_input_sizes
#         x_query: (M, d) query coords
#         returns: (B, M)
#         """
#         if len(u_list) != len(self.branches):
#             raise ValueError("u_list length must match number of branches")
#         # branch features
#         feats = [b(u) for b, u in zip(self.branches, u_list)]   # each (B, out_dim)
#         if self.fuse == "sum":
#             b = torch.stack(feats, dim=0).sum(0)                # (B, out_dim)
#         else:  # concat + linear fuse
#             b = self.fuser(torch.cat(feats, dim=-1))            # (B, out_dim)
#         # trunk features
#         t = self.trunk(x_query)                                  # (M, out_dim)
#         # inner product over feature dim
#         y = torch.einsum("bp,mp->bm", b, t)                      # (B, M)
#         return y

