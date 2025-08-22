# MIT License
# Copyright (c) 2025
#
# This is part of the dino_tutorial package
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
# For additional questions contact Thomas O'Leary-Roseberry
import torch
from torch import nn


class RBLift(nn.Module):
    def __init__(self, coeff_net, in_cobasis: torch.Tensor,out_basis: torch.Tensor,\
                                         out_shift: torch.Tensor, trainable=False):
        super().__init__()
        self.coeff_net = coeff_net
        if trainable:
            self.in_cobasis = nn.Parameter(in_cobasis)
            self.out_basis = nn.Parameter(out_basis)                         # (n, r)
            self.out_shift = nn.Parameter(out_shift.reshape(1, -1))          # (1, n)
        else:
            self.register_buffer("in_cobasis",in_cobasis)
            self.register_buffer("out_basis", out_basis)                     # (n, r)
            self.register_buffer("out_shift", out_shift.reshape(1, -1))      # (1, n)

    def _reduce(self, x):
        """Project full x (..., n) to reduced coords (..., r) using co-basis; handles (r,n) or (n,r), dense or sparse."""
        C = self.in_cobasis
        B = x.reshape(-1, x.shape[-1])                  # (B, n_in)
        if C.is_sparse:
            if C.shape[0] == B.shape[1]:                # C: (n_in, r)  → xr = x @ C
                xr = torch.sparse.mm(C.t(), B.t()).t()  # (r,n)@(n,B) -> (r,B)^T=(B,r)
            elif C.shape[1] == B.shape[1]:              # C: (r, n_in) → xr = x @ C^T
                xr = torch.sparse.mm(C, B.t()).t()      # (r,n)@(n,B)->(r,B)^T=(B,r)
            else:
                raise ValueError("in_cobasis shape must be (n_in,r) or (r,n_in)")
        else:
            if C.shape[0] == B.shape[1]:                # (n_in, r)
                xr = B @ C                               # (B,n)@(n,r)->(B,r)
            elif C.shape[1] == B.shape[1]:              # (r, n_in)
                xr = B @ C.t()                           # (B,n)@(n,r)->(B,r)
            else:
                raise ValueError("in_cobasis shape must be (n_in,r) or (r,n_in)")
        return xr.reshape(*x.shape[:-1], -1)

    def forward(self, x):
        xr = self._reduce(x)                             # (..., r_in)
        c  = self.coeff_net(xr)                          # (..., r_out)
        C2 = c.reshape(-1, c.shape[-1])                  # (B, r_out)
        if self.out_basis.is_sparse:                     # out_basis: (n_out, r_out)
            y = torch.sparse.mm(self.out_basis, C2.t()).t()  # (n,r)@(r,B)->(n,B)^T=(B,n)
        else:
            y = C2 @ self.out_basis.t()                  # (B,r)@(r,n)->(B,n)
        return (y + self.out_shift).reshape(*c.shape[:-1], self.out_basis.shape[0])