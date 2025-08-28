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
    def __init__(self, coeff_net, in_cobasis=None, out_basis=None, out_shift=None, trainable=False):
        """
        Reduced Basis Lifting wrapper.
        Optionally performs reduction and lifting. If any of the bases or shift are None,
        the corresponding operation is skipped.
        """
        super().__init__()
        self.coeff_net = coeff_net

        # Handle optional in_cobasis
        if in_cobasis is not None:
            if trainable:
                self.in_cobasis = nn.Parameter(in_cobasis)
            else:
                self.register_buffer("in_cobasis", in_cobasis)
        else:
            self.in_cobasis = None

        # Handle optional out_basis
        if out_basis is not None:
            if trainable:
                self.out_basis = nn.Parameter(out_basis)
            else:
                self.register_buffer("out_basis", out_basis)
        else:
            self.out_basis = None

        # Handle optional out_shift
        if out_shift is not None:
            if trainable:
                self.out_shift = nn.Parameter(out_shift.reshape(1, -1))
            else:
                self.register_buffer("out_shift", out_shift.reshape(1, -1))
        else:
            self.out_shift = None

    def _reduce(self, x):
        """Project full x (..., n) to reduced coords (..., r) using co-basis; or passthrough if None."""
        if self.in_cobasis is None:
            return x  # passthrough when no reduction is needed

        C = self.in_cobasis
        B = x.reshape(-1, x.shape[-1])  # flatten to (B, n_in)

        if C.is_sparse:
            if C.shape[0] == B.shape[1]:  # (n_in, r)
                xr = torch.sparse.mm(C.t(), B.t()).t()
            elif C.shape[1] == B.shape[1]:  # (r, n_in)
                xr = torch.sparse.mm(C, B.t()).t()
            else:
                raise ValueError("in_cobasis shape must be (n_in, r) or (r, n_in)")
        else:
            if C.shape[0] == B.shape[1]:  # (n_in, r)
                xr = B @ C
            elif C.shape[1] == B.shape[1]:  # (r, n_in)
                xr = B @ C.t()
            else:
                raise ValueError("in_cobasis shape must be (n_in, r) or (r, n_in)")

        return xr.reshape(*x.shape[:-1], -1)

    def forward(self, x):
        # Reduce input if in_cobasis provided
        xr = self._reduce(x)

        # Pass through coefficient network
        c = self.coeff_net(xr)

        # Lift back to full space if out_basis is provided
        if self.out_basis is not None:
            C2 = c.reshape(-1, c.shape[-1])  # (B, r_out)
            if self.out_basis.is_sparse:
                y = torch.sparse.mm(self.out_basis, C2.t()).t()
            else:
                y = C2 @ self.out_basis.t()
        else:
            y = c

        # Apply output shift if provided
        if self.out_shift is not None:
            y = y + self.out_shift

        # Reshape to match original dimensions
        return y.reshape(*c.shape[:-1], y.shape[-1])



        