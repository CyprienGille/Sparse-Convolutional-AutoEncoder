# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 14:56:00 2022


implentation of projections 

@author: YFGI6212
@author: CGille
"""
import torch
from numpy import floor
import numpy as np


def proj_l1ball(w0, eta=1.0, device="cpu"):
    # To help you understand, this function will perform as follow:
    #    a1 = torch.cumsum(torch.sort(torch.abs(y),dim = 0,descending=True)[0],dim=0)
    #    a2 = (a1 - eta)/(torch.arange(start=1,end=y.shape[0]+1))
    #    a3 = torch.abs(y)- torch.max(torch.cat((a2,torch.tensor([0.0]))))
    #    a4 = torch.max(a3,torch.zeros_like(y))
    #    a5 = a4*torch.sign(y)
    #    return a5

    w = torch.as_tensor(w0, dtype=torch.get_default_dtype(), device=device)

    init_shape = w.size()

    if w.dim() > 1:
        init_shape = w.size()
        w = w.reshape(-1)

    Res = torch.sign(w) * torch.max(
        torch.abs(w)
        - torch.max(
            torch.cat(
                (
                    (
                        torch.cumsum(
                            torch.sort(torch.abs(w), dim=0, descending=True)[0],
                            dim=0,
                            dtype=torch.get_default_dtype(),
                        )
                        - eta
                    )
                    / torch.arange(
                        start=1,
                        end=w.numel() + 1,
                        device=device,
                        dtype=torch.get_default_dtype(),
                    ),
                    torch.tensor([0.0], dtype=torch.get_default_dtype(), device=device),
                )
            )
        ),
        torch.zeros_like(w),
    )

    Q = Res.reshape(init_shape).clone().detach()

    if not torch.is_tensor(w0):
        Q = Q.data.numpy()
    return Q


def proj_l11ball(w2, eta=1.0, direction="col", device="cpu"):

    w = torch.as_tensor(w2, dtype=torch.get_default_dtype(), device=device)

    if w.dim() == 1:
        Q = proj_l1ball(w, eta, device=device)
    else:
        if direction == "row":
            w = torch.transpose(w, 0, 1)

        init_shape = w.shape
        Res = torch.empty(init_shape)
        nrow, ncol = init_shape[0:2]

        W = torch.tensor(
            [torch.sum(torch.abs(w[:, i])).data.item() for i in range(ncol)]
        )

        PW = proj_l1ball(W, eta, device=device)

        for i in range(ncol):
            Res[:, i] = proj_l1ball(w[:, i], PW[i].data.item(), device=device)

        if direction == "row":
            Res = torch.transpose(Res, 0, 1)

        Q = Res.clone().detach().requires_grad_(True)

    if not torch.is_tensor(w2):
        Q = Q.data.numpy()
    return Q


def sparse_global(w2, fraction=0.5, device="cpu"):
    w = torch.as_tensor(w2, dtype=torch.get_default_dtype(), device=device)

    init_size = w.size()

    # flatten weights tensor
    if w.dim() > 1:
        w = w.reshape(-1)

    num_weights = w.shape[0]
    # Note: we use numpy's floor to avoid wrapping the number in a tensor just to
    # be able to use torch.floor (which only takes tensors as arguments)
    num_to_prune = floor(num_weights * fraction).astype(int)

    # all weights under the threshold are set to zero
    threshold = torch.sort(w)[0][num_to_prune].item()
    new_weights = torch.where(w >= threshold, w, torch.zeros_like(w))

    # restore original shape
    new_weights = new_weights.reshape(init_size)

    return new_weights


def proj_l21ball(w2, eta, direction="col", device="cpu"):

    w = torch.as_tensor(w2, dtype=torch.get_default_dtype(), device=device)

    if w.dim() == 1:
        Q = proj_l1ball(w, eta, device=device)
    else:
        if direction == "row":
            w = torch.transpose(w, 0, 1)

        init_shape = w.shape
        Res = torch.empty(init_shape)
        nrow, ncol = init_shape[0:2]

        W = torch.tensor(
            [torch.sum(torch.abs(w[:, i])).data.item() for i in range(ncol)]
        )

        PW = proj_l1ball(W, eta, device=device)

        for i in range(ncol):
            Res[:, i] = proj_l2ball(w[:, i], PW[i].data.item(), device=device)

        if direction == "row":
            Res = torch.transpose(Res, 0, 1)
        Q = Res.clone().detach().requires_grad_(True)

    if not torch.is_tensor(w2):
        Q = Q.data.numpy()
    return Q


def proj_l2ball(w0, eta, device="cpu"):
    w = torch.as_tensor(w0, dtype=torch.get_default_dtype(), device=device)

    init_shape = w.shape
    if w.dim() > 1:
        w = w.reshape(-1)

    n = torch.linalg.norm(w, ord=2)
    if n <= eta:
        Res = w
    else:
        Res = torch.mul(eta / n, w)
    Q = Res.reshape(init_shape).clone().detach().requires_grad_(True)
    return Q


def proj_l1inf_numpy(Y, c, tol=1e-4):
    """
    {X : sum_n max_m |X(n,m)| <= c}
    for some given c>0

        Author: Laurent Condat
        Version: 1.0, Sept. 1, 2017
    
    This algorithm is new, to the author's knowledge. It is based
    on the same ideas as for projection onto the l1 ball, see
    L. Condat, "Fast projection onto the simplex and the l1 ball",
    Mathematical Programming, vol. 158, no. 1, pp. 575-585, 2016. 
    
    The algorithm is exact and terminates in finite time*. Its
    average complexity, for Y of size N x M, is O(NM.log(M)). 
    Its worst case complexity, never found in practice, is
    O(NM.log(M) + N^2.M).

    Note : This is a numpy transcription of the original MATLAB code
    *Due to floating point errors, the actual implementation of the algorithm
    uses a tolerance parameter to guarantee halting of the program
    """
    added_dim = False
    if Y.ndim == 1:
        # vector -> matrix
        Y = np.expand_dims(Y, axis=0)
        added_dim = True

    Y = np.transpose(Y)  # to induce sparsity on cols instead of lines

    X = np.flip(np.sort(np.abs(Y), axis=1), axis=1)
    v = np.sum(X[:, 0])
    if v <= c:
        return Y
    N, M = Y.shape
    S = np.cumsum(X, axis=1)
    idx = np.ones((N, 1), dtype=int)
    theta = (v - c) / N
    mu = np.zeros((N, 1))
    active = np.ones((N, 1))
    theta_old = 0
    while np.abs(theta_old - theta) > tol:
        for n in range(N):
            if active[n]:
                j = idx[n]
                while (j < M) and ((S[n, j - 1] - theta) / j) < X[n, j]:
                    j += 1
                idx[n] = j
                mu[n] = S[n, j - 1] / j
                if j == M and (mu[n] - (theta / j)) <= 0:
                    active[n] = 0
                    mu[n] = 0
        theta_old = theta
        theta = (np.sum(mu) - c) / (np.sum(active / idx))
    X = np.minimum(np.abs(Y), (mu - theta / idx) * active)
    X = X * np.sign(Y)

    X = np.transpose(X)

    if added_dim:
        X = np.squeeze(X)

    return X


def proj_l1infball(w0, eta, device="cpu"):
    if torch.is_tensor(w0):
        w = w0.detach().cpu().clone().numpy()
    else:
        w = w0
    init_shape = w.shape
    if len(init_shape) > 2:
        # flatten the insides of the tensor
        w = w.reshape((init_shape[0], np.prod(init_shape[1:])))
    res = proj_l1inf_numpy(w, eta)
    res = res.reshape(init_shape)
    Q = torch.as_tensor(res, dtype=torch.get_default_dtype(), device=device)
    return Q
