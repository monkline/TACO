import warnings
from typing import Literal, Protocol

import torch
from torch import Tensor
from tqdm import trange

from utils import uniform_distributions


class _SupportsInfo(Protocol):
    def info(self, msg: str, /) -> None: ...


class _SupportsWarning(Protocol):
    def warning(self, msg: str, /) -> None: ...


class SinkhornKnopp:
    
    def __init__(
        self,
        *,
        reg1: float, reg2: float = 1, log_alpha: float = 10,
        Hy: Literal['H1', 'H2', 'H3'] = 'H1',
        max_iter: int = 1000, tol: float = 1e-4,
        verbose: bool | _SupportsInfo = False,
        warn: bool | _SupportsWarning = True,
        progress: bool = True,
        use_einsum: bool = False
    ) -> None:
        self.reg1 = reg1
        self.reg2 = reg2
        self.log_alpha = log_alpha
        self.Hy = Hy
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.warn = warn
        self.progress = progress
        self.use_einsum = use_einsum
    
    def _warn(self, msg: str) -> None:
        if isinstance(self.warn, bool):
            if self.warn:
                warnings.warn(msg)
        else:
            self.warn.warning(msg)
    
    def _info(self, msg: str) -> None:
        if isinstance(self.verbose, bool):
            if self.verbose:
                print(msg)
        else:
            self.verbose.info(msg)

    @torch.enable_grad()
    def fit(
        self,
        a: Tensor,
        b: Tensor,
        M: Tensor,
        h: Tensor,
        u: Tensor | None = None,
        v: Tensor | None = None
    ) -> None:
        if a.ndim != 1 or b.ndim != 1:
            raise ValueError('The states must be 1d tensors.')
        if (len(a), len(b)) != M.shape:
            raise ValueError(
                'The shape of the cost matrix does not '
                'match the size of the states.'
            )

        if u is None or v is None:
            u = uniform_distributions(len(a), device=M.device)
            v = uniform_distributions(len(b), device=M.device)

        K = torch.exp(M / (-self.reg1))
        Kp = a.reciprocal().unsqueeze(1) * K

        h.requires_grad_()

        self.errors_ = []
        i = -1
        for i in trange(
            self.max_iter,
            desc='Sinkhorn Knopp Solving',
            disable=not self.progress, leave=False
        ):
            u_prev, v_prev = u, v
            KTu = K.T @ u
            v = b.detach() / KTu
            u = 1 / (Kp @ v)

            if (
                torch.any(KTu == 0)
                or not u.isfinite().all()
                or not v.isfinite().all()
            ):
                # we have reached the machine precision
                # comes back to the previous solution and quit loop
                self._warn(f'numerical errors at iteration {i}')
                
                u, v = u_prev, v_prev
                break

            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if i % 10 == 0:
                if self.use_einsum:
                    tmp = torch.einsum('i,ij,j->j', u, K, v)
                else:
                    tmp = u @ K * v

                err = torch.norm(tmp - b.detach()).item()
                self.errors_.append(err)

                if err < self.tol:
                    break

                msg = []
                if i % 200 == 0:
                    msg.append(f'{"It.":5s}|{"Err":12s}|')
                    msg.append('-' * 19)
                msg.append(f'{i:5d}|{err:8e}|')
                self._info('\n'.join(msg))

            if self.Hy == 'H1':
                g = self.reg1 * torch.log(v.detach())
                bfh = 1000
                for _ in range(10):
                    g[g == h] += 1e-5
                    tmp = (g - h) * self.log_alpha
                    delta = tmp ** 2 + 4 * (self.reg2 ** 2)
                    b = ((tmp + 2 * self.reg2) - delta.sqrt()) / (2 * tmp)

                    fh = b.sum() - 1
                    fh.backward()
                    assert h.grad is not None
                    h.data.sub_(fh.data / h.grad.data)
                    fh = torch.abs(fh).item()

                    if fh < bfh:
                        bfh = fh
                    else:
                        break
            if self.Hy == 'H3':
                g = -self.reg1 * torch.log(v)
                b = b * torch.exp(g / self.reg2)
                b = b / b.sum()
        else:
            self._warn(
                'Sinkhorn did not converge. You might want to '
                'increase the number of iterations `max_iter` '
                'or the regularization parameters `reg1` or `reg2`.'
            )

        self.n_iter_ = i
        self.u_ = u
        self.v_ = v
        self.b_ = b.detach()
        self.transport_matrix_ = u.unsqueeze(1) * K * v.unsqueeze(0)
        self.confidences_, self.labels_ = self.transport_matrix_.max(dim=1)
    
    def fit_predict(
        self,
        a: Tensor,
        b: Tensor,
        M: Tensor,
        h: Tensor,
        u: Tensor | None = None,
        v: Tensor | None = None
    ) -> Tensor:
        self.fit(a, b, M, h, u, v)
        return self.labels_
