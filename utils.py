import os
import operator
from itertools import repeat, chain
from pathlib import Path
from collections.abc import Iterable, Iterator, Callable
from typing import TypeVar, Generic, overload, Literal, Any, SupportsIndex

import torch
import torch.nn.functional as F
import numpy as np
import sklearn.manifold
import matplotlib.pyplot as plt
from torch import Tensor
from torch.types import Device
from numpy.typing import NDArray, ArrayLike
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.optimize import linear_sum_assignment as hungarian
from accelerate.tracking import GeneralTracker, on_main_process
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from my_types import OneDimSubPlots, TwoDimSubPlots

_T = TypeVar('_T')
_ArrayT = TypeVar('_ArrayT', NDArray, Tensor)


class TensorboardTracker(GeneralTracker):
    name = 'tensorboard'
    requires_logging_directory = True

    @on_main_process
    def __init__(self, run_name: str, logging_dir: str | os.PathLike, **kwargs) -> None:
        super().__init__()
        self.run_name = run_name
        self.logging_dir = os.path.join(logging_dir, run_name)
        self.writer = SummaryWriter(self.logging_dir, **kwargs)
    
    @property
    def tracker(self) -> SummaryWriter:
        return self.writer
    
    @on_main_process
    def add_scalar(self, tag: str, scalar: float, global_step: int | None, **kwargs) -> None:
        self.writer.add_scalar(tag, scalar, global_step, **kwargs)
    
    @on_main_process
    def log(self, values: dict[str, float], step: int) -> None:
        for tag, scalar in values.items():
            self.writer.add_scalar(tag, scalar, step)
    
    @on_main_process
    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: dict[str, float],
        global_step: int | None,
        **kwargs
    ) -> None:
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step, **kwargs)
    
    @on_main_process
    def add_histogram(
        self,
        tag: str,
        values: Tensor | NDArray,
        global_step: int | None,
        bins: str | int = 'tensorflow',
        **kwargs
    ) -> None:
        self.writer.add_histogram(tag, values, global_step, bins, **kwargs)  # type: ignore[call-arg]
    
    @on_main_process
    def add_figure(
        self,
        tag: str,
        figure: Figure,
        global_step: int | None = None,
        close: bool = True,
        **kwargs
    ) -> None:
        self.writer.add_figure(tag, figure, global_step, close, **kwargs)


def outer1d(
    a: Tensor,
    b: Tensor,
    *,
    op: Callable[[Tensor, Tensor], Tensor] = torch.mul
) -> Tensor:
    return op(a.unsqueeze(1), b.unsqueeze(0))


class EMA:

    def __init__(self, momentum: float, initial = None) -> None:
        self.m = momentum
        self.reset(initial)

        self._log_cb = None
    
    def reset(self, value) -> None:
        self.value = value
    
    def set_log_callback(self, cb) -> None:
        self._log_cb = cb
        if self.value is not None:
            self._log_cb({'ema': self.value})
    
    def __call__(self, new) -> Any:
        if self.value is None:
            self.value = new
        else:
            self.value = self.m * self.value + (1. - self.m) * new
        
        if self._log_cb is not None:
            self._log_cb({'orig': new, 'ema': self.value})
        
        return self.value


def exact_batched(iterable: Iterable[_T], size: int) -> Iterator[tuple[_T]]:
    it = iter(iterable)
    args = (it,) * size
    return zip(*args)


def uniform_distributions(*shape: int, device: Device = None) -> Tensor:
    if not shape:
        return torch.ones((), device=device)
    
    return torch.full(shape, 1 / shape[-1], device=device)


def random_distributions(*shape: int, device: Device = None) -> Tensor:
    if not shape:
        return torch.ones((), device=device)
    
    # or
    # concentration = torch.ones(shape[-1], device=device)
    # distribution = torch.distributions.Dirichlet(concentration)
    # return distribution.sample(shape[:-1])

    samples = torch.empty(shape, device=device)
    samples.exponential_()
    samples /= samples.sum(dim=-1, keepdim=True)
    return samples


def category_distribution(labels: _ArrayT, n_classes: int = 0) -> _ArrayT:
    if isinstance(labels, Tensor):
        counts = labels.bincount(minlength=n_classes)
    else:
        # np.bincount can handle array like objects
        counts = np.bincount(labels, minlength=n_classes)
    
    return counts / len(labels)


def bingroupby(a: _ArrayT) -> list[_ArrayT]:
    if isinstance(a, np.ndarray):
        indices = a.argsort(stable=True)
        counts = np.bincount(a)
        return np.split(indices, counts.cumsum()[:-1])
    
    if isinstance(a, Tensor):
        indices = a.argsort(stable=True)
        counts = a.bincount()
        return list(torch.split_with_sizes(indices, counts.tolist()))
    
    raise TypeError


def matching_matrix(y_true: NDArray, y_pred: NDArray, n_classes: int) -> NDArray:
    mat = np.zeros((n_classes, n_classes), np.intp)
    indices = (y_true, y_pred)
    # 1d is faster than 2d
    np.add.at(np.ravel(mat), np.ravel_multi_index(indices, mat.shape), 1)
    return mat


def reassign_table(y_pred: NDArray, y_true: NDArray, n_classes: int) -> NDArray[np.intp]:
    mat = matching_matrix(y_true, y_pred, n_classes)
    _, assign = hungarian(mat.T, maximize=True)
    return assign


def reassign_labels(y_pred: NDArray, y_true: NDArray, n_classes: int) -> NDArray:
    return reassign_table(y_pred, y_true, n_classes)[y_pred]


def save_tensors(
    tensors: Iterable[tuple[str, Tensor]],
    step: int,
    logging_root: str | os.PathLike[str],
    tensor_dirname: str = 'tensors',
    ext: str = '.pth'
) -> None:
    tensor_root = Path(logging_root) / tensor_dirname
    for name, tensor in tensors:
        output_dir = tensor_root / name
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, output_dir / step_path(step, ext))


@overload
def load_tensors(
    names: str,
    step: int,
    logging_root: str | os.PathLike[str],
    tensor_dirname: str = 'tensors',
    ext: str = '.pth',
    *,
    asarray: Literal[False] = ...
) -> Tensor:
    ...

@overload
def load_tensors(
    names: str,
    step: int,
    logging_root: str | os.PathLike[str],
    tensor_dirname: str = 'tensors',
    ext: str = '.pth',
    *,
    asarray: Literal[True]
) -> NDArray:
    ...

@overload
def load_tensors(
    names: Iterable[str],
    step: int,
    logging_root: str | os.PathLike[str],
    tensor_dirname: str = 'tensors',
    ext: str = '.pth',
    *,
    asarray: Literal[False] = ...
) -> list[Tensor]:
    ...

@overload
def load_tensors(
    names: Iterable[str],
    step: int,
    logging_root: str | os.PathLike[str],
    tensor_dirname: str = 'tensors',
    ext: str = '.pth',
    *,
    asarray: Literal[True]
) -> list[NDArray]:
    ...

def load_tensors(
    names: str | Iterable[str],
    step: int,
    logging_root: str | os.PathLike[str],
    tensor_dirname: str = 'tensors',
    ext: str = '.pth',
    *,
    asarray: bool = False,
) -> list[Tensor] | list[NDArray] | Tensor | NDArray:
    tensor_root = Path(logging_root) / tensor_dirname

    if isinstance(names, str):
        out: Tensor = torch.load(tensor_root / names / step_path(step, ext))
        return out.numpy(force=True) if asarray else out

    outputs: list[Tensor] = [
        torch.load(tensor_root / name / step_path(step, ext))
        for name in names
    ]

    return [tensor.numpy(force=True) for tensor in outputs] if asarray else outputs


class describing:

    def __init__(self, tbar: tqdm, desc: str, refresh: bool = True) -> None:
        self.tbar = tbar
        self.desc = desc
        self.refresh = refresh
        self.old_desc: str | None = None
    
    def __enter__(self) -> None:
        self.old_desc = self.tbar.desc
        self.tbar.set_description(self.desc, self.refresh)
    
    def __exit__(self, *_) -> None:
        self.tbar.set_description_str(self.old_desc, self.refresh)


def reduce_to_2d(embeddings: Tensor | NDArray, seed: int | None = None, verbose: int = 0) -> NDArray:
    tsne = sklearn.manifold.TSNE(
        n_components=2,
        init='random',
        method='barnes_hut',
        random_state=seed,
        verbose=verbose,
    )

    if isinstance(embeddings, Tensor):
        embeddings = embeddings.numpy(force=True)
    
    return tsne.fit_transform(embeddings)


def plot_embeddings2d(
    X: NDArray,
    c: NDArray | str,
    cmap: str | None = None,
    ax: Axes | None = None,
    s: float = 4,
    alpha: float = 0.98,
    figsize: tuple[int, int] = (8, 8),
    **scatter_kwargs,
) -> Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        from matplotlib.figure import SubFigure
        if isinstance(fig, SubFigure):
            fig = fig.figure
    
    if cmap is None and isinstance(c, np.ndarray):
        cmap = 'Spectral'

    ax.scatter(X[:, 0], X[:, 1], c=c, cmap=cmap, s=s, alpha=alpha, **scatter_kwargs)
    ax.axis('off')

    return fig


def get_opt_steps(n_optimizations: int, start_step: int, max_steps: int) -> list[int]:
    optimize_steps = (
        torch.linspace(0, 1, n_optimizations + 1)
        .square()
        .mul(max_steps - start_step)
        .add(start_step)
        .tolist()
    )
    optimize_steps.reverse()

    result = []
    for step in range(start_step, max_steps):
        if step >= optimize_steps[-1]:
            del optimize_steps[-1]
            result.append(step)
    
    return result


def step_path(step: int, ext: str = '.pth') -> str:
    return f'step_{step}{ext}'


def figsize(
    nrows: int,
    ncols: int,
    subheight: int,
    subwidth: int,
    row_spacing: int = 1,
    col_spacing: int = 1
) -> tuple[int, int]:
    height = nrows * (subheight + row_spacing) - row_spacing
    width = ncols * (subwidth + col_spacing) - col_spacing
    return (width, height)


@overload
def subplots_with_size(
    nrows: Literal[1],
    ncols: Literal[1],
    subheight: int,
    subwidth: int,
    row_spacing: int = ...,
    col_spacing: int = ...
) -> tuple[Figure, Axes]: ...

@overload
def subplots_with_size(
    nrows: Literal[1],
    ncols: int,
    subheight: int,
    subwidth: int,
    row_spacing: int = ...,
    col_spacing: int = ...
) -> tuple[Figure, OneDimSubPlots]: ...

@overload
def subplots_with_size(
    nrows: int,
    ncols: Literal[1],
    subheight: int,
    subwidth: int,
    row_spacing: int = ...,
    col_spacing: int = ...
) -> tuple[Figure, OneDimSubPlots]: ...

@overload
def subplots_with_size(
    nrows: int,
    ncols: int,
    subheight: int,
    subwidth: int,
    row_spacing: int = ...,
    col_spacing: int = ...
) -> tuple[Figure, TwoDimSubPlots]: ...

def subplots_with_size(
    nrows: int,
    ncols: int,
    subheight: int,
    subwidth: int,
    row_spacing: int = 1,
    col_spacing: int = 1
) -> tuple[Figure, Axes | OneDimSubPlots | TwoDimSubPlots]:
    return plt.subplots(
        nrows,
        ncols,
        figsize=figsize(
            nrows, ncols,
            subheight, subwidth,
            row_spacing, col_spacing
        )
    )
