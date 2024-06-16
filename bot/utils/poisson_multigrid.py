import typing
from dataclasses import dataclass
from enum import Enum, auto
from functools import cache

import numpy as np
import scipy.signal
from skimage.measure import block_reduce


class MultigridMode(Enum):
    VCycle = auto()
    WCycle = auto()
    FCycle = auto()


@dataclass
class MultigridOptions:
    presmooth: int = 2
    postsmooth: int = 1
    gsrb: bool = False
    reduction: int = 2
    mean_reduction: bool = True


SUB_CYCLES = {
    MultigridMode.VCycle: [MultigridMode.VCycle],
    MultigridMode.WCycle: [MultigridMode.WCycle, MultigridMode.WCycle],
    MultigridMode.FCycle: [MultigridMode.FCycle, MultigridMode.VCycle],
}


def prolongate(array, block_size):
    tmp_shape = np.array((array.shape, block_size)).T.flatten()
    tmp = np.empty(tmp_shape, array.dtype)
    indexer = tuple([slice(None), None] * array.ndim)
    tmp[...] = array[indexer]
    out_shape = np.multiply(array.shape, block_size)
    out = tmp.reshape(out_shape)
    return out


def smooth_jacobi(
    x: np.ndarray,
    rhs: np.ndarray,
    b: np.ndarray,
    n: np.ndarray,
    h2: float,
    omega: float = 4 / 5,
) -> np.ndarray:
    s = convolve_jacobi(b * x)
    # s2 = s + (4 - n) * x
    # x2 = (s2 + h2 * rhs) / 4
    x2 = (s + h2 * rhs) / np.maximum(1e-10, n)
    x3 = (1 - omega) * x + omega * x2
    return x3


@cache
def _get_checkerboard(shape):
    return np.indices(shape).sum(axis=0) % 2


JACOBI_KERNEL = np.array(
    [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]
)


def convolve(A: np.ndarray, k: np.ndarray) -> np.ndarray:
    return scipy.ndimage.convolve(A, k)


def convolve_jacobi(A: np.ndarray) -> np.ndarray:
    Ap = np.pad(A, ((1, 1), (1, 1)))
    cp = np.roll(Ap, -1, 0) + np.roll(Ap, +1, 0) + np.roll(Ap, -1, 1) + np.roll(Ap, +1, 1)
    c = cp[1:-1, 1:-1]
    return c


def smooth_gsrb(
    x: np.ndarray,
    rhs: np.ndarray,
    b: np.ndarray,
    n: np.ndarray,
    h2: float,
) -> np.ndarray:
    checkerboard = _get_checkerboard(x.shape)
    for c in (0, 1):
        s = convolve_jacobi(b * x)
        x2 = (s + h2 * rhs) / np.maximum(1e-3, n)
        x = np.where(checkerboard == c, x2, x)
    return x


def mg_opt(
    x: np.ndarray,
    rhs: np.ndarray,
    b: np.ndarray,
    mode: MultigridMode = MultigridMode.VCycle,
    n: np.ndarray | None = None,
    h: float = 1.0,
    options: MultigridOptions | None = None,
) -> np.ndarray:
    options = options or MultigridOptions()

    if any(d < 8 for d in x.shape[:2]):
        return x

    if n is None:
        n = convolve_jacobi(b)

    def smooth(
        x: np.ndarray,
        rhs: np.ndarray,
        b: np.ndarray,
        n: np.ndarray,
        h2: float,
    ) -> np.ndarray:
        if options.gsrb:
            return smooth_gsrb(x, rhs, b, n, h2)
        else:
            return smooth_jacobi(x, rhs, b, n, h2)

    # pre-smooth
    for _ in range(options.presmooth):
        x = smooth(x, rhs, b, n, h * h)

    for sub_mode in SUB_CYCLES[mode]:
        r = residual(x, rhs, b, n, h)
        e = _coarse_grid_correction(r, b, sub_mode, h, options)
        x += e

        for _ in range(options.postsmooth):
            x = smooth(x, rhs, b, n, h * h)

    return x


def residual(
    x: np.ndarray,
    rhs: np.ndarray,
    b: np.ndarray,
    n: typing.Optional[np.ndarray] = None,
    h: float = 1.0,
) -> np.ndarray:
    if n is None:
        n = convolve_jacobi(b)

    s = convolve_jacobi(b * x)
    L = n * x - s
    r = rhs - L / (h * h)
    return r


def _coarse_grid_correction(
    r: np.ndarray,
    b: np.ndarray,
    mode: MultigridMode,
    h: float,
    options: MultigridOptions,
) -> np.ndarray:
    # RESTRICT
    block_sizes = (options.reduction, options.reduction)

    if options.mean_reduction:
        rc = block_reduce(r * b, block_sizes, np.mean)
        bc = block_reduce(b, block_sizes, np.mean)
    else:
        rc = (r * b)[::2, ::2]
        bc = b[::2, ::2]
    nc = convolve_jacobi(bc)

    # RECURSE
    ec = mg_opt(np.zeros_like(rc), rc, bc, mode, nc, options.reduction * h, options)

    # PROLONGATE
    e = prolongate(ec, block_sizes)
    e = e[: r.shape[0], : r.shape[1]]
    return e


def divergence(
    x: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    return convolve_jacobi(b * x) - convolve_jacobi(b) * x
