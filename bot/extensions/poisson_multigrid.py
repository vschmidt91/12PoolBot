import enum
import functools
import itertools
import math
import typing

import numpy
import scipy.signal
import skimage.measure
import skimage.transform

class MultigridMode(enum.Enum):
    VCycle = enum.auto()
    WCycle = enum.auto()
    FCycle = enum.auto()


SUB_CYCLES = {
    MultigridMode.VCycle: [MultigridMode.VCycle],
    MultigridMode.WCycle: [MultigridMode.WCycle, MultigridMode.WCycle],
    MultigridMode.FCycle: [MultigridMode.FCycle, MultigridMode.VCycle],
}

JACOBI_KERNEL = numpy.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
])

def prolongate(array, block_size):
    tmp_shape = [
        d
        for p in zip(array.shape, block_size)
        for d in p
    ]
    tmp = numpy.empty(tmp_shape, array.dtype)
    indexer = tuple(
        d
        for _ in range(len(array.shape))
        for d in (slice(None), None)
    )
    tmp[...] = array[indexer]
    out_shape = [numpy.prod(p) for p in zip(array.shape, block_size)]
    out = tmp.reshape(*out_shape)
    return out

def smooth_jacobi(
    x: numpy.ndarray,
    rhs: numpy.ndarray,
    b: numpy.ndarray,
    n: numpy.ndarray,
    h2: float,
    omega: float = 4/5,
) -> numpy.ndarray:

    s = convolve(b * x, JACOBI_KERNEL)
    # s2 = s + (4 - n) * x
    # x2 = (s2 + h2 * rhs) / 4
    x2 = (s + h2 * rhs) / numpy.maximum(1e-10, n)
    x3 = (1 - omega) * x + omega * x2
    return x3

def gsrb_standalone(
    x: numpy.ndarray,
    rhs: numpy.ndarray,
    b: numpy.ndarray,
):
    return smooth_gsrb(x, rhs, b, convolve(b, JACOBI_KERNEL), 1.0)


@functools.cache
def _get_checkerboard(shape):
    return numpy.indices(shape).sum(axis=0) % 2

def convolve(A: numpy.ndarray, k: numpy.ndarray) -> numpy.ndarray:
    while len(k.shape) < len(A.shape):
        k = numpy.expand_dims(k, -1)
    return scipy.ndimage.convolve(A, k, mode="nearest")

def smooth_gsrb(
    x: numpy.ndarray,
    rhs: numpy.ndarray,
    b: numpy.ndarray,
    n: numpy.ndarray,
    h2: float,
) -> numpy.ndarray:
    checkerboard = _get_checkerboard(x.shape)
    for c in (0, 1):
        s = convolve(b * x, JACOBI_KERNEL)
        x2 = (s + h2 * rhs) / numpy.maximum(1e-3, n)
        x = numpy.where(checkerboard == c, x2, x)
    return x


def mg_opt(
    x: numpy.ndarray,
    rhs: numpy.ndarray,
    b: numpy.ndarray,
    mode: MultigridMode = MultigridMode.VCycle,
    n: typing.Optional[numpy.ndarray] = None,
    h: float = 1.0,
) -> numpy.ndarray:

    if any(d < 8 for d in x.shape[:2]):
        return x
    
    if n is None:
        n = convolve(b, JACOBI_KERNEL)

    # pre-smooth
    for _ in range(2):
        x = smooth_gsrb(x, rhs, b, n, h*h)

    for sub_mode in SUB_CYCLES[mode]:
        r = residual(x, rhs, b, n, h)
        e = _coarse_grid_correction(r, b, sub_mode, h)
        x += e

        for _ in range(2):
            x = smooth_gsrb(x, rhs, b, n, h*h)

    return x

def residual(
    x: numpy.ndarray,
    rhs: numpy.ndarray,
    b: numpy.ndarray,
    n: typing.Optional[numpy.ndarray] = None,
    h: float = 1.0,
) -> numpy.ndarray:


    if n is None:
        n = convolve(b, JACOBI_KERNEL)

    s = convolve(b * x, JACOBI_KERNEL)
    L = n * x - s
    r = rhs - L / (h * h)
    return r

def _coarse_grid_correction(
    r: numpy.ndarray,
    b: numpy.ndarray,
    mode: MultigridMode,
    h: float,
) -> numpy.ndarray:

    ### RESTRICTION

    # bilinear
    # rc = restrict(r, b)
    block_sizes = (2, 2) if len(r.shape) == 2 else (2, 2, 1)
    rc = skimage.measure.block_reduce(r * b, block_sizes, numpy.mean)
    bc = skimage.measure.block_reduce(b, block_sizes, numpy.mean)

    # rc = skimage.transform.downscale_local_mean(b * r, (2, 2))
    # bc = skimage.transform.downscale_local_mean(b, (2, 2))

    # sc = (np.asarray(r.shape) + 1) // 2
    # rc = resize(b * r, sc)
    # bc = resize(b, sc)

    nc = convolve(bc, JACOBI_KERNEL)

    # RECURSE
    ec = mg_opt(numpy.zeros_like(rc), rc, bc, mode, nc, 2 * h)

    ### PROLONGATE

    # naive
    e = prolongate(ec, block_sizes)
    # e = skimage.transform.resize_local_mean(ec, np.multiply(2, r.shape))
    # e = skimage.transform.rescale(ec, 2)
    # e = resize(ec, numpy.multiply(r.shape, 2))

    e = e[:r.shape[0],:r.shape[1]]


    # optimized
    # e = cy_prolongate(ec, r.shape)

    # correct
    return e


def divergence(
    x: numpy.ndarray,
    b: numpy.ndarray,
) -> numpy.ndarray:

    return convolve(b * x, JACOBI_KERNEL) - convolve(b, JACOBI_KERNEL) * x
