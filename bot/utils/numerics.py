import functools

import numpy
import scipy.ndimage
import sklearn.preprocessing

JACOBI_KERNEL = [
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
]


def maxabs(x: numpy.ndarray) -> numpy.ndarray:
    return sklearn.preprocessing.maxabs_scale(x.reshape((-1, 1))).reshape(x.shape)


def minmax(x: numpy.ndarray) -> numpy.ndarray:
    return sklearn.preprocessing.minmax_scale(x.reshape((-1, 1))).reshape(x.shape)


def normalize(v):
    return v / max(1e-10, numpy.linalg.norm(v))


def gradient2d(f: numpy.ndarray, x: int, y: int) -> numpy.ndarray:
    df_dx = (f[min(x + 1, f.shape[0] - 1), y] - f[max(x - 1, 0), y]) / 2
    df_dy = (f[x, min(y + 1, f.shape[1] - 1)] - f[x, max(y - 1, 0)]) / 2
    grad_f = df_dx, df_dy
    return numpy.array(grad_f)


@functools.cache
def checkerboard(shape):
    return numpy.indices(shape).sum(axis=0) % 2


def convolve(A, k):
    return scipy.ndimage.convolve(A, k, mode="nearest")


def smooth_gsrb(
    x: numpy.ndarray,
    rhs: numpy.ndarray,
    # omega: float = 5 / 3,
    omega: float = 1.0,
) -> numpy.ndarray:
    mask = checkerboard(x.shape)
    for c in (0, 1):
        s = convolve(x, JACOBI_KERNEL)
        x2 = (s + rhs) / 4
        x3 = (1 - omega) * x + omega * x2
        x = numpy.where(mask == c, x3, x)
    return x
