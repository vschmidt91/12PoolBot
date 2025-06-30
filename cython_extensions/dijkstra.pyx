from cython import boundscheck, wraparound
import numpy as np
cimport numpy as cnp

from libc.stdlib cimport malloc

DEF HEAP_ARITY = 4

ctypedef cnp.float64_t DTYPE_t

cdef Py_ssize_t[8] NEIGHBOURS_X = [-1, 1, 0, 0, -1, 1, -1, 1]
cdef Py_ssize_t[8] NEIGHBOURS_Y = [0, 0, -1, 1, -1, -1, 1, 1]
cdef DTYPE_t SQRT2 = np.sqrt(2)
cdef DTYPE_t[8] NEIGHBOURS_D = [1, 1, 1, 1, SQRT2, SQRT2, SQRT2, SQRT2]


cdef struct PriorityQueueItem:
    Py_ssize_t x, y
    DTYPE_t distance


cdef class DijkstraOutput:
    cdef public Py_ssize_t[:, :] forward_x
    """Forward pointer grid (x-coordinates)."""
    cdef public Py_ssize_t[:, :] forward_y
    """Forward pointer grid (y-coordinates)."""
    cdef public DTYPE_t[:, :] distance
    """Distance grid."""
    def __cinit__(self,
                  Py_ssize_t[:, :] forward_x,
                  Py_ssize_t[:, :] forward_y,
                  DTYPE_t[:, :] distance):
        self.forward_x = forward_x
        self.forward_y = forward_y
        self.distance = distance

    @boundscheck(False)
    @wraparound(False)
    cpdef get_path(self, (int, int) source, int limit=0):
        """

        Follow the path from a given source using the forward pointer grids.

        Parameters
        ----------
        source :
            Start point.
        limit :
            Maximum length of the returned path. Defaults to 0 indicating no limit.

        Returns
        -------
        list[tuple[int, int]] :
            The lowest cost path from source to any of the targets.

        """
        path = list[tuple[int, int]]()
        x, y = source
        if limit == 0:
            # set a fallback limit to be safe
            # a path longer than this must contain a cycle, so it should never be hit anyway
            limit = self.distance.shape[0] * self.distance.shape[1]

        while len(path) < limit:
            if x < 0 or y < 0:
                break
            path.append((x, y))
            x, y = self.forward_x[x, y], self.forward_y[x, y]
        return path


@boundscheck(False)
@wraparound(False)
cpdef DijkstraOutput cy_dijkstra(
    DTYPE_t[:, :] cost,
    Py_ssize_t[:, :] targets,
    bint checks_enabled = True,
):
    """

    Run Dijkstras algorithm on a grid, yielding many-target-shortest paths for each position.

    Parameters
    ----------
    cost :
        Cost grid. Entries must be positive. Set unpathable cells to infinity.
    targets :
        Target array of shape (*, 2) containing x and y coordinates of the target points.
    checks_enabled :
        Pass False to deactivate grid value and target coordinates checks. Defaults to True.

    Returns
    -------
    DijkstraOutput :
        Pathfinding object containing containing distance and forward pointer grids.

    """

    cdef:
        PriorityQueueItem* pq
        PriorityQueueItem root
        Py_ssize_t i, swap, index, parent

        PriorityQueueItem u
        Py_ssize_t capacity, size
        Py_ssize_t x, y, x2, y2
        DTYPE_t c, alternative
        DTYPE_t[:, :] cost_padded = np.pad(cost, 1, "constant", constant_values=np.inf)
        DTYPE_t[:, :] distance = np.full_like(cost, np.inf)
        Py_ssize_t[:, :] forward_x = np.full_like(cost, -1, np.intp)
        Py_ssize_t[:, :] forward_y = np.full_like(cost, -1, np.intp)

    if checks_enabled:
        if np.any(np.less_equal(cost, 0.0)):
            raise Exception("invalid cost: entries must be strictly positive")

        if any((
            np.less(targets, 0).any(),
            np.greater_equal(targets[:, 0], cost.shape[0]).any(),
            np.greater_equal(targets[:, 1], cost.shape[1]).any(),
        )):
            raise Exception(f"Target out of bounds")

    capacity = cost.size
    heap = <PriorityQueueItem*>malloc(capacity * sizeof(PriorityQueueItem))
    size = targets.shape[0]

    # initialize queue with targets
    for i in range(targets.shape[0]):

        # add to heap
        x = targets[i, 0]
        y = targets[i, 1]
        c = cost[x, y]
        heap[i] = PriorityQueueItem(x, y, c)
        distance[x, y] = c

        # heapify
        index = i
        while index != 0:
            parent = (index - 1) // HEAP_ARITY
            if heap[index].distance < heap[parent].distance:
                heap[index], heap[parent] = heap[parent], heap[index]
                index = parent
            else:
                break

    while size != 0:

        x = heap[0].x
        y = heap[0].y

        # dequeue
        size -= 1
        heap[0] = heap[size]
        index = 0
        while True:
            swap = index
            i = HEAP_ARITY * index + 1
            for child in range(i, i + min(HEAP_ARITY, size - i)):
                if heap[child].distance < heap[swap].distance:
                    swap = child
            if swap != index:
                heap[index], heap[swap] = heap[swap], heap[index]
                index = swap
            else:
                break

        for k in range(8):
            x2 = x + NEIGHBOURS_X[k]
            y2 = y + NEIGHBOURS_Y[k]
            alternative = distance[x, y] + NEIGHBOURS_D[k] * cost_padded[x2+1, y2+1]
            if alternative < distance[x2, y2]:
                distance[x2, y2] = alternative
                forward_x[x2, y2] = x
                forward_y[x2, y2] = y

                # enqueue
                index = size
                size += 1
                heap[index] = PriorityQueueItem(x2, y2, alternative)
                while index != 0:
                    parent = (index - 1) // HEAP_ARITY
                    if heap[index].distance < heap[parent].distance:
                        heap[index], heap[parent] = heap[parent], heap[index]
                        index = parent
                    else:
                        break

    return DijkstraOutput(forward_x, forward_y, distance)