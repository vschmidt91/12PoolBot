#include <functional>
#include <queue>

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport free, malloc
# data type for the key value
ctypedef cnp.float64_t DTYPE_t
cdef DTYPE_t DTYPE_INF = <DTYPE_t>np.finfo(dtype=np.float64).max
cdef enum ElementState:
   SCANNED     = 1     # popped from the heap
   NOT_IN_HEAP = 2     # never been in the heap
   IN_HEAP     = 3     # in the heap
cdef struct Element:
    ElementState state # element state wrt the heap
    size_t node_idx   # index of the corresponding node in the tree
    DTYPE_t key        # key value
cdef struct PriorityQueue:
    size_t  length    # maximum heap size
    size_t  size      # number of elements in the heap
    size_t* A         # array storing the binary tree
    Element* Elements  # array storing the elements
cdef void init_pqueue(
    PriorityQueue* pqueue,
    size_t length) noexcept nogil:
    """Initialize the priority queue.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * size_t length : length (maximum size) of the heap
    """
    cdef size_t i
    pqueue.length = length
    pqueue.size = 0
    pqueue.A = <size_t*> malloc(length * sizeof(size_t))
    pqueue.Elements = <Element*> malloc(length * sizeof(Element))
    for i in range(length):
        pqueue.A[i] = length
        _initialize_element(pqueue, i)
cdef inline void _initialize_element(
    PriorityQueue* pqueue,
    size_t element_idx) noexcept nogil:
    """Initialize a single element.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * size_t element_idx : index of the element in the element array
    """
    pqueue.Elements[element_idx].key = DTYPE_INF
    pqueue.Elements[element_idx].state = NOT_IN_HEAP
    pqueue.Elements[element_idx].node_idx = pqueue.length
cdef void free_pqueue(
    PriorityQueue* pqueue) noexcept nogil:
    """Free the priority queue.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    """
    free(pqueue.A)
    free(pqueue.Elements)
cdef void insert(
    PriorityQueue* pqueue,
    size_t element_idx,
    DTYPE_t key) noexcept nogil:
    """Insert an element into the priority queue and reorder the heap.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * size_t element_idx : index of the element in the element array
    * DTYPE_t key : key value of the element

    assumptions
    ===========
    * the element pqueue.Elements[element_idx] is not in the heap
    * its new key is smaller than DTYPE_INF
    """
    cdef size_t node_idx = pqueue.size
    pqueue.size += 1
    pqueue.Elements[element_idx].state = IN_HEAP
    pqueue.Elements[element_idx].node_idx = node_idx
    pqueue.A[node_idx] = element_idx
    _decrease_key_from_node_index(pqueue, node_idx, key)
cdef void decrease_key(
    PriorityQueue* pqueue,
    size_t element_idx,
    DTYPE_t key_new) noexcept nogil:
    """Decrease the key of a element in the priority queue, 
    given its element index.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * size_t element_idx : index of the element in the element array
    * DTYPE_t key_new : new value of the element key 

    assumption
    ==========
    * pqueue.Elements[idx] is in the heap
    """
    _decrease_key_from_node_index(
        pqueue,
        pqueue.Elements[element_idx].node_idx,
        key_new)
cdef size_t extract_min(PriorityQueue* pqueue) noexcept nogil:
    """Extract element with min key from the priority queue, 
    and return its element index.

    input
    =====
    * PriorityQueue* pqueue : priority queue

    output
    ======
    * size_t : element index with min key

    assumption
    ==========
    * pqueue.size > 0
    """
    cdef:
        size_t element_idx = pqueue.A[0]  # min element index
        size_t node_idx = pqueue.size - 1  # last leaf node index
    # exchange the root node with the last leaf node
    _exchange_nodes(pqueue, 0, node_idx)
    # remove this element from the heap
    pqueue.Elements[element_idx].state = SCANNED
    pqueue.Elements[element_idx].node_idx = pqueue.length
    pqueue.A[node_idx] = pqueue.length
    pqueue.size -= 1
    # reorder the tree elements from the root node
    _min_heapify(pqueue, 0)
    return element_idx
cdef inline void _exchange_nodes(
    PriorityQueue* pqueue,
    size_t node_i,
    size_t node_j) noexcept nogil:
    """Exchange two nodes in the heap.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * size_t node_i: first node index
    * size_t node_j: second node index
    """
    cdef:
        size_t element_i = pqueue.A[node_i]
        size_t element_j = pqueue.A[node_j]

    # exchange element indices in the heap array
    pqueue.A[node_i] = element_j
    pqueue.A[node_j] = element_i
    # exchange node indices in the element array
    pqueue.Elements[element_j].node_idx = node_i
    pqueue.Elements[element_i].node_idx = node_j

cdef inline void _min_heapify(
    PriorityQueue* pqueue,
    size_t node_idx) noexcept nogil:
    """Re-order sub-tree under a given node (given its node index) 
    until it satisfies the heap property.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * size_t node_idx : node index
    """
    cdef:
        size_t l, r, i = node_idx, s
    while True:
        l =  2 * i + 1
        r = l + 1

        if (
            (l < pqueue.size) and
            (pqueue.Elements[pqueue.A[l]].key < pqueue.Elements[pqueue.A[i]].key)
        ):
            s = l
        else:
            s = i
        if (
            (r < pqueue.size) and
            (pqueue.Elements[pqueue.A[r]].key < pqueue.Elements[pqueue.A[s]].key)
        ):
            s = r
        if s != i:
            _exchange_nodes(pqueue, i, s)
            i = s
        else:
            break

cdef inline void _decrease_key_from_node_index(
    PriorityQueue* pqueue,
    size_t node_idx,
    DTYPE_t key_new) noexcept nogil:
    """Decrease the key of an element in the priority queue, given its tree index.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * size_t node_idx : node index
    * DTYPE_t key_new : new key value

    assumptions
    ===========
    * pqueue.elements[pqueue.A[node_idx]] is in the heap (node_idx < pqueue.size)
    * key_new < pqueue.elements[pqueue.A[node_idx]].key
    """
    cdef:
        size_t i = node_idx, j
        DTYPE_t key_j
    pqueue.Elements[pqueue.A[i]].key = key_new
    while i > 0:
        j = (i - 1) // 2
        key_j = pqueue.Elements[pqueue.A[j]].key
        if key_j > key_new:
            _exchange_nodes(pqueue, i, j)
            i = j
        else:
            break

# Simple example
# ==============
cpdef test_01():
    cdef PriorityQueue pqueue
    init_pqueue(&pqueue, 4)
    insert(&pqueue, 1, 3.0)
    insert(&pqueue, 0, 2.0)
    insert(&pqueue, 3, 4.0)
    insert(&pqueue, 2, 1.0)
    assert pqueue.size == 4
    A_ref = [2, 0, 3, 1]
    n_ref = [1, 3, 0, 2]
    key_ref = [2.0, 3.0, 1.0, 4.0]
    for i in range(4):
        assert pqueue.A[i] == A_ref[i]
        assert pqueue.Elements[i].node_idx == n_ref[i]
        assert pqueue.Elements[i].state == IN_HEAP
        assert pqueue.Elements[i].key == key_ref[i]
    decrease_key(&pqueue, 3, 0.0)
    assert pqueue.size == 4
    A_ref = [3, 0, 2, 1]
    n_ref = [1, 3, 2, 0]
    key_ref = [2.0, 3.0, 1.0, 0.0]
    for i in range(4):
        assert pqueue.A[i] == A_ref[i]
        assert pqueue.Elements[i].node_idx == n_ref[i]
        assert pqueue.Elements[i].state == IN_HEAP
        assert pqueue.Elements[i].key == key_ref[i]

    element_idx = extract_min(&pqueue)
    assert element_idx == 3
    free_pqueue(&pqueue)


cpdef DTYPE_t[:, :] dijkstra_ref(
    DTYPE_t[:, :] cost,
    Py_ssize_t[:, :] targets,
):


    cdef:
        DTYPE_t _sqrt2 = np.sqrt(2)
        Py_ssize_t[8] NEIGHBOURS_X = [-1, 1, 0, 0, -1, 1, -1, 1]
        Py_ssize_t[8] NEIGHBOURS_Y = [0, 0, -1, 1, -1, -1, 1, 1]
        DTYPE_t[8] NEIGHBOURS_DISTANCE = [1, 1, 1, 1, _sqrt2, _sqrt2, _sqrt2, _sqrt2]
        Py_ssize_t width = cost.shape[0]
        Py_ssize_t height = cost.shape[1]
        DTYPE_t[:, :] dist = np.full_like(cost, np.inf)
        Py_ssize_t[:, :] prev_x = np.full_like(cost, -1, np.intp)
        Py_ssize_t[:, :] prev_y = np.full_like(cost, -1, np.intp)
        PriorityQueue q
        Py_ssize_t x, y, idx, x2, y2
        DTYPE_t alternative

    init_pqueue(&q, width * height)
    for i in range(targets.shape[0]):
        x = targets[i, 0]
        y = targets[i, 1]
        insert(&q, x + y * width, 0.0)
        dist[x, y] = 0.0

    while 0 < q.size:
        idx = extract_min(&q)
        x = idx % width
        y = idx // width
        for k in range(8):
            x2 = x + NEIGHBOURS_X[k]
            y2 = y + NEIGHBOURS_Y[k]
            alternative = dist[x, y] + NEIGHBOURS_DISTANCE[k] * cost[x2, y2]
            if alternative < dist[x2, y2]:
                dist[x2, y2] = alternative
                prev_x[x2, y2] = x
                prev_y[x2, y2] = y
                insert(&q, x2 + y2 * width, alternative)

    return dist