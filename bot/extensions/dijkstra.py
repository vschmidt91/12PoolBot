import dataclasses
import functools
import heapq
import math
import typing

import numpy

Point = tuple[int, int]


@functools.total_ordering
@dataclasses.dataclass
class HeapElement:
    position: Point
    distance: float

    def __lt__(self, other):
        return self.distance < other.distance


@dataclasses.dataclass
class DijkstraOutput:
    dist: numpy.ndarray
    prev: numpy.ndarray
    sources: set[Point]

    def get_path(self, target: Point):
        path = []
        u = target
        if self.prev[u] or u in self.sources:
            while u:
                path.append(u)
                u = self.prev[u]
        return path


def _neighbours(
    p: Point,
    shape: Point,
) -> typing.Iterable[Point]:
    x, y = p
    w, h = shape
    if 0 < x:
        yield (x - 1, y)
    if 0 < y:
        yield (x, y - 1)
    if x < w - 1:
        yield (x + 1, y)
    if y < h - 1:
        yield (x, y + 1)


def shortest_paths_opt(
    cost: numpy.ndarray,
    sources: list[Point],
) -> DijkstraOutput:
    dist = numpy.full_like(cost, math.inf)
    prev = numpy.full_like(cost, None, dtype=object)

    ss = numpy.array(sources)
    dist[ss[:, 0], ss[:, 1]] = 0.0
    Q = [HeapElement(s, 0.0) for s in sources]

    while Q:
        elem = heapq.heappop(Q)
        u = elem.position
        du = dist[u]
        if elem.distance != du:
            continue
        for v in _neighbours(u, cost.shape):
            alt = du + cost[v]
            if dist[v] <= alt:
                continue
            dist[v] = alt
            prev[v] = u
            heapq.heappush(Q, HeapElement(v, alt))

    output = DijkstraOutput(dist, prev, set(sources))
    return output
