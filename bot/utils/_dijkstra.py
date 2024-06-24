import heapq
import math
from dataclasses import dataclass
from functools import total_ordering

import numpy as np

Point = tuple[int, int]

_DIAGONAL_WEIGHT = math.sqrt(2)


@total_ordering
@dataclass
class HeapElement:
    position: Point
    distance: float

    def __lt__(self, other: "HeapElement"):
        return self.distance < other.distance


@dataclass
class DijkstraOutput:
    dist: np.ndarray
    prev: dict[Point, Point]
    sources: set[Point]

    def get_path(self, target: Point, limit: float = math.inf):
        path: list[Point] = []
        u: Point | None = target
        while u and len(path) < limit:
            path.append(u)
            u = self.prev.get(u)
        return path


def _neighbours(x: int, y: int) -> list[Point]:
    return [
        (x - 1, y),
        (x, y - 1),
        (x + 1, y),
        (x, y + 1),
    ]


def _neighbours_diagonal(x: int, y: int) -> list[Point]:
    return [
        (x - 1, y - 1),
        (x - 1, y + 1),
        (x + 1, y - 1),
        (x + 1, y + 1),
    ]

NEIGHBOURS = [
    (-1, 0),
    (+1, 0),
    (0, -1),
    (0, +1),
]


def shortest_paths_opt(
    cost: np.ndarray, sources: list[Point],
) -> DijkstraOutput:
    dist = np.full_like(cost, math.inf, dtype=float)
    prev = dict()

    Q: list[HeapElement] = []
    for s in sources:
        dist[s] = 0.0
        Q.append(HeapElement(s, 0.0))

    while Q:
        u = x, y = heapq.heappop(Q).position
        for dx, dy in NEIGHBOURS:
            v = x + dx, y + dy
            alt = dist[u] + cost[v]
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(Q, HeapElement(v, alt))

    output = DijkstraOutput(dist, prev, set(sources))
    return output
