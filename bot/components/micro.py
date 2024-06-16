from itertools import chain
from typing import Iterable

import numpy as np
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from skimage.measure import block_reduce

from ..action import Action, Attack, Move, UseAbility
from ..combat_predictor import CombatPrediction
from ..utils.dijkstra import DijkstraOutput, Point, shortest_paths_opt
from .component import Component

_OFFSET = Point2((0.5, 0.5))


def _point2_to_point(p: Point2) -> Point:
    x, y = p.rounded
    return int(x), int(y)


class Micro(Component):
    _target_dict: dict[int, Point2] = dict()

    def micro(self, combat_prediction: CombatPrediction) -> Iterable[Action]:
        return chain(
            self.micro_army(combat_prediction),
            self.micro_queens(),
        )

    def micro_army(self, combat_prediction: CombatPrediction) -> Iterable[Action]:
        start_locations = sorted(self.enemy_start_locations, key=self.is_visible, reverse=True)
        attack_target = next(
            chain(
                (s.position for s in self.enemy_structures),
                start_locations,
            )
        )
        pathing = self.game_info.pathing_grid.data_numpy.T
        paths: DijkstraOutput | None = None

        for unit in self.units({UnitTypeId.ZERGLING, UnitTypeId.MUTALISK}):
            if unit.is_idle:
                self._target_dict.pop(unit.tag, None)
            target = self._target_dict.setdefault(unit.tag, attack_target)

            x, y = unit.position.rounded
            local_confidence = combat_prediction.confidence[x, y]

            threshold = -0.5
            reduction = 2
            path_limit = 3

            if local_confidence > threshold:
                yield Attack(unit, target)
            else:
                if paths is None:
                    sources = [_point2_to_point(w.position / reduction) for w in self.workers]
                    cost = np.where(pathing == 0, np.inf, np.exp(-3 * combat_prediction.confidence))
                    cost_reduced = block_reduce(cost, reduction, np.max)
                    paths = shortest_paths_opt(cost_reduced, sources, diagonal=True)

                retreat_path = paths.get_path(_point2_to_point(unit.position / reduction), limit=path_limit)
                if len(retreat_path) < 2:
                    yield Move(unit, self.start_location)
                else:
                    target = Point2(retreat_path[-1]).offset(_OFFSET) * reduction
                    yield Move(unit, target)

    def micro_queens(self) -> Iterable[Action]:
        queens = (q for q in self.mediator.get_own_army_dict[UnitTypeId.QUEEN] if q.energy >= 25 and q.is_idle)
        hatcheries = (h for h in self.townhalls if h.is_ready and not h.has_buff(BuffId.QUEENSPAWNLARVATIMER))
        return (UseAbility(queen, AbilityId.EFFECT_INJECTLARVA, hatch) for queen, hatch in zip(queens, hatcheries))

    def random_scout_targets(self) -> Iterable[Point2]:
        a = self.game_info.playable_area
        target = Point2(np.random.uniform((a.x, a.y), (a.right, a.top)))
        if self.in_pathing_grid(target) and not self.is_visible(target):
            yield target
