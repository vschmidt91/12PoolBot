from itertools import chain, cycle
from typing import Iterable

import numpy as np
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit

from ..action import Action, Attack, Move, UseAbility
from ..combat_predictor import CombatPrediction
from .component import Component
from ..utils.numerics import normalize, gradient2d


def retreat_target(p: Point2, potential: np.ndarray, pathing: np.ndarray) -> Point2 | None:
    x, y = p.rounded
    gradient = normalize(
        gradient2d(potential, x, y)
    )
    retreat_to = Point2(p + 4 * gradient)
    if pathing[x, y]:
        return retreat_to
    for i in range(10):
        sigma = np.sqrt(1+i)
        p = np.random.normal(loc=p, scale=sigma)
        p = np.clip(p, (0, 0), pathing.shape)
        px, py = p.astype(int)
        if pathing[x, y] and potential[px, py] > potential[x, y]:
            return retreat_to
    return None


class Micro(Component):

    _target_dict: [int, Point2] = dict()

    def micro(self, combat_prediction: CombatPrediction) -> Iterable[Action]:
        return chain(
            self.micro_army(combat_prediction),
            self.micro_queens(),
        )

    def micro_army(self, combat_prediction: CombatPrediction) -> Iterable[Action]:

        start_locations = sorted(self.enemy_start_locations, key=self.is_visible, reverse=True)
        attack_target = next(chain(
            (s.position for s in self.enemy_structures),
            start_locations,
        ))
        pathing = self.game_info.pathing_grid.data_numpy.T

        for unit in self.units({UnitTypeId.ZERGLING, UnitTypeId.MUTALISK}):

            if unit.is_idle:
                self._target_dict.pop(unit.tag, None)
            target = self._target_dict.setdefault(unit.tag, attack_target)

            x, y = unit.position.rounded
            tx, ty = target.rounded
            local_confidence = np.mean((
                combat_prediction.confidence[x, y],
                # combat_prediction.confidence[tx, ty],
            ))

            if local_confidence > -1/2:
                yield Attack(unit, target)
            else:
                self._target_dict.pop(unit.tag, None)
                retreat_to = retreat_target(unit.position, combat_prediction.combat_outcome, pathing)
                if retreat_to is None:
                    retreat_to = retreat_target(unit.position, combat_prediction.civilian_presence, pathing)
                if retreat_to is not None:
                    yield Move(unit, retreat_to)

    def micro_queens(self) -> Iterable[Action]:
        queens = (q for q in self.mediator.get_own_army_dict[UnitTypeId.QUEEN] if q.energy >= 25 and q.is_idle)
        hatcheries = (h for h in self.townhalls if h.is_ready and not h.has_buff(BuffId.QUEENSPAWNLARVATIMER))
        return (UseAbility(queen, AbilityId.EFFECT_INJECTLARVA, hatch) for queen, hatch in zip(queens, hatcheries))

    def random_scout_targets(self) -> Iterable[Point2]:
        a = self.game_info.playable_area
        target = Point2(np.random.uniform((a.x, a.y), (a.right, a.top)))
        if self.in_pathing_grid(target) and not self.is_visible(target):
            yield target
