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
from .strategy import StrategyDecision
from ..utils.numerics import normalize, gradient2d


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

        for unit in self.units({UnitTypeId.ZERGLING, UnitTypeId.MUTALISK}):

            if unit.is_idle:
                self._target_dict.pop(unit.tag, None)
            target = self._target_dict.setdefault(unit.tag, attack_target)

            x, y = unit.position.rounded
            tx, ty = target.rounded
            local_confidence = np.mean((
                combat_prediction.confidence[x, y],
                combat_prediction.confidence[tx, ty],
            ))

            if local_confidence > -1/2:
                yield Attack(unit, target)
            else:
                self._target_dict.pop(unit.tag, None)
                retreat = normalize(
                    gradient2d(combat_prediction.retreat_potential, x, y)
                )
                yield Move(unit, Point2(unit.position + 4 * retreat))

    def micro_queens(self) -> Iterable[Action]:
        queens = (q for q in self.mediator.get_own_army_dict[UnitTypeId.QUEEN] if q.energy >= 25 and q.is_idle)
        hatcheries = (h for h in self.townhalls if h.is_ready and not h.has_buff(BuffId.QUEENSPAWNLARVATIMER))
        return (UseAbility(queen, AbilityId.EFFECT_INJECTLARVA, hatch) for queen, hatch in zip(queens, hatcheries))

    def random_scout_targets(self) -> Iterable[Point2]:
        a = self.game_info.playable_area
        target = Point2(np.random.uniform((a.x, a.y), (a.right, a.top)))
        if self.in_pathing_grid(target) and not self.is_visible(target):
            yield target
