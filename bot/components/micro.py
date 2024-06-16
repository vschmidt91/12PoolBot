from itertools import chain
from typing import Iterable

import numpy as np
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from ..action import Action, Attack, Move, UseAbility
from ..combat_predictor import CombatPrediction
from ..utils.dijkstra import Point, shortest_paths_opt
from .component import Component


def _point2_to_point(p: Point2) -> Point:
    x, y = p.rounded
    return int(x), int(y)


class Micro(Component):
    def micro(self, combat_prediction: CombatPrediction) -> Iterable[Action]:
        return chain(
            self.micro_army(combat_prediction),
            self.micro_queens(),
        )

    def micro_army(self, combat_prediction: CombatPrediction) -> Iterable[Action]:
        pathing = self.game_info.pathing_grid.data_numpy.T
        pathing_cost = np.where(pathing == 0, np.inf, np.exp(-3 * combat_prediction.confidence))

        retreat_targets = [_point2_to_point(w.position) for w in self.workers]
        retreat_paths = shortest_paths_opt(pathing_cost, retreat_targets, diagonal=True)

        attack_targets = [_point2_to_point(s.position) for s in self.enemy_units.not_flying]
        attack_paths = shortest_paths_opt(pathing_cost, attack_targets, diagonal=True)

        for unit in self.units({UnitTypeId.ZERGLING, UnitTypeId.MUTALISK}):
            p = _point2_to_point(unit.position.rounded)
            attack_path = attack_paths.get_path(p, limit=5)

            if combat_prediction.confidence[attack_path[-1]] < -0.5:
                if combat_prediction.presence.enemy_force[p] == 0:
                    yield UseAbility(unit, AbilityId.HOLDPOSITION)
                elif retreat_paths.dist[p] == np.inf:
                    yield Move(unit, self.start_location)
                else:
                    retreat_path = retreat_paths.get_path(p, limit=3)
                    yield Move(unit, Point2(retreat_path[-1]))
            else:
                if attack_paths.dist[p] == np.inf:
                    yield Attack(unit, self.enemy_start_locations[0])
                else:
                    attack_target = attack_path[min(2, len(attack_path) - 1)]
                    yield Attack(unit, Point2(attack_target))

    def micro_queens(self) -> Iterable[Action]:
        queens = (q for q in self.mediator.get_own_army_dict[UnitTypeId.QUEEN] if q.energy >= 25 and q.is_idle)
        hatcheries = (h for h in self.townhalls if h.is_ready and not h.has_buff(BuffId.QUEENSPAWNLARVATIMER))
        return (UseAbility(queen, AbilityId.EFFECT_INJECTLARVA, hatch) for queen, hatch in zip(queens, hatcheries))

    def random_scout_targets(self) -> Iterable[Point2]:
        a = self.game_info.playable_area
        target = Point2(np.random.uniform((a.x, a.y), (a.right, a.top)))
        if self.in_pathing_grid(target) and not self.is_visible(target):
            yield target
