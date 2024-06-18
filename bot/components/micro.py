from enum import Enum, auto
from itertools import chain
from typing import Iterable

import numpy as np
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from ..action import Action, AttackMove, HoldPosition, Move, UseAbility
from ..combat_predictor import CombatPrediction
from ..utils.dijkstra import Point, shortest_paths_opt
from .component import Component


def _point2_to_point(p: Point2) -> Point:
    x, y = p.rounded
    return int(x), int(y)


class CombatAction(Enum):
    Attack = auto()
    Hold = auto()
    Retreat = auto()


class Micro(Component):
    _action_cache: dict[int, Action] = {}

    def micro(self, combat_prediction: CombatPrediction) -> Iterable[Action]:
        return chain(
            self.micro_army(combat_prediction),
            self.micro_queens(),
        )

    def micro_army(self, combat_prediction: CombatPrediction) -> Iterable[Action]:
        attack_targets = [
            _point2_to_point(u.position)
            for u in chain(combat_prediction.context.combatants, combat_prediction.context.civilians)
            if u.is_enemy and not u.is_flying
        ]
        retreat_targets = [_point2_to_point(w.position) for w in self.workers]

        pathing = self.game_info.pathing_grid.data_numpy.T
        pathing_cost = np.where(pathing == 0, np.inf, 1 + np.maximum(0, -7 * combat_prediction.confidence))
        retreat_pathing = shortest_paths_opt(pathing_cost, retreat_targets, diagonal=True)
        attack_pathing = shortest_paths_opt(pathing_cost, attack_targets, diagonal=True)

        for unit in self.units({UnitTypeId.ZERGLING, UnitTypeId.MUTALISK}):
            p = _point2_to_point(unit.position.rounded)

            attack_path_limit = 8
            retreat_path_limit = 8
            attack_path = attack_pathing.get_path(p, limit=attack_path_limit)

            combat_action: CombatAction
            if 0 <= combat_prediction.confidence[attack_path[-1]]:
                combat_action = CombatAction.Attack
            elif 0 == combat_prediction.presence.enemy_force[p]:
                combat_action = CombatAction.Hold
            else:
                combat_action = CombatAction.Retreat

            action: Action | None = None
            if combat_action == CombatAction.Attack:
                if attack_pathing.dist[p] < np.inf:
                    action = AttackMove(unit, Point2(attack_path[-1]))
                elif unit.is_idle or unit.is_using_ability(AbilityId.HOLDPOSITION):
                    action = AttackMove(unit, self.random_scout_target())
            elif combat_action == CombatAction.Retreat:
                retreat_path = retreat_pathing.get_path(p, limit=retreat_path_limit)
                if retreat_pathing.dist[p] == np.inf:
                    action = Move(unit, self.start_location)
                elif len(retreat_path) < retreat_path_limit:
                    action = AttackMove(unit, Point2(retreat_path[-1]))
                else:
                    action = Move(unit, Point2(retreat_path[-1]))
            else:
                action = HoldPosition(unit)

            is_repeated = action == self._action_cache.get(unit.tag, None)
            if action and not is_repeated:
                self._action_cache[unit.tag] = action
                yield action

    def remove_unit(self, unit_tag: int):
        self._action_cache.pop(unit_tag, None)

    def micro_queens(self) -> Iterable[Action]:
        queens = (q for q in self.mediator.get_own_army_dict[UnitTypeId.QUEEN] if q.energy >= 25 and q.is_idle)
        hatcheries = (h for h in self.townhalls if h.is_ready and not h.has_buff(BuffId.QUEENSPAWNLARVATIMER))
        return (UseAbility(queen, AbilityId.EFFECT_INJECTLARVA, hatch) for queen, hatch in zip(queens, hatcheries))

    def random_scout_target(self, num_attempts=10) -> Point2:
        def sample() -> Point2:
            a = self.game_info.playable_area
            return Point2(np.random.uniform((a.x, a.y), (a.right, a.top)))

        if self.enemy_structures.not_flying.exists:
            return self.enemy_structures.not_flying.random.position
        for p in self.enemy_start_locations:
            if not self.is_visible(p):
                return p
        for _ in range(num_attempts):
            target = sample()
            if self.in_pathing_grid(target) and not self.is_visible(target):
                return target
        return sample()
