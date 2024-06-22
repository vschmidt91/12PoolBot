from enum import Enum, auto
from itertools import chain, cycle
from typing import Iterable

import numpy as np
import scipy
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from ..action import Action, AttackMove, HoldPosition, Move, UseAbility
from ..utils.dijkstra import Point, shortest_paths_opt
from .combat_predictor import CombatPrediction
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
        target_units = combat_prediction.context.enemy_units.not_flying
        attack_targets = [_point2_to_point(u.position) for u in target_units]
        attack_targets.extend(p.rounded for p in self.enemy_start_locations)
        retreat_targets = [_point2_to_point(w.position) for w in self.workers]

        pathing = combat_prediction.context.pathing
        #pathing_cost = np.where(pathing != 1.0, np.inf, 1 + np.log1p(combat_prediction.enemy_presence))
        pathing_cost = np.where(pathing != 1.0, np.inf, 1)
        pathing_cost = self.mediator.get_map_data_object.get_pyastar_grid()
        #pathing_cost += scipy.stats.halfnorm.rvs(0, 1e-3, size=pathing.shape)
        #retreat_pathing = shortest_paths_opt(pathing_cost, retreat_targets)
        #attack_pathing = shortest_paths_opt(pathing_cost, attack_targets, limit=20)

        units = self.units({UnitTypeId.ZERGLING, UnitTypeId.MUTALISK})
        for unit, target, retreat_target in zip(units, cycle(attack_targets), cycle(retreat_targets)):
            p = _point2_to_point(unit.position.rounded)

            attack_path_limit = int(unit.sight_range) - 2
            retreat_path_limit = 3
            #attack_path = attack_pathing.get_path(p, limit=attack_path_limit)
            attack_path = self.mediator.get_map_data_object.pathfind(
                start=unit.position,
                goal=target,
                grid=pathing_cost,
            ) or [p]
            if attack_path_limit < len(attack_path):
                attack_path = attack_path[:attack_path_limit]

            combat_action: CombatAction
            combat_simulation = combat_prediction.confidence[attack_path[-1]]
            if -1/4 <= combat_simulation:
                combat_action = CombatAction.Attack
            elif 0 < combat_prediction.enemy_presence[p]:
                combat_action = CombatAction.Retreat
            else:
                combat_action = CombatAction.Hold

            action: Action | None = None
            if combat_action == CombatAction.Attack:
                if 1 < len(attack_path):
                    action = AttackMove(unit, Point2(attack_path[-1]))
                elif target_units:
                    action = AttackMove(unit, Point2(target))
                elif unit.is_idle:
                    action = AttackMove(unit, self.random_scout_target())
            elif combat_action == CombatAction.Retreat:
                retreat_path = self.mediator.get_map_data_object.pathfind(
                    start=unit.position,
                    goal=retreat_target,
                    grid=pathing_cost,
                ) or [p]
                if retreat_path_limit < len(retreat_path):
                    retreat_path = retreat_path[:retreat_path_limit]
                if 1 == len(retreat_path):
                    action = Move(unit, Point2(retreat_target))
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
