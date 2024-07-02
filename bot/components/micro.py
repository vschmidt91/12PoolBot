import math
from dataclasses import dataclass
from enum import Enum, auto
from itertools import chain, cycle
from typing import Iterable

import numpy as np
from ares.consts import DEBUG
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from ..action import Action, AttackMove, HoldPosition, Move, UseAbility
from ..combat_predictor import CombatPrediction
from ..utils.cy_dijkstra import cy_dijkstra  # type: ignore
from .component import Component

Point = tuple[int, int]
HALF = Point2((0.5, 0.5))


class CombatAction(Enum):
    Attack = auto()
    Hold = auto()
    Retreat = auto()


@dataclass
class DijkstraOutput:
    prev_x: np.ndarray
    prev_y: np.ndarray
    dist: np.ndarray

    @classmethod
    def from_cy(cls, o) -> "DijkstraOutput":
        return DijkstraOutput(
            np.asarray(o.prev_x).astype(int),
            np.asarray(o.prev_y).astype(int),
            np.asarray(o.dist).astype(float),
        )

    def get_path(self, target: Point, limit: float = math.inf):
        path: list[Point] = []
        x, y = target
        while len(path) < limit:
            path.append((x, y))
            x2 = self.prev_x[x, y]
            y2 = self.prev_y[x, y]
            if x2 < 0:
                break
            x, y = x2, y2
        return path


class Micro(Component):
    _action_cache: dict[int, Action] = {}

    def micro(self, combat_prediction: CombatPrediction) -> Iterable[Action]:
        return chain(
            self.micro_army(combat_prediction),
            self.micro_queens(),
        )

    def micro_army(self, combat_prediction: CombatPrediction) -> Iterable[Action]:
        target_units = combat_prediction.context.enemy_units.not_flying
        attack_targets = self.enemy_start_locations + [u.position for u in target_units]
        attack_center = Point2(np.median(np.array(attack_targets), axis=0))
        attack_targets.sort(key=lambda t: t.distance_to(attack_center), reverse=True)

        retreat_targets = [self.start_location] + [w.position for w in self.workers]
        retreat_center = Point2(np.median(np.array(retreat_targets), axis=0))
        retreat_targets.sort(key=lambda t: t.distance_to(retreat_center))

        pathing = np.where(combat_prediction.context.pathing == 1, 1.0, np.inf)
        pathing_cost = pathing + combat_prediction.enemy_presence.dps

        attack_pathing = DijkstraOutput.from_cy(
            cy_dijkstra(
                pathing_cost.astype(np.float64),
                np.array(attack_targets, dtype=np.intp),
            )
        )
        retreat_pathing = DijkstraOutput.from_cy(
            cy_dijkstra(
                pathing_cost.astype(np.float64),
                np.array(retreat_targets, dtype=np.intp),
            )
        )

        if self.config[DEBUG]:
            self.mediator.get_map_data_object.draw_influence_in_game(pathing_cost)

        units = sorted(self.units({UnitTypeId.ZERGLING, UnitTypeId.MUTALISK}), key=lambda u: u.tag)
        for unit, target, retreat_target in zip(units, cycle(attack_targets), cycle(retreat_targets)):
            p = unit.position.rounded
            attack_path_limit = 5
            retreat_path_limit = 3

            attack_path = attack_pathing.get_path(p, attack_path_limit)

            combat_action: CombatAction
            combat_simulation = combat_prediction.confidence(Point2(attack_path[-1]))
            if np.isnan(combat_simulation):
                combat_action = CombatAction.Attack
            elif 0 <= combat_simulation:
                combat_action = CombatAction.Attack
            elif 0 < combat_prediction.enemy_presence.dps[p]:
                combat_action = CombatAction.Retreat
            else:
                combat_action = CombatAction.Hold

            action: Action | None = None
            if combat_action == CombatAction.Attack:
                if attack_pathing.dist[p] < np.inf:
                    action = AttackMove(unit, Point2(attack_path[-1]).offset(HALF))
                elif target_units:
                    action = AttackMove(unit, Point2(target))
                elif unit.is_idle:
                    action = AttackMove(unit, self.random_scout_target())
            elif combat_action == CombatAction.Retreat:
                retreat_path = retreat_pathing.get_path(p, retreat_path_limit)
                if retreat_pathing.dist[p] == np.inf:
                    action = Move(unit, Point2(retreat_target))
                elif len(retreat_path) < retreat_path_limit:
                    action = AttackMove(unit, Point2(retreat_path[-1]).offset(HALF))
                else:
                    action = Move(unit, Point2(retreat_path[-1]).offset(HALF))
            else:
                action = HoldPosition(unit)

            is_repeated = action == self._action_cache.get(unit.tag, None)
            if action and not is_repeated:
                self._action_cache[unit.tag] = action
                yield action

    def remove_unit(self, unit_tag: int):
        self._action_cache.pop(unit_tag, None)

    def micro_queens(self) -> Iterable[Action]:
        queens = sorted(self.mediator.get_own_army_dict[UnitTypeId.QUEEN], key=lambda u: u.tag)
        hatcheries = sorted(self.townhalls.ready, key=lambda u: u.tag)
        for queen, hatchery in zip(queens, hatcheries):
            if 25 <= queen.energy:
                yield UseAbility(queen, AbilityId.EFFECT_INJECTLARVA, hatchery)
            else:
                yield AttackMove(queen, hatchery.position)

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
