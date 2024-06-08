from ares import AresBot

from itertools import chain, cycle
from typing import Iterable

import numpy as np
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit

from actions import Action, Attack, GatherResources, UseAbility
from strategy import Strategy


class Micro(AresBot):

    def micro(self, strategy: Strategy) -> Iterable[Action]:
        return chain(
            self.micro_workers(strategy),
            self.micro_army(),
            self.micro_queens(),
        )

    def micro_workers(self, strategy: Strategy) -> Iterable[Action]:
        workers_per_gas = 3 if strategy.gather_vespene else 0
        yield GatherResources(workers_per_gas)

    def micro_army(self) -> Iterable[Action]:
        invisible_enemy_start_locations = [p for p in self.enemy_start_locations if not self.is_visible(p)]
        targets = chain(
            (s.position for s in self.enemy_structures),
            invisible_enemy_start_locations,
            self.random_scout_targets(),
        )

        def micro_unit(unit: Unit, target: Point2) -> Iterable[Action]:
            if unit.is_idle:
                yield Attack(unit, target)
        army = self.units({UnitTypeId.ZERGLING, UnitTypeId.MUTALISK})
        return chain.from_iterable(
            micro_unit(u, t)
            for u, t in zip(army, cycle(targets))
        )

    def micro_queens(self) -> Iterable[Action]:
        queens = (q for q in self.mediator.get_own_army_dict[UnitTypeId.QUEEN] if q.energy >= 25 and q.is_idle)
        hatcheries = (h for h in self.townhalls if h.is_ready and not h.has_buff(BuffId.QUEENSPAWNLARVATIMER))
        return (
            UseAbility(queen, AbilityId.EFFECT_INJECTLARVA, hatch)
            for queen, hatch in zip(queens, hatcheries)
        )

    def random_scout_targets(self) -> Iterable[Point2]:
        a = self.game_info.playable_area
        target = Point2(np.random.uniform((a.x, a.y), (a.right, a.top)))
        if self.in_pathing_grid(target) and not self.is_visible(target):
            yield target
