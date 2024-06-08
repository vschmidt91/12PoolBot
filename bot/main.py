from ares import AresBot
from ares.behaviors.macro import AutoSupply, Mining, SpawnController
from ares.consts import UnitRole

import random
from typing import Optional

import numpy as np
from sc2.bot_ai import Result
from sc2.dicts.unit_trained_from import UNIT_TRAINED_FROM
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit, UnitCommand


def larva_from(hatchery: Unit) -> float:
    larva = 0.0
    if hatchery.is_ready:
        larva += 1 / 11
        if hatchery.has_buff(BuffId.QUEENSPAWNLARVATIMER):
            larva += 3 / 29
    return larva


class TwelvePoolBot(AresBot):

    def __init__(self, game_step_override: Optional[int] = None):
        super().__init__(game_step_override)

    """
    === <OVERRIDES>
    """

    async def on_step(self, iteration: int) -> None:
        await super().on_step(iteration)
        if not self.townhalls:
            await self.surrender()
        elif not self.build_order_runner.build_completed:
            self.register_behavior(Mining())
        else:
            await self.macro()
            self.micro()

    async def on_start(self) -> None:
        await super().on_start()

    async def on_end(self, game_result: Result) -> None:
        await super().on_end(game_result)

    async def on_building_construction_complete(self, unit: Unit) -> None:
        await super().on_building_construction_complete(unit)

    async def on_unit_created(self, unit: Unit) -> None:
        await super().on_unit_created(unit)
        if unit.type_id in {UnitTypeId.ZERGLING, UnitTypeId.MUTALISK}:
            self.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)

    async def on_unit_destroyed(self, unit_tag: int) -> None:
        await super().on_unit_destroyed(unit_tag)

    async def on_unit_took_damage(self, unit: Unit, amount_damage_taken: float) -> None:
        await super().on_unit_took_damage(unit, amount_damage_taken)

    """
    === </OVERRIDES>
    """

    async def surrender(self) -> None:
        await self.client.chat_send('(gg)', False)
        await self.client.quit()

    async def macro(self) -> Optional[UnitCommand]:
        go_mutas = self.enemy_structures.flying and not self.enemy_structures.not_flying
        research_speed = self.vespene < 96 and not self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED)
        gas_harvester_target = 3 if research_speed or go_mutas else 0
        army_composition = self.get_army_composition(go_mutas)
        self.register_behavior(Mining(workers_per_gas=gas_harvester_target))
        self.register_behavior(AutoSupply(base_location=self.start_location))
        return (
            await self.make_tech(go_mutas)
            or self.get_upgrades()
            or self.expand()
            or self.train_queen()
            or self.train_army(army_composition)
        )

    def expand(self) -> Optional[UnitCommand]:
        if (
            self.can_afford(UnitTypeId.HATCHERY)
            and not self.already_pending(UnitTypeId.HATCHERY)
            and (builder := next((u for u in self.workers.collecting), None))
            and (target := self.get_next_expansion())
        ):
            self.mediator.assign_role(tag=builder.tag, role=UnitRole.PERSISTENT_BUILDER)
            return builder.build(UnitTypeId.HATCHERY, target)

    async def make_tech(self, make_mutas: bool) -> Optional[UnitCommand]:
        if self.tech_requirement_progress(UnitTypeId.ZERGLING) < 1:
            return await self.ensure_structure_exists(UnitTypeId.SPAWNINGPOOL)
        elif not make_mutas:
            return None
        elif self.tech_requirement_progress(UnitTypeId.SPIRE) < 1:
            return self.ensure_lair_exists()
        else:
            return await self.ensure_structure_exists(UnitTypeId.SPIRE)

    def train_queen(self) -> Optional[UnitCommand]:
        if (
            self.can_afford(UnitTypeId.QUEEN)
            and not self.already_pending(UnitTypeId.QUEEN)
            and len(self.mediator.get_own_army_dict[UnitTypeId.QUEEN]) < self.townhalls.amount
            and 2 <= self.supply_left
            and (hatch := next((t for t in self.townhalls if t.is_idle), None))
        ):
            return hatch.train(UnitTypeId.QUEEN)

    def get_upgrades(self) -> Optional[UnitCommand]:
        if (
            self.can_afford(UpgradeId.ZERGLINGMOVEMENTSPEED)
            and not self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED)
            and (
                pool := next(
                    (p for p in self.mediator.get_own_structures_dict[UnitTypeId.SPAWNINGPOOL] if p.is_idle), None
                )
            )
        ):
            return pool.research(UpgradeId.ZERGLINGMOVEMENTSPEED)

    async def ensure_structure_exists(self, type_id: UnitTypeId) -> Optional[UnitCommand]:
        placement_near = self.start_location.towards(self.game_info.map_center, 8)
        trainer_type_id = min(UNIT_TRAINED_FROM[type_id], key=lambda v: v.value)
        if (
            self.can_afford(type_id)
            and not self.already_pending(type_id)
            and (trainer := next((t for t in self.mediator.get_own_army_dict[trainer_type_id]), None))
            and (placement := await self.find_placement(type_id, near=placement_near))
        ):
            self.mediator.assign_role(tag=trainer.tag, role=UnitRole.PERSISTENT_BUILDER)
            return trainer.build(type_id, placement)

    def ensure_lair_exists(self) -> Optional[UnitCommand]:
        if (
            self.can_afford(UnitTypeId.LAIR)
            and not self.already_pending(UnitTypeId.LAIR)
            and (
                hatch := next(
                    (t for t in self.mediator.get_own_structures_dict[UnitTypeId.HATCHERY] if t.is_idle), None
                )
            )
        ):
            return hatch(AbilityId.UPGRADETOLAIR_LAIR)

    def should_drone(self) -> bool:
        larva_per_second = sum(larva_from(h) for h in self.townhalls)
        minerals_for_lings = 50 * 60 * larva_per_second  # maximum we can possibly spend on lings
        max_spending = minerals_for_lings  # aim for a 20% surplus
        should_drone = (
            self.minerals < 150
            and self.state.score.collection_rate_minerals < 1.2 * max_spending  # aim for a 20% surplus
            and self.state.score.food_used_economy < sum(h.ideal_harvesters for h in self.townhalls)
            and not self.already_pending(UnitTypeId.DRONE)
        )
        return should_drone

    def get_army_composition(self, make_mutas: bool) -> dict:
        if self.should_drone():
            return {
                UnitTypeId.DRONE: {"proportion": 1.0, "priority": 0},
            }
        elif make_mutas:
            return {
                UnitTypeId.MUTALISK: {"proportion": 0.5, "priority": 0},
                UnitTypeId.ZERGLING: {"proportion": 0.5, "priority": 0},
            }
        else:
            return {
                UnitTypeId.ZERGLING: {"proportion": 1.0, "priority": 0},
            }

    def train_army(self, army_composition: dict) -> Optional[UnitCommand]:
        if self.larva and 1 <= self.supply_left:
            self.register_behavior(SpawnController(army_composition))
            return "making army"

    def micro(self) -> None:
        self.micro_army()
        self.micro_queens()

    def micro_army(self) -> None:
        invisible_enemy_start_locations = [p for p in self.enemy_start_locations if not self.is_visible(p)]
        for unit in self.mediator.get_units_from_role(role=UnitRole.ATTACKING):
            if not unit.is_idle:
                pass
            elif self.enemy_structures:
                unit.attack(self.enemy_structures.random.position)
            elif any(invisible_enemy_start_locations):
                unit.attack(random.choice(invisible_enemy_start_locations))
            else:
                a = self.game_info.playable_area
                target = np.random.uniform((a.x, a.y), (a.right, a.top))
                target = Point2(target)
                if self.in_pathing_grid(target) and not self.is_visible(target):
                    unit.attack(target)

    def micro_queens(self) -> None:
        inject_queens = (q for q in self.mediator.get_own_army_dict[UnitTypeId.QUEEN] if q.energy >= 25 and q.is_idle)
        inject_hatches = (h for h in self.townhalls if h.is_ready and not h.has_buff(BuffId.QUEENSPAWNLARVATIMER))
        for queen, hatch in zip(inject_queens, inject_hatches):
            queen(AbilityId.EFFECT_INJECTLARVA, hatch)

    def get_next_expansion(self) -> Optional[Point2]:
        townhall_positions = {th.position for th in self.townhalls}
        return next((p for p, d in self.mediator.get_own_expansions if p not in townhall_positions), None)
