from typing import Iterable

from ares.consts import ALL_STRUCTURES
from sc2.dicts.unit_trained_from import UNIT_TRAINED_FROM
from sc2.dicts.upgrade_researched_from import UPGRADE_RESEARCHED_FROM
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from ...action import Action
from ..component import Component
from ..strategy import StrategyDecision
from .macro_action import BuildUnit, MacroAction, ResearchUpgrade, WaitForResources


class Macro(Component):
    def macro(self, strategy: StrategyDecision) -> Iterable[Action]:
        if not self.build_order_runner.build_completed:
            return
        macro_action = (
            self.wait_for_build_order_completion()
            or self.spend_larva(strategy.build_unit)
            or self.train_queen()
            or self.research_speed()
            or self.expand()
            or self.make_tech(build_spire=strategy.build_unit == UnitTypeId.MUTALISK)
            or self.morph_overlord()
            or WaitForResources()
        )
        yield macro_action.execute()

    def wait_for_build_order_completion(self) -> MacroAction | None:
        if self.build_order_runner.build_completed:
            return None
        return WaitForResources()

    def expand(self) -> MacroAction | None:
        if not self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED):
            return None
        elif self.already_pending(UnitTypeId.HATCHERY):
            return None
        elif not (target := self.get_next_expansion()):
            return None
        elif not (trainer := self.find_trainer(UnitTypeId.HATCHERY, target)):
            return None
        elif not self.can_afford(UnitTypeId.HATCHERY):
            return WaitForResources()
        return BuildUnit(UnitTypeId.HATCHERY, trainer, target)

    def make_tech(self, build_spire: bool) -> MacroAction | None:
        tech_building_position = self.start_location.towards(self.game_info.map_center, 8)
        if build_spire:
            return (
                self.build_structure(UnitTypeId.SPAWNINGPOOL, tech_building_position)
                or self.build_structure(UnitTypeId.LAIR)
                or self.build_structure(UnitTypeId.SPIRE, tech_building_position)
            )
        else:
            return self.build_structure(UnitTypeId.SPAWNINGPOOL, tech_building_position)

    def train_single(self, unit: UnitTypeId) -> MacroAction | None:
        if not (trainer := self.find_trainer(unit)):
            return None
        elif self.already_pending(unit):
            return None
        elif self.supply_left < self.calculate_supply_cost(unit):
            return None
        elif not self.can_afford(unit):
            return WaitForResources()
        return BuildUnit(unit, trainer)

    def morph_overlord(self) -> MacroAction | None:
        if 0 < self.supply_left:
            return None
        return self.train_single(UnitTypeId.OVERLORD)

    def train_queen(self) -> MacroAction | None:
        if self.tech_requirement_progress(UnitTypeId.QUEEN) < 1:
            return None
        elif self.townhalls.amount <= len(self.mediator.get_own_army_dict[UnitTypeId.QUEEN]):
            return None
        return self.train_single(UnitTypeId.QUEEN)

    def research_speed(self) -> MacroAction | None:
        if self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED):
            return None
        elif not (trainer := self.find_trainer(UpgradeId.ZERGLINGMOVEMENTSPEED)):
            return None
        elif self.vespene < 80:
            return None
        elif not self.can_afford(UpgradeId.ZERGLINGMOVEMENTSPEED):
            return WaitForResources()
        return ResearchUpgrade(UpgradeId.ZERGLINGMOVEMENTSPEED, trainer)

    def build_structure(self, type_id: UnitTypeId, target: Point2 | None = None) -> MacroAction | None:
        if self.mediator.get_own_structures_dict[type_id]:
            return None
        elif self.already_pending(type_id):
            return None
        elif not (trainer := self.find_trainer(type_id)):
            return None
        elif not self.can_afford(type_id):
            return WaitForResources()
        return BuildUnit(type_id, trainer, target=target)

    def spend_larva(self, unit: UnitTypeId) -> MacroAction | None:
        if not (larva := next(iter(self.larva), None)):
            return None
        elif self.supply_left < self.calculate_supply_cost(unit):
            return None
        elif self.tech_requirement_progress(unit) < 1:
            return None
        elif not self.can_afford(unit):
            return WaitForResources()
        return BuildUnit(unit, larva)

    def get_next_expansion(self) -> Point2 | None:
        taken = {th.position for th in self.townhalls}
        return next((p for p, d in self.mediator.get_own_expansions if p not in taken), None)

    def find_trainer(
        self,
        type_id: UnitTypeId | UpgradeId,
        target: Point2 | None = None,
    ) -> Unit | None:
        def filter_trainer(t: Unit) -> bool:
            # TODO: handle reactors
            if t.type_id in ALL_STRUCTURES and not t.is_idle:
                return False
            return True

        def trainer_priority(t: Unit) -> float:
            return -t.position.distance_to(target or self.start_location)

        trainer_types = (
            UNIT_TRAINED_FROM[type_id] if isinstance(type_id, UnitTypeId) else {UPGRADE_RESEARCHED_FROM[type_id]}
        )

        def trainer_pool(t: UnitTypeId) -> Units:
            if t in ALL_STRUCTURES:
                return self.mediator.get_own_structures_dict[t]
            else:
                return self.mediator.get_own_army_dict[t]

        trainers = (t for t_id in trainer_types for t in trainer_pool(t_id) if filter_trainer(t))

        return max(trainers, key=trainer_priority, default=None)