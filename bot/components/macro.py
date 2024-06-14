import math
from typing import Iterable, Optional

from ares.consts import ALL_STRUCTURES

from sc2.dicts.unit_trained_from import UNIT_TRAINED_FROM
from sc2.dicts.upgrade_researched_from import UPGRADE_RESEARCHED_FROM
from sc2.dicts.unit_train_build_abilities import TRAIN_INFO
from sc2.dicts.unit_research_abilities import RESEARCH_INFO
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.unit import Unit
from sc2.position import Point2

from ..action import Action, Build, DoNothing, UseAbility
from .component import Component


class Macro(Component):
    def macro(self, build_spire: bool) -> Iterable[Action]:
        return [
            (
                self.wait_for_build_order_completion()
                or self.make_tech(build_spire)
                or self.train_queen()
                or self.get_upgrades()
                or self.train_army()
                or self.expand()
                or self.morph_overlord()
                or DoNothing()
            )
        ]

    def wait_for_build_order_completion(self) -> Optional[Action]:
        if not self.build_order_runner.build_completed:
            return DoNothing()
        else:
            return None

    def expand(self) -> Optional[Action]:
        if self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED) and (target := self.get_next_expansion()):
            return self.make_unit(UnitTypeId.HATCHERY, target=target, max_pending=1)
        else:
            return None

    def make_tech(self, build_spire: bool) -> Optional[Action]:
        if build_spire:
            return (
                self.ensure_structure_exists(UnitTypeId.SPAWNINGPOOL)
                or self.ensure_lair_exists()
                or self.ensure_structure_exists(UnitTypeId.SPIRE)
            )
        else:
            return self.ensure_structure_exists(UnitTypeId.SPAWNINGPOOL)

    def morph_overlord(self) -> Optional[Action]:
        if self.supply_left < 1:
            return self.make_unit(UnitTypeId.OVERLORD, max_pending=1)
        else:
            return None

    def train_queen(self) -> Optional[Action]:
        if len(self.mediator.get_own_army_dict[UnitTypeId.QUEEN]) < self.townhalls.amount:
            return self.make_unit(UnitTypeId.QUEEN, idle_trainers=True, max_pending=1)
        else:
            return None

    def get_upgrades(self) -> Optional[Action]:
        if 80 < self.vespene:
            return self.make_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED)
        else:
            return None

    def ensure_structure_exists(self, type_id: UnitTypeId) -> Optional[Action]:
        placement_near = self.start_location.towards(self.game_info.map_center, 8)
        if not self.mediator.get_own_structures_dict[type_id]:
            return self.make_unit(type_id, target=placement_near, max_pending=1)
        else:
            return None

    def ensure_lair_exists(self) -> Optional[Action]:
        if not self.mediator.get_own_structures_dict[UnitTypeId.LAIR]:
            return self.make_unit(UnitTypeId.LAIR, idle_trainers=True, max_pending=1)
        else:
            return None

    def train_army(self) -> Optional[Action]:
        larva_per_second = sum(
            sum(
                (
                    1 / 11 if h.is_ready else 0,
                    3 / 29 if h.has_buff(BuffId.QUEENSPAWNLARVATIMER) else 0,
                )
            )
            for h in self.townhalls
        )
        minerals_for_lings = 50 * 60 * larva_per_second  # maximum we can possibly spend on lings
        max_spending = minerals_for_lings  # aim for a 20% surplus
        should_drone = (
            self.minerals < 150
            and self.state.score.collection_rate_minerals < 1.2 * max_spending  # aim for a 20% surplus
            and self.state.score.food_used_economy < sum(h.ideal_harvesters for h in self.townhalls)
            and not self.already_pending(UnitTypeId.DRONE)
        )
        if should_drone:
            return self.make_unit(UnitTypeId.DRONE)
        else:
            return self.make_unit(UnitTypeId.MUTALISK) or self.make_unit(UnitTypeId.ZERGLING)

    def get_next_expansion(self) -> Optional[Point2]:
        taken = {th.position for th in self.townhalls}
        return next((p for p, d in self.mediator.get_own_expansions if p not in taken), None)

    def make_unit(
        self,
        type_id: UnitTypeId,
        target: Point2 | None = None,
        idle_trainers: bool = False,
        max_pending: float = math.inf,
    ) -> Optional[Action]:
        if max_pending is not None and max_pending <= self.already_pending(type_id):
            return None

        def filter_trainer(t: Unit) -> bool:
            if idle_trainers and not t.is_idle:
                return False
            return True

        trainer_type_id = min(UNIT_TRAINED_FROM[type_id], key=lambda v: v.value)
        trainer_dict = (
            self.mediator.get_own_structures_dict
            if trainer_type_id in ALL_STRUCTURES
            else self.mediator.get_own_army_dict
        )
        trainers = (t for t in trainer_dict[trainer_type_id] if filter_trainer(t))
        if (
            self.can_afford(type_id)
            and self.tech_requirement_progress(type_id) == 1
            and (trainer := next(trainers, None))
        ):
            if target is not None:
                return Build(trainer, type_id, target)
            else:
                ability = TRAIN_INFO[trainer_type_id][type_id]["ability"]
                return UseAbility(trainer, ability)
        else:
            return None

    def make_upgrade(
        self,
        upgrade_id: UpgradeId,
    ) -> Optional[Action]:
        if self.already_pending_upgrade(upgrade_id):
            return None

        trainer_type_id = UPGRADE_RESEARCHED_FROM[upgrade_id]
        trainer_dict = (
            self.mediator.get_own_structures_dict
            if trainer_type_id in ALL_STRUCTURES
            else self.mediator.get_own_army_dict
        )
        trainers = (t for t in trainer_dict[trainer_type_id] if t.is_idle)
        if self.can_afford(upgrade_id) and (trainer := next(trainers, None)):
            ability = RESEARCH_INFO[trainer_type_id][upgrade_id]["ability"]
            return UseAbility(trainer, ability)
        else:
            return None
