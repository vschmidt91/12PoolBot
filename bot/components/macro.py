from typing import Iterable, Optional

from sc2.dicts.unit_trained_from import UNIT_TRAINED_FROM
from sc2.dicts.unit_train_build_abilities import TRAIN_INFO
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2

from actions import Action, Build, DoNothing, UseAbility
from .component import Component
from .strategy import StrategyDecision


class Macro(Component):
    def macro(self, strategy: StrategyDecision) -> Iterable[Action]:
        return [
            (
                self.wait_for_build_order_completion()
                or self.make_tech(strategy)
                or self.get_upgrades()
                or self.train_army()
                or self.train_queen()
                or self.expand()
                or self.morph_overlord()
                or DoNothing()
            )
        ]

    def wait_for_build_order_completion(self) -> Optional[Action]:
        if not self.build_order_runner.build_completed:
            return DoNothing()
        return None

    def expand(self) -> Optional[Action]:
        if (
            self.can_afford(UnitTypeId.HATCHERY)
            and not self.already_pending(UnitTypeId.HATCHERY)
            and (builder := next((u for u in self.workers.collecting), None))
            and (target := self.get_next_expansion())
        ):
            return Build(builder, UnitTypeId.HATCHERY, target)
        return None

    def make_tech(self, strategy: StrategyDecision) -> Optional[Action]:
        if strategy.mutalisk_switch:
            return (
                self.ensure_structure_exists(UnitTypeId.SPAWNINGPOOL)
                or self.ensure_lair_exists()
                or self.ensure_structure_exists(UnitTypeId.SPIRE)
            )
        else:
            return self.ensure_structure_exists(UnitTypeId.SPAWNINGPOOL)

    def morph_overlord(self) -> Optional[Action]:
        if self.supply_left < 1 and not self.already_pending(UnitTypeId.OVERLORD):
            return self.train_unit(UnitTypeId.OVERLORD)
        return None

    def train_queen(self) -> Optional[Action]:
        if (
            self.can_afford(UnitTypeId.QUEEN)
            and not self.already_pending(UnitTypeId.QUEEN)
            and self.tech_requirement_progress(UnitTypeId.QUEEN) == 1
            and len(self.mediator.get_own_army_dict[UnitTypeId.QUEEN]) < self.townhalls.amount
            and (hatch := next((t for t in self.townhalls if t.is_idle), None))
        ):
            return UseAbility(hatch, AbilityId.TRAINQUEEN_QUEEN)
        return None

    def get_upgrades(self) -> Optional[Action]:
        if (
            self.can_afford(UpgradeId.ZERGLINGMOVEMENTSPEED)
            and not self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED)
            and (
                pool := next(
                    (p for p in self.mediator.get_own_structures_dict[UnitTypeId.SPAWNINGPOOL] if p.is_idle), None
                )
            )
        ):
            return UseAbility(pool, AbilityId.RESEARCH_ZERGLINGMETABOLICBOOST)
        return None

    def ensure_structure_exists(self, type_id: UnitTypeId) -> Optional[Action]:
        placement_near = self.start_location.towards(self.game_info.map_center, 8)
        trainer_type_id = min(UNIT_TRAINED_FROM[type_id], key=lambda v: v.value)
        if (
            self.can_afford(type_id)
            and not self.mediator.get_own_structures_dict[type_id]
            and not self.already_pending(type_id)
            and (trainer := next((t for t in self.mediator.get_own_army_dict[trainer_type_id]), None))
        ):
            return Build(trainer, type_id, placement_near)
        return None

    def ensure_lair_exists(self) -> Optional[Action]:
        if (
            self.can_afford(UnitTypeId.LAIR)
            and not self.mediator.get_own_structures_dict[UnitTypeId.LAIR]
            and not self.already_pending(UnitTypeId.LAIR)
            and (
                hatch := next(
                    (t for t in self.mediator.get_own_structures_dict[UnitTypeId.HATCHERY] if t.is_idle), None
                )
            )
        ):
            return UseAbility(hatch, AbilityId.UPGRADETOLAIR_LAIR)
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
            return self.train_unit(UnitTypeId.DRONE)
        else:
            return self.train_unit(UnitTypeId.MUTALISK) or self.train_unit(UnitTypeId.ZERGLING)

    def train_unit(self, type_id: UnitTypeId) -> Optional[Action]:
        trainer_type_id = min(UNIT_TRAINED_FROM[type_id], key=lambda v: v.value)
        if (
            self.can_afford(type_id)
            and self.tech_requirement_progress(type_id) == 1
            and (trainer := next((t for t in self.mediator.get_own_army_dict[trainer_type_id]), None))
        ):
            ability = TRAIN_INFO[trainer_type_id][type_id]["ability"]
            return UseAbility(trainer, ability)
        return None

    def get_next_expansion(self) -> Optional[Point2]:
        taken = {th.position for th in self.townhalls}
        return next((p for p, d in self.mediator.get_own_expansions if p not in taken), None)
