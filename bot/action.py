from ares import AresBot
from ares.behaviors.macro import Mining
from ares.consts import UnitRole

from dataclasses import dataclass
from typing import Optional, Protocol

from loguru import logger
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit


class Action(Protocol):
    async def execute(self, bot: AresBot) -> bool:
        ...


class DoNothing(Action):
    async def execute(self, bot: AresBot) -> bool:
        return True


@dataclass
class Attack(Action):
    unit: Unit
    target: Point2

    async def execute(self, bot: AresBot) -> bool:
        return self.unit.attack(self.target)


@dataclass
class UseAbility(Action):
    unit: Unit
    ability: AbilityId
    target: Optional[Point2] = None

    async def execute(self, bot: AresBot) -> bool:
        logger.info(self)
        return self.unit(self.ability, target=self.target)


@dataclass
class Build(Action):
    unit: Unit
    type_id: UnitTypeId
    near: Point2

    async def execute(self, bot: AresBot) -> bool:
        logger.info(self)
        bot.mediator.assign_role(tag=self.unit.tag, role=UnitRole.PERSISTENT_BUILDER)
        if placement := await bot.find_placement(self.type_id, near=self.near):
            return self.unit.build(self.type_id, placement)
        else:
            return False


@dataclass
class GatherResources(Action):
    workers_per_gas: int = 3

    async def execute(self, bot: AresBot) -> bool:
        bot.register_behavior(Mining(workers_per_gas=self.workers_per_gas))
        return True
