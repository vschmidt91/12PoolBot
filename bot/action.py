from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ares import AresBot
from ares.consts import UnitRole
from loguru import logger
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit


class Action(ABC):
    @abstractmethod
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
class Move(Action):
    unit: Unit
    target: Point2

    async def execute(self, bot: AresBot) -> bool:
        return self.unit.move(self.target)


@dataclass
class UseAbility(Action):
    unit: Unit
    ability: AbilityId
    target: Optional[Point2] = None

    async def execute(self, bot: AresBot) -> bool:
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
