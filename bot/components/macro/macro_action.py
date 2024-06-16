from abc import ABC, abstractmethod
from dataclasses import dataclass

from sc2.dicts.unit_research_abilities import RESEARCH_INFO
from sc2.dicts.unit_train_build_abilities import TRAIN_INFO
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit

from ...action import Action, Build, DoNothing, UseAbility


class MacroAction(ABC):
    @abstractmethod
    def execute(self) -> Action:
        raise NotImplementedError()


class WaitForResources(MacroAction):
    def execute(self) -> Action:
        return DoNothing()


@dataclass
class BuildUnit(MacroAction):
    unit: UnitTypeId
    trainer: Unit
    target: Point2 | None = None

    def execute(self) -> Action:
        if self.target is not None:
            return Build(self.trainer, self.unit, self.target)
        else:
            ability = TRAIN_INFO[self.trainer.type_id][self.unit]["ability"]
            return UseAbility(self.trainer, ability)


@dataclass
class ResearchUpgrade(MacroAction):
    upgrade: UpgradeId
    trainer: Unit

    def execute(self) -> Action:
        ability = RESEARCH_INFO[self.trainer.type_id][self.upgrade]["ability"]
        return UseAbility(self.trainer, ability)
