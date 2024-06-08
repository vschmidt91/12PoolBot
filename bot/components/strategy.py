from dataclasses import dataclass

from sc2.ids.upgrade_id import UpgradeId

from .component import Component


@dataclass
class StrategyDecision:
    mutalisk_switch: bool
    gather_vespene: bool


class Strategy(Component):
    def decide_strategy(self) -> StrategyDecision:
        mutalisk_switch = self.enemy_structures.flying and not self.enemy_structures.not_flying
        saving_for_speed = self.vespene < 92 and not self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED)
        gather_vespene = saving_for_speed or mutalisk_switch
        return StrategyDecision(
            mutalisk_switch=mutalisk_switch,
            gather_vespene=gather_vespene,
        )
