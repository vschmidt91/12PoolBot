from dataclasses import dataclass

from ares.behaviors.macro import Mining
from sc2.ids.upgrade_id import UpgradeId
import numpy as np

from .component import Component


@dataclass
class StrategyDecision:
    mutalisk_switch: bool
    vespene_target: int


class Strategy(Component):
    def decide_strategy(self) -> StrategyDecision:
        mine_gas_for_speed = (
            0 if self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED) else (100 - self.vespene) // 4
        )
        mutalisk_switch = self.enemy_structures.flying and not self.enemy_structures.not_flying
        vespene_target = 3 if mutalisk_switch else np.clip(mine_gas_for_speed, 0, 3)
        self.register_behavior(Mining(workers_per_gas=vespene_target))
        return StrategyDecision(
            mutalisk_switch=mutalisk_switch,
            vespene_target=vespene_target,
        )
