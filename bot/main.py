from ares import AresBot

from itertools import chain
from sc2.ids.upgrade_id import UpgradeId

from micro import Micro
from macro import Macro
from strategy import Strategy


class TwelvePoolBot(Micro, Macro, AresBot):

    async def on_step(self, iteration: int) -> None:
        await super().on_step(iteration)

        strategy = self.pick_strategy()
        actions = chain(
            self.macro(strategy),
            self.micro(strategy),
        )
        for action in actions:
            success = await action.execute(self)
            if not success:
                raise Exception(f"Action failed: {action}")

    def pick_strategy(self) -> Strategy:
        mutalisk_switch = self.enemy_structures.flying and not self.enemy_structures.not_flying
        saving_for_speed = self.vespene < 92 and not self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED)
        gather_vespene = saving_for_speed or mutalisk_switch
        return Strategy(
            mutalisk_switch=mutalisk_switch,
            gather_vespene=gather_vespene,
        )
