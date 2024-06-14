from ares import AresBot, DEBUG

from itertools import chain

from .components.strategy import Strategy
from .components.micro import Micro
from .components.macro import Macro
from .combat_predictor import predict, CombatPredictionContext
from .utils.debug import save_map

from loguru import logger


class TwelvePoolBot(Strategy, Micro, Macro, AresBot):
    async def on_start(self) -> None:
        await super().on_start()

        if self.config[DEBUG]:
            save_map(self.game_info, "resources")

    async def on_step(self, iteration: int) -> None:
        await super().on_step(iteration)

        strategy = self.decide_strategy()
        combat_prediction = predict(self.prediction_context)
        actions = chain(
            self.macro(build_spire=strategy.mutalisk_switch),
            self.micro(combat_prediction),
        )
        for action in actions:
            success = await action.execute(self)
            if not success:
                if self.config[DEBUG]:
                    raise Exception(f"Action failed: {action}")
                else:
                    logger.warning(f"Action failed: {action}")

    @property
    def prediction_context(self) -> CombatPredictionContext:
        return CombatPredictionContext(
            pathing=self.game_info.pathing_grid.data_numpy.T,
            civilians=chain(self.structures, self.enemy_structures),
            combatants=chain(self.all_own_units, self.all_enemy_units),
        )
