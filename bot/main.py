from itertools import chain

from ares import DEBUG, AresBot
from ares.behaviors.macro import Mining
from loguru import logger
from sc2.constants import WORKER_TYPES

from .combat_predictor import CombatPredictionContext, predict
from .components.macro import Macro
from .components.micro import Micro
from .components.strategy import Strategy
from .utils.debug import save_map


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
            self.macro(strategy.build_unit),
            self.micro(combat_prediction),
        )
        for action in actions:
            success = await action.execute(self)
            if not success:
                if self.config[DEBUG]:
                    raise Exception(f"Action failed: {action}")
                else:
                    logger.warning(f"Action failed: {action}")

        self.register_behavior(Mining(workers_per_gas=strategy.vespene_target))

    @property
    def prediction_context(self) -> CombatPredictionContext:
        combatants = [u for u in chain(self.all_own_units, self.all_enemy_units) if u.type_id not in WORKER_TYPES]
        return CombatPredictionContext(
            pathing=self.game_info.pathing_grid.data_numpy.T,
            civilians=chain(self.structures, self.enemy_structures),
            combatants=combatants,
        )
