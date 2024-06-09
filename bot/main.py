from ares import AresBot

from itertools import chain

from .components.strategy import Strategy
from .components.micro import Micro
from .components.macro import Macro
from .combat_predictor import CombatPredictor, CombatPredictionContext


class TwelvePoolBot(Strategy, Micro, Macro, AresBot):

    combat_predictor = CombatPredictor()

    async def on_start(self) -> None:
        await super().on_start()

    async def on_step(self, iteration: int) -> None:
        await super().on_step(iteration)


        strategy = self.decide_strategy()
        combat_prediction = self.combat_predictor.predict(self.prediction_context)
        actions = chain(
            self.macro(strategy),
            self.micro(combat_prediction),
        )
        for action in actions:
            success = await action.execute(self)
            if not success:
                raise Exception(f"Action failed: {action}")

    @property
    def prediction_context(self) -> CombatPredictionContext:
        return CombatPredictionContext(
            pathing=self.game_info.pathing_grid.data_numpy.T,
            civilians=chain(self.structures, self.enemy_structures),
            combatants=chain(self.all_own_units, self.all_enemy_units),
        )
