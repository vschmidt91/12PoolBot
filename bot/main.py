from itertools import chain

import numpy as np
from ares import DEBUG, AresBot
from ares.behaviors.macro import Mining
from ares.consts import CHANGELING_TYPES
from loguru import logger
from sc2.constants import WORKER_TYPES
from sc2.ids.unit_typeid import UnitTypeId

from .combat_predictor import CombatPredictionContext, predict
from .components.macro import Macro
from .components.micro import Micro
from .components.strategy import Strategy
from .utils.debug import save_map

EXCLUDE_TYPES = WORKER_TYPES | CHANGELING_TYPES | {UnitTypeId.LARVA, UnitTypeId.EGG, UnitTypeId.BROODLING}


class TwelvePoolBot(Strategy, Micro, Macro, AresBot):
    max_micro_actions = 100

    async def on_start(self) -> None:
        await super().on_start()

        if self.config[DEBUG]:
            save_map(self.game_info, "resources")

    async def on_step(self, iteration: int) -> None:
        await super().on_step(iteration)

        strategy = self.decide_strategy()
        combat_prediction = predict(self.prediction_context)

        macro_actions = list(self.macro(strategy.build_unit))
        micro_actions = list(self.micro(combat_prediction))

        # avoid APM bug
        if self.max_micro_actions < len(micro_actions):
            logger.info(f"Limiting micro actions: {len(micro_actions)} => {self.max_micro_actions}")
            micro_actions = np.random.choice(micro_actions, size=self.max_micro_actions, replace=False)

        actions = chain(macro_actions, micro_actions)
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
        combatants = [u for u in chain(self.all_own_units, self.all_enemy_units) if u.type_id not in EXCLUDE_TYPES]
        return CombatPredictionContext(
            pathing=self.game_info.pathing_grid.data_numpy.T,
            civilians=chain(self.structures, self.enemy_structures),
            combatants=combatants,
        )
