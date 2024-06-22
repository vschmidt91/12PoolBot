import cProfile
import io
import os
import pstats
import sys
from itertools import chain

import numpy as np
from ares import DEBUG, AresBot
from ares.behaviors.macro import Mining
from loguru import logger
from sc2.ids.unit_typeid import UnitTypeId

from .components.combat_predictor import CombatPredictor
from .components.macro import Macro
from .components.micro import Micro
from .components.strategy import Strategy
from .components.tags import Tags
from .consts import (
    PROFILING_FILE,
    TAG_ACTION_FAILED,
    TAG_MICRO_THROTTLING,
    UNKNOWN_VERSION,
    VERSION_FILE,
)
from .utils.debug import save_map


class TwelvePoolBot(CombatPredictor, Strategy, Micro, Macro, Tags, AresBot):
    max_micro_actions = 100
    version: str = UNKNOWN_VERSION

    async def on_start(self) -> None:
        await super().on_start()

        if sys.gettrace():
            self.config[DEBUG] = True

        if self.config[DEBUG]:
            save_map(self.game_info, "resources")
            await self.client.debug_create_unit([[UnitTypeId.ZERGLING, 100, self.game_info.map_center, 1]])
            await self.client.debug_create_unit([[UnitTypeId.ZERGLING, 100, self.game_info.map_center, 2]])

        if os.path.exists(VERSION_FILE):
            with open(VERSION_FILE) as f:
                self.version = f.read()

        await self.add_tag(f"version_{self.version}")

    async def on_step(self, iteration: int) -> None:
        await super().on_step(iteration)

        profiler: cProfile.Profile | None = None
        if self.config[DEBUG] and (iteration % 30) == 10:
            profiler = cProfile.Profile()

        if profiler:
            profiler.enable()
        strategy = self.decide_strategy()
        combat_prediction = self.predict_combat()

        if strategy.build_unit not in {UnitTypeId.ZERGLING, UnitTypeId.DRONE}:
            await self.add_tag(f"macro_{strategy.build_unit.name}")

        macro_actions = list(self.macro(strategy.build_unit))
        micro_actions = list(self.micro(combat_prediction))

        # avoid APM bug
        if self.max_micro_actions < len(micro_actions):
            await self.add_tag(TAG_MICRO_THROTTLING)
            logger.info(f"Limiting micro actions: {len(micro_actions)} => {self.max_micro_actions}")
            micro_actions = np.random.choice(np.asarray(micro_actions), size=self.max_micro_actions, replace=False)

        if profiler:
            profiler.disable()
            stats_io = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_io).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(24)
            logger.info(stats_io.getvalue())
            stats.dump_stats(PROFILING_FILE)

        actions = chain(macro_actions, micro_actions)
        for action in actions:
            success = await action.execute(self)
            if not success:
                await self.add_tag(TAG_ACTION_FAILED)
                if self.config[DEBUG]:
                    raise Exception(f"Action failed: {action}")
                else:
                    logger.warning(f"Action failed: {action}")

        self.register_behavior(Mining(workers_per_gas=strategy.vespene_target))
