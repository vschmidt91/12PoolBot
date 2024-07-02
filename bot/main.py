import cProfile
import io
import os
import pickle
import pstats
import random
import sys
from functools import lru_cache
from itertools import chain

from ares import DEBUG, AresBot
from ares.behaviors.macro import Mining
from loguru import logger
from sc2.data import Result
from sc2.ids.unit_typeid import UnitTypeId

from .combat_predictor import CombatContext, CombatPrediction, predict_combat
from .components.macro import Macro
from .components.micro import Micro
from .components.strategy import Strategy
from .consts import (
    EXCLUDE_FROM_COMBAT,
    PROFILING_FILE,
    RESULT_PREDICTOR_FILE,
    TAG_ACTION_FAILED,
    TAG_MICRO_THROTTLING,
    UNKNOWN_VERSION,
    VERSION_FILE,
)
from .data import GameReplay, GameResult, GameState
from .result_predictor import ResultPredictor
from .tags import Tags
from .utils.debug import save_map


class TwelvePoolBot(Strategy, Micro, Macro, AresBot):
    max_micro_actions = 80
    version: str = UNKNOWN_VERSION
    tags: Tags
    result_predictor: ResultPredictor
    states: list[GameState] = []

    async def on_start(self) -> None:
        await super().on_start()

        result_predictor: ResultPredictor | None = None
        if os.path.exists(RESULT_PREDICTOR_FILE):
            try:
                with open(RESULT_PREDICTOR_FILE, "rb") as f:
                    result_predictor = pickle.load(f)
            except Exception:
                pass
        if result_predictor is None:
            (state_size,) = GameState.from_bot(self).to_tensor().shape
            result_predictor = ResultPredictor(input_size=state_size)
        self.result_predictor = result_predictor

        self.tags = Tags(lambda m: self.chat_send(m, team_only=True))

        if sys.gettrace():
            self.config[DEBUG] = True

        if self.config[DEBUG]:
            # increase number of decimal places
            pstats.f8 = lambda x: "%14.9f" % x  # type: ignore
            save_map(self.game_info, "resources")
            # await self.client.debug_create_unit([[UnitTypeId.ZERGLING, 20, self.game_info.map_center, 2]])
            # await self.client.debug_create_unit([[UnitTypeId.ZERGLING, 20, self.game_info.map_center, 1]])

        if os.path.exists(VERSION_FILE):
            with open(VERSION_FILE) as f:
                self.version = f.read()

        await self.tags.add_tag(f"version_{self.version}")

    async def on_step(self, iteration: int) -> None:
        await super().on_step(iteration)

        profiler: cProfile.Profile | None = None
        # if self.config[DEBUG] and (iteration % 30) == 10:
        #     profiler = cProfile.Profile()

        if profiler:
            profiler.enable()
        strategy = self.decide_strategy()

        state = GameState.from_bot(self)
        self.states.append(state)
        if (iteration % 100) == 0:
            result_prediction = self.result_predictor.predict(state)
            logger.info(f"{iteration} {self.time_formatted} Predicted Result: {result_prediction:.3f}")
            if result_prediction < 0.01:
                await self.tags.add_tag("confidence_very_low")
            if result_prediction < 0.1:
                await self.tags.add_tag("confidence_low")
            if 0.9 < result_prediction:
                await self.tags.add_tag("confidence_high")
            if 0.99 < result_prediction:
                await self.tags.add_tag("confidence_very_high")

        combat_prediction = self.predict_combat()

        if strategy.build_unit not in {UnitTypeId.ZERGLING, UnitTypeId.DRONE}:
            await self.tags.add_tag(f"macro_{strategy.build_unit.name}")

        macro_actions = list(self.macro(strategy.build_unit))
        micro_actions = list(self.micro(combat_prediction))

        # avoid APM bug
        if self.max_micro_actions < len(micro_actions):
            await self.tags.add_tag(TAG_MICRO_THROTTLING)
            logger.info(f"Limiting micro actions: {len(micro_actions)} => {self.max_micro_actions}")
            random.shuffle(micro_actions)
            micro_actions = micro_actions[: self.max_micro_actions]

        if profiler:
            profiler.disable()
            stats_io = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_io).sort_stats(pstats.SortKey.TIME).print_stats(24)
            logger.info(stats_io.getvalue())
            stats.dump_stats(PROFILING_FILE)

        actions = chain(macro_actions, micro_actions)
        for action in actions:
            success = await action.execute(self)
            if not success:
                await self.tags.add_tag(TAG_ACTION_FAILED)
                if self.config[DEBUG]:
                    raise Exception(f"Action failed: {action}")
                else:
                    logger.warning(f"Action failed: {action}")

        self.register_behavior(Mining(workers_per_gas=strategy.vespene_target))

    async def on_end(self, game_result: Result) -> None:
        await super().on_end(game_result)
        result = GameResult(game_result)
        game = GameReplay(
            states=self.states,
            result=result,
        )
        self.result_predictor.train(game)
        with open(RESULT_PREDICTOR_FILE, "wb") as f:
            pickle.dump(self.result_predictor, f)

    def predict_combat(self) -> CombatPrediction:
        units = self.all_own_units.exclude_type(EXCLUDE_FROM_COMBAT)
        enemy_units = self.all_enemy_units.exclude_type(EXCLUDE_FROM_COMBAT)
        dps_provider = self.dps_fast
        pathing = self.mediator.get_ground_grid
        context = CombatContext(
            units=units,
            enemy_units=enemy_units,
            dps_provider=dps_provider,
            pathing=pathing,
        )
        return predict_combat(context)

    @lru_cache(maxsize=None)
    def dps_fast(self, unit: UnitTypeId) -> float:
        if units := self.all_units(unit):
            return max(units[0].ground_dps, units[0].air_dps)
        else:
            return 0.0
