import lzma
import os
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import chain

import numpy as np
from combat import Combat, CombatDataset, CombatOutcome, CombatSetup, CombatUnit
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race, UnitTypeId
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.unit import Unit
from sc2.units import Units
from sc2_helper import CombatPredictor, CombatSettings
from tqdm import tqdm

SET_ENERGY = 1
SET_HEALTH = 2
SET_SHIELDS = 3


class CombatSimulationBot(BotAI):
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_path = "dataset.xz"
    unit_types = {
        Race.Zerg: [
            UnitTypeId.BANELING,
            # UnitTypeId.BROODLORD,
            # UnitTypeId.CORRUPTOR,
            UnitTypeId.DRONE,
            # UnitTypeId.HYDRALISK,
            # UnitTypeId.LURKERMPBURROWED,
            # UnitTypeId.MUTALISK,
            UnitTypeId.QUEEN,
            # UnitTypeId.RAVAGER,
            UnitTypeId.ROACH,
            # UnitTypeId.ULTRALISK,
            UnitTypeId.ZERGLING,
        ],
        Race.Terran: [
            # UnitTypeId.AUTOTURRET,
            # UnitTypeId.BANSHEE,
            # UnitTypeId.BATTLECRUISER,
            UnitTypeId.CYCLONE,
            # UnitTypeId.GHOST,
            UnitTypeId.HELLION,
            # UnitTypeId.LIBERATOR,
            UnitTypeId.MARINE,
            UnitTypeId.SCV,
            # UnitTypeId.THOR,
            # UnitTypeId.VIKINGASSAULT,
            # UnitTypeId.VIKINGFIGHTER,
            # UnitTypeId.WIDOWMINEBURROWED,
        ],
        Race.Protoss: [
            UnitTypeId.ADEPT,
            UnitTypeId.ARCHON,
            # UnitTypeId.CARRIER,
            # UnitTypeId.MOTHERSHIP,
            # UnitTypeId.ORACLE,
            # UnitTypeId.PHOENIX,
            UnitTypeId.PROBE,
            UnitTypeId.SENTRY,
            UnitTypeId.STALKER,
            # UnitTypeId.TEMPEST,
            # UnitTypeId.VOIDRAY,
        ],
    }
    max_count = 1
    num_combats = 10_000

    @property
    def all_unit_types(self) -> list[UnitTypeId]:
        return [t for r, ts in self.unit_types.items() for t in ts]

    def sample_army(self, units: Units) -> list[Unit]:
        race = np.random.choice(list(self.unit_types.keys()))
        complexity = np.random.choice((1, 2, 3))
        composition = set(np.random.choice(list(self.unit_types[race]), replace=False, size=complexity))
        units = [
            units(u)[0]
            for u in composition
            for _ in range(1 + np.random.poisson(12 / complexity))
        ]
        return units

    async def on_step(self, iteration):
        if iteration == 0:
            await self.client.debug_show_map()

        elif iteration == 1:
            await self.client.debug_kill_unit({u.tag for u in self.all_units(UnitTypeId.SCV)})

            self.combat_predictor = CombatPredictor()
            self.combat_settings = CombatSettings()
            # self.combat_settings.debug = True

            position = self.game_info.map_center

            await self.client.debug_create_unit([[t, self.max_count, position, 1] for t in self.all_unit_types])
            await self.client.debug_create_unit([[t, self.max_count, position, 2] for t in self.all_unit_types])

        elif iteration == 2:
            combats: list[Combat] = []
            for _ in tqdm(range(self.num_combats)):
                units = self.sample_army(self.units)
                enemy_units = self.sample_army(self.enemy_units)

                # set random health and shields
                # for u in units + enemy_units:
                #     health = 1 + np.random.randint(0, u.health_max + 1)
                #     await self.client.debug_set_unit_value(u, SET_HEALTH, value=health)
                #     shield = np.random.randint(0, u.shield_max + 1)
                #     await self.client.debug_set_unit_value(u, SET_SHIELDS, value=shield)

                health = sum([u.health + u.shield for u in units]) + 1e-16
                enemy_health = sum([u.health + u.shield for u in enemy_units]) + 1e-16

                settings = self.combat_settings
                setup = CombatSetup(
                    units=[CombatUnit.from_unit(u) for u in units],
                    enemy_units=[CombatUnit.from_unit(u) for u in enemy_units],
                    health=health,
                    enemy_health=enemy_health,
                )

                winner, winner_health = self.combat_predictor.predict_engage(units, enemy_units, 1, settings)
                win = winner == 1
                result = winner_health / health if win else -winner_health / enemy_health
                outcome = CombatOutcome(
                    win=win,
                    winner_health=winner_health,
                    result=result,
                )
                combats.append(
                    Combat(
                        setup=setup,
                        outcome=outcome,
                    )
                )

            dataset = CombatDataset(
                unit_types=self.all_unit_types,
                combats=combats,
            )
            with lzma.open(self.output_path, "wb") as f:
                pickle.dump(dataset, f)

            await self.client.leave()


if __name__ == "__main__":
    run_game(
        maps.get("SiteDelta513AIE"),
        [
            Bot(Race.Terran, CombatSimulationBot()),
            Computer(Race.Terran, Difficulty.VeryEasy),
        ],
        realtime=False,
    )
