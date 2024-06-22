import functools
from dataclasses import dataclass

import numpy as np
import skimage.draw
from sc2.position import Point2
from sc2.units import Units

from ..consts import EXCLUDE_FROM_COMBAT
from .component import Component


@dataclass
class CombatPredictionContext:
    pathing: np.ndarray
    units: Units
    enemy_units: Units


@dataclass
class CombatPrediction:
    context: CombatPredictionContext
    presence: np.ndarray
    enemy_presence: np.ndarray
    confidence: np.ndarray


@functools.lru_cache(maxsize=None)
def _disk(radius: float):
    p = radius, radius
    n = 2 * radius + 1
    d = skimage.draw.disk(center=p, radius=radius, shape=(n, n))
    return -Point2(p) + d


class CombatPredictor(Component):
    def predict_combat(self) -> CombatPrediction:
        units = self.all_own_units.exclude_type(EXCLUDE_FROM_COMBAT)
        enemy_units = self.all_enemy_units.exclude_type(EXCLUDE_FROM_COMBAT)
        pather = self.mediator.get_map_data_object
        pathing = pather.get_pyastar_grid()
        context = CombatPredictionContext(
            pathing=pathing,
            units=units,
            enemy_units=enemy_units,
        )

        presence = self.combat_presence(context.units)
        enemy_presence = self.combat_presence(context.enemy_units)
        confidence = np.log(presence / enemy_presence)
        return CombatPrediction(
            context=context,
            presence=presence,
            enemy_presence=enemy_presence,
            confidence=confidence,
        )

    def combat_presence(self, units: Units) -> np.ndarray:
        grid = self.mediator.get_map_data_object.get_pyastar_grid()
        for unit in units:
            dps = self.dps_fast(unit.type_id)
            if 0 < dps:
                health = unit.health + unit.shield
                grid = self.mediator.get_map_data_object.add_cost(
                    position=unit.position,
                    radius=unit.sight_range,
                    grid=grid,
                    weight=dps * health,
                )
                # d = _disk(unit.sight_range)
                # i, j = unit.position.rounded + d
                #
                # force[i.astype(int), j.astype(int)] += dps * health

        return grid
