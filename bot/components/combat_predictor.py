import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

import numpy as np
import skimage.draw
from ares.dicts.weight_costs import WEIGHT_COSTS
from consts import EXCLUDE_FROM_COMBAT
from loguru import logger
from sc2.position import Point2
from sc2.units import Units
from scipy import ndimage
from utils.dijkstra import Point

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
        pathing = self.mediator.get_map_data_object.get_pyastar_grid()
        context = CombatPredictionContext(
            pathing=pathing,
            units=units,
            enemy_units=enemy_units,
        )

        presence = self.combat_presence(context.units)
        enemy_presence = self.combat_presence(context.enemy_units)
        confidence = (presence - enemy_presence) / np.maximum(presence, enemy_presence)
        return CombatPrediction(
            context=context,
            presence=presence,
            enemy_presence=enemy_presence,
            confidence=confidence,
        )

    def combat_presence(self, units: Units) -> np.ndarray:
        force = np.zeros(self.game_info.map_size, dtype=float)
        for unit in units:
            dps = self.ground_dps_fast(unit.type_id)
            if 0 < dps:
                d = _disk(unit.sight_range)
                i, j = unit.position.rounded + d
                health = unit.health + unit.shield
                force[i.astype(int), j.astype(int)] += dps * health
        return force
