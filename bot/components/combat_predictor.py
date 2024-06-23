from functools import cached_property, lru_cache
from dataclasses import dataclass

import numpy as np
import scipy.ndimage
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
class CombatPresence:
    dps: np.ndarray
    health: np.ndarray


@dataclass
class CombatPrediction:
    context: CombatPredictionContext
    dimensionality: np.ndarray
    presence: CombatPresence
    enemy_presence: CombatPresence

    def confidence(self, p: Point2) -> float:
        pr = p.rounded
        e = self.dimensionality[pr]
        force = self.presence.dps[pr] * (self.presence.health[pr] ** e)
        enemy_force = self.enemy_presence.dps[pr] * (self.enemy_presence.health[pr] ** e)
        return np.log1p(force) - np.log1p(enemy_force)


@lru_cache(maxsize=None)
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
        return CombatPrediction(
            context=context,
            dimensionality=self.dimensionality,
            presence=presence,
            enemy_presence=enemy_presence,
        )

    def combat_presence(self, units: Units) -> CombatPresence:
        count_map = self.mediator.get_map_data_object.get_clean_air_grid(0)
        dps_map = self.mediator.get_map_data_object.get_pyastar_grid(0)
        health_map = self.mediator.get_map_data_object.get_pyastar_grid(0)
        for unit in units:
            dps = self.dps_fast(unit.type_id)
            if 0 < dps:
                position = unit.position
                radius = unit.sight_range
                dps_map = self.mediator.get_map_data_object.add_cost(
                    position=position,
                    radius=radius,
                    grid=dps_map,
                    weight=dps,
                    safe=False,
                )
                health_map = self.mediator.get_map_data_object.add_cost(
                    position=position,
                    radius=radius,
                    grid=health_map,
                    weight=unit.shield + unit.health,
                    safe=False,
                )
                count_map = self.mediator.get_map_data_object.add_cost(
                    position=position,
                    radius=radius,
                    grid=count_map,
                    weight=1.0,
                    safe=False,
                )

        dps_map /= np.maximum(1.0, count_map)

        return CombatPresence(dps_map, health_map)

    @cached_property
    def dimensionality(self) -> np.ndarray:
        local_dimensionality = self.game_info.pathing_grid.data_numpy.T.astype(float)
        dimensionality = scipy.ndimage.gaussian_filter(local_dimensionality, sigma=5.0) ** 2
        dimensionality = np.clip(1 + dimensionality, 1, 2)
        return dimensionality