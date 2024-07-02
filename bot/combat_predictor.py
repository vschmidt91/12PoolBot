from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
import skimage.draw
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.units import Units
from scipy import ndimage

DpsProvider = Callable[[UnitTypeId], float]


@dataclass
class CombatContext:
    units: Units
    enemy_units: Units
    dps: DpsProvider
    pathing: np.ndarray


@dataclass
class CombatPresence:
    dps: np.ndarray
    health: np.ndarray


@dataclass
class CombatPrediction:
    context: CombatContext
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
def _disk(radius: float) -> tuple[np.ndarray, np.ndarray]:
    r = int(radius + 0.5)
    p = radius, radius
    n = 2 * r + 1
    dx, dy = skimage.draw.disk(center=p, radius=radius, shape=(n, n))
    return dx - r, dy - r


def _combat_presence(context: CombatContext, units: Units) -> CombatPresence:
    dps_map = np.zeros_like(context.pathing, dtype=float)
    health_map = np.zeros_like(context.pathing, dtype=float)
    for unit in units:
        dps = context.dps(unit.type_id)
        px, py = unit.position.rounded
        if 0 < dps:
            dx, dy = _disk(unit.sight_range)
            d = px + dx, py + dy
            health_map[d] += unit.shield + unit.health
            dps_map[d] = np.maximum(dps_map[d], dps)
    return CombatPresence(dps_map, health_map)


def _dimensionality(pathing: np.ndarray) -> np.ndarray:
    dimensionality_local = pathing.astype(float)
    dimensionality_filtered = ndimage.gaussian_filter(dimensionality_local, sigma=5.0) ** 2
    return np.clip(1 + dimensionality_filtered, 1, 2)


def predict_combat(context: CombatContext) -> CombatPrediction:
    presence = _combat_presence(context, context.units)
    enemy_presence = _combat_presence(context, context.enemy_units)
    dimensionality = _dimensionality(context.pathing)
    return CombatPrediction(
        context=context,
        dimensionality=dimensionality,
        presence=presence,
        enemy_presence=enemy_presence,
    )
