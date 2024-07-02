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
    confidence: np.ndarray
    presence: CombatPresence
    enemy_presence: CombatPresence


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
    dimensionality_local = np.where(pathing == np.inf, 1.0, 2.0)
    dimensionality_filtered = ndimage.gaussian_filter(dimensionality_local, sigma=5.0)
    return dimensionality_filtered


def predict_combat(context: CombatContext) -> CombatPrediction:
    presence = _combat_presence(context, context.units)
    enemy_presence = _combat_presence(context, context.enemy_units)
    dimensionality = _dimensionality(context.pathing)

    force = presence.dps * np.power(presence.health, dimensionality)
    enemy_force = enemy_presence.dps * np.power(enemy_presence.health, dimensionality)
    confidence = np.log1p(force) - np.log1p(enemy_force)

    return CombatPrediction(
        context=context,
        dimensionality=dimensionality,
        confidence=confidence,
        presence=presence,
        enemy_presence=enemy_presence,
    )
