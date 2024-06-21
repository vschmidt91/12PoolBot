from dataclasses import dataclass

import numpy as np
import skimage.draw
from ares.dicts.weight_costs import WEIGHT_COSTS
from sc2.position import Point2
from sc2.units import Units
from scipy import ndimage

from .utils.dijkstra import Point


@dataclass
class CombatPredictionContext:
    pathing: np.ndarray
    units: Units
    enemy_units: Units

    def disk(self, center: Point2, radius: float):
        return skimage.draw.disk(center=center, radius=radius)


def _simulate(health: float, enemy_dps: float, o: float, e: float) -> np.ndarray:
    health_simulated = (np.maximum(0, o) / enemy_dps) ** (1 / e)
    health_after = np.where(enemy_dps == 0, health, health_simulated)
    casualty_rate = np.where(health == 0, 0, (health - health_after) / health)
    return np.clip(casualty_rate, 0.01, 1)


@dataclass
class CombatPrediction:
    context: CombatPredictionContext
    presence: np.ndarray
    enemy_presence: np.ndarray
    confidence: np.ndarray


def _combat_presence(context: CombatPredictionContext, units: Units) -> np.ndarray:
    force = np.zeros_like(context.pathing, dtype=float)
    for unit in units:
        if unit.ground_dps == 0:
            continue
        d = context.disk(unit.position, max(unit.ground_range, unit.air_range, unit.sight_range))
        force[d] += unit.ground_dps * (unit.health + unit.shield)
    return force


def predict(context: CombatPredictionContext) -> CombatPrediction:
    presence = _combat_presence(context, context.units)
    enemy_presence = _combat_presence(context, context.enemy_units)
    confidence = (presence - enemy_presence) / np.maximum(presence, enemy_presence)
    return CombatPrediction(
        context=context,
        presence=presence,
        enemy_presence=enemy_presence,
        confidence=confidence,
    )
