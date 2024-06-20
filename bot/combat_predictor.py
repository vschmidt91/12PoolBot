from dataclasses import dataclass

import numpy as np
import skimage.draw
from sc2.position import Point2
from sc2.units import Units


@dataclass
class CombatPredictionContext:
    pathing: np.ndarray
    units: Units
    enemy_units: Units

    def disk(self, center: Point2, radius: float):
        return skimage.draw.disk(center=center, radius=radius, shape=self.pathing.shape)


@dataclass
class CombatPresence:
    health: np.ndarray
    ground_dps: np.ndarray


@dataclass
class CombatPrediction:
    context: CombatPredictionContext
    presence: CombatPresence
    enemy_presence: CombatPresence
    bitterness: np.ndarray
    intensity: np.ndarray


def _combat_presence(context: CombatPredictionContext, units: Units) -> CombatPresence:
    health = np.zeros_like(context.pathing, dtype=float)
    ground_dps = np.zeros_like(context.pathing, dtype=float)
    for unit in units:
        d = context.disk(unit.position, unit.radius + max(unit.ground_range, unit.air_range, unit.sight_range))
        health[d] += (0 < unit.ground_dps) * (unit.health + unit.shield)
        ground_dps[d] = np.maximum(ground_dps[d], unit.ground_dps)
    return CombatPresence(health, ground_dps)


def predict(context: CombatPredictionContext) -> CombatPrediction:
    e = 1.5

    def force_from(p: CombatPresence) -> np.ndarray:
        return p.ground_dps * (p.health**e)

    def simulate(p: CombatPresence, ep: CombatPresence, o: np.ndarray) -> np.ndarray:
        health_simulated = (np.maximum(0, o) / ep.ground_dps) ** (1 / e)
        health_after = np.where(ep.ground_dps == 0, p.health, health_simulated)
        casualty_rate = np.where(p.health == 0, 0, (p.health - health_after) / p.health)
        return np.clip(casualty_rate, 0.01, 1)

    presence = _combat_presence(context, context.units)
    enemy_presence = _combat_presence(context, context.enemy_units)
    force = force_from(presence)
    enemy_force = force_from(enemy_presence)
    outcome = force - enemy_force

    casualties = simulate(presence, enemy_presence, outcome)
    enemy_casualties = simulate(enemy_presence, presence, -outcome)

    bitterness = np.log(enemy_casualties / casualties)
    intensity = 0.5 * np.log(casualties * enemy_casualties)

    return CombatPrediction(
        context=context,
        presence=presence,
        enemy_presence=enemy_presence,
        bitterness=bitterness,
        intensity=intensity,
    )
