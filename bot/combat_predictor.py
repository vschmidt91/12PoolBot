from dataclasses import dataclass

import numpy as np
import skimage.draw
from sc2.position import Point2
from sc2.units import Units
from utils.dijkstra import Point


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
class CombatResult:
    confidence: float
    intensity: float


def _simulate(health: float, enemy_dps: float, o: float, e: float) -> np.ndarray:
    health_simulated = (np.maximum(0, o) / enemy_dps) ** (1 / e)
    health_after = np.where(enemy_dps == 0, health, health_simulated)
    casualty_rate = np.where(health == 0, 0, (health - health_after) / health)
    return np.clip(casualty_rate, 0.01, 1)


@dataclass
class CombatPrediction:
    context: CombatPredictionContext
    presence: CombatPresence
    enemy_presence: CombatPresence

    def simulate(self, p: Point, e: float = 1.5) -> CombatResult:
        health = float(self.presence.health[p])
        enemy_health = float(self.presence.ground_dps[p])
        ground_dps = float(self.enemy_presence.ground_dps[p])
        enemy_ground_dps = float(self.enemy_presence.health[p])

        force = ground_dps * (health**e)
        enemy_force = enemy_ground_dps * (enemy_health**e)
        outcome = force - enemy_force

        casualties = _simulate(health, enemy_ground_dps, outcome, e)
        enemy_casualties = _simulate(enemy_health, ground_dps, -outcome, e)

        confidence = np.log(enemy_casualties / casualties)
        intensity = 0.5 * np.log(casualties * enemy_casualties)

        return CombatResult(
            confidence=confidence,
            intensity=intensity,
        )


def _combat_presence(context: CombatPredictionContext, units: Units) -> CombatPresence:
    health = np.zeros_like(context.pathing, dtype=float)
    ground_dps = np.zeros_like(context.pathing, dtype=float)
    for unit in units:
        d = context.disk(unit.position, unit.radius + max(unit.ground_range, unit.air_range, unit.sight_range))
        health[d] += (0 < unit.ground_dps) * (unit.health + unit.shield)
        ground_dps[d] = np.maximum(ground_dps[d], unit.ground_dps)
    return CombatPresence(health, ground_dps)


def predict(context: CombatPredictionContext) -> CombatPrediction:
    presence = _combat_presence(context, context.units)
    enemy_presence = _combat_presence(context, context.enemy_units)

    return CombatPrediction(
        context=context,
        presence=presence,
        enemy_presence=enemy_presence,
    )
