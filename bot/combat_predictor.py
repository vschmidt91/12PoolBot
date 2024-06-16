from dataclasses import dataclass
from typing import Iterable

import numpy as np
import skimage.draw
from sc2.position import Point2
from sc2.unit import Unit


@dataclass
class CombatPredictionContext:
    pathing: np.ndarray
    civilians: Iterable[Unit]
    combatants: Iterable[Unit]

    def disk(self, center: Point2, radius: float):
        return skimage.draw.disk(center=center, radius=radius, shape=self.pathing.shape)


@dataclass
class CombatPresence:
    force: np.ndarray
    enemy_force: np.ndarray


@dataclass
class CombatPrediction:
    presence: CombatPresence
    combat_outcome: np.ndarray
    confidence: np.ndarray


def _civilian_presence(context: CombatPredictionContext) -> np.ndarray:
    civilian_presence = np.zeros_like(context.pathing, dtype=float)
    for civilian in context.civilians:
        d = context.disk(civilian.position, civilian.radius)
        s = -1.0 if civilian.is_enemy else +1.0
        civilian_presence[d] += s * civilian.shield_health_percentage
    return civilian_presence


def _combat_presence(context: CombatPredictionContext, lancester_power: float = 1.0) -> CombatPresence:
    force = np.zeros_like(context.pathing, dtype=float)
    enemy_force = np.zeros_like(context.pathing, dtype=float)
    for unit in context.combatants:
        d = context.disk(unit.position, unit.radius + max(unit.ground_range, unit.air_range, unit.sight_range))
        m = enemy_force if unit.is_enemy else force
        m[d] += unit.ground_dps * ((unit.health + unit.shield) ** lancester_power)
    return CombatPresence(force, enemy_force)


def predict(context: CombatPredictionContext) -> CombatPrediction:
    combat_presence = _combat_presence(context)
    combat_outcome = combat_presence.force - combat_presence.enemy_force
    confidence = combat_outcome / np.maximum(combat_presence.force, combat_presence.enemy_force)
    return CombatPrediction(
        combat_outcome=combat_outcome,
        confidence=confidence,
        presence=combat_presence,
    )
