from typing import Iterable
from dataclasses import dataclass

import numpy as np
import skimage.draw
from sc2.unit import Unit
from sc2.position import Point2


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
    # civilian_presence: np.ndarray
    combat_outcome: np.ndarray
    confidence: np.ndarray


def _civilian_presence(context: CombatPredictionContext) -> np.ndarray:
    civilian_presence = np.zeros_like(context.pathing, dtype=float)
    for civilian in context.civilians:
        d = context.disk(civilian.position, civilian.radius)
        s = -1.0 if civilian.is_enemy else +1.0
        civilian_presence[d] += s * civilian.shield_health_percentage
    return civilian_presence


def _combat_presence(context: CombatPredictionContext) -> CombatPresence:
    force = np.ones_like(context.pathing, dtype=float)
    enemy_force = np.ones_like(context.pathing, dtype=float)
    for unit in context.combatants:
        d = context.disk(unit.position, unit.radius + unit.sight_range)
        m = enemy_force if unit.is_enemy else force
        m[d] += unit.ground_dps * ((unit.health + unit.shield) ** 1.5)
    return CombatPresence(force, enemy_force)


def predict(context: CombatPredictionContext) -> CombatPrediction:
    combat_presence = _combat_presence(context)
    # civilian_presence = _civilian_presence(context)
    combat_outcome = combat_presence.force - combat_presence.enemy_force
    confidence = combat_outcome / (combat_presence.force + combat_presence.enemy_force)
    return CombatPrediction(
        # civilian_presence=civilian_presence,
        combat_outcome=combat_outcome,
        confidence=confidence,
    )
