from typing import Iterable
from dataclasses import dataclass

import numpy as np
import skimage.draw
from sc2.unit import Unit

from .extensions.poisson_multigrid import mg_opt, convolve_jacobi


@dataclass
class CombatPredictionContext:
    pathing: np.ndarray
    civilians: Iterable[Unit]
    combatants: Iterable[Unit]


@dataclass
class CombatPresence:
    force: np.ndarray
    enemy_force: np.ndarray


@dataclass
class CombatPrediction:
    civilian_presence: np.ndarray
    combat_outcome: np.ndarray
    confidence: np.ndarray
    retreat_potential: np.ndarray


class CombatPredictor:
    _retreat_potential: np.ndarray | None = None

    def predict(self, context: CombatPredictionContext) -> CombatPrediction:
        combat_presence = self.combat_presence(context)
        civilian_presence = self.civilian_presence(context)
        combat_outcome = combat_presence.force - combat_presence.enemy_force
        confidence = combat_outcome / (combat_presence.force + combat_presence.enemy_force)
        if self._retreat_potential is None:
            self._retreat_potential = np.zeros_like(context.pathing, dtype=float)

        rhs = civilian_presence
        b = context.pathing
        b *= convolve_jacobi(b) > 0
        self._retreat_potential = mg_opt(self._retreat_potential, rhs, b)

        return CombatPrediction(civilian_presence, combat_outcome, confidence, self._retreat_potential)

    def civilian_presence(self, context: CombatPredictionContext) -> np.ndarray:
        civilian_presence = np.zeros_like(context.pathing, dtype=float)
        for civilian in context.civilians:
            d = skimage.draw.disk(
                center=civilian.position,
                radius=max(1.0, civilian.radius),
                shape=context.pathing.shape,
            )
            civilian_presence[d] += (-1.0 if civilian.is_enemy else +1.0) * civilian.shield_health_percentage

        return civilian_presence

    def combat_presence(self, context: CombatPredictionContext) -> CombatPresence:

        def draw(u, r): return skimage.draw.disk(center=u.position, radius=r, shape=context.pathing.shape)

        force = np.ones_like(context.pathing, dtype=float)
        enemy_force = np.ones_like(context.pathing, dtype=float)

        for unit in context.combatants:

            d = draw(unit, unit.radius + unit.sight_range)
            m = enemy_force if unit.is_enemy else force
            m[d] += unit.ground_dps * (unit.health + unit.shield)

        return CombatPresence(force, enemy_force)
