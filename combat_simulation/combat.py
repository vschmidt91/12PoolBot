from dataclasses import dataclass

from sc2.ids.unit_typeid import UnitTypeId
from sc2.unit import Unit


@dataclass
class CombatUnit:
    unit: UnitTypeId
    health: float
    shield: float
    ground_dps: float
    air_dps: float
    ground_range: float
    air_range: float

    @classmethod
    def from_unit(cls, unit: Unit) -> "CombatUnit":
        return CombatUnit(
            unit=unit.type_id,
            health=unit.health,
            shield=unit.shield,
            ground_dps=unit.ground_dps,
            air_dps=unit.air_dps,
            ground_range=unit.ground_range,
            air_range=unit.air_range,
        )


@dataclass
class CombatSetup:
    units: list[CombatUnit]
    enemy_units: list[CombatUnit]
    health: float
    enemy_health: float


@dataclass
class CombatOutcome:
    win: bool
    winner_health: float
    result: float


@dataclass
class Combat:
    setup: CombatSetup
    outcome: CombatOutcome


@dataclass
class CombatDataset:
    unit_types: list[UnitTypeId]
    combats: list[Combat]
