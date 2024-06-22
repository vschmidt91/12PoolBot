import functools

from ares import AresBot
from sc2.ids.unit_typeid import UnitTypeId


class Component(AresBot):
    @functools.lru_cache(maxsize=None)
    def dps_fast(self, unit: UnitTypeId) -> float:
        if units := self.all_units(unit):
            return max(units[0].ground_dps, units[0].air_dps)
        else:
            return 0.0
