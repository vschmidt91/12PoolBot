from ares.consts import ALL_STRUCTURES, CHANGELING_TYPES
from sc2.constants import WORKER_TYPES, abilityid_to_unittypeid
from sc2.ids.unit_typeid import UnitTypeId

VERSION_FILE: str = "version.txt"
UNKNOWN_VERSION: str = "unknown_version"
TAG_MICRO_THROTTLING: str = "micro_throttling"
TAG_ACTION_FAILED: str = "action_failed"
ALL_UNITS = ALL_STRUCTURES | set(abilityid_to_unittypeid.values())
EXCLUDE_FROM_COMBAT = WORKER_TYPES | CHANGELING_TYPES | {UnitTypeId.LARVA, UnitTypeId.EGG}
PROFILING_FILE = "profiling"

DPS_OVERRIDE = {
    UnitTypeId.BUNKER: 40,
}
