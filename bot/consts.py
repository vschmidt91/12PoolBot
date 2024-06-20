from ares.consts import CHANGELING_TYPES
from sc2.constants import WORKER_TYPES
from sc2.ids.unit_typeid import UnitTypeId

VERSION_FILE: str = "version.txt"
UNKNOWN_VERSION: str = "unknown_version"
TAG_MICRO_THROTTLING: str = "micro_throttling"
TAG_ACTION_FAILED: str = "action_failed"
EXCLUDE_FROM_COMBAT = WORKER_TYPES | CHANGELING_TYPES | {UnitTypeId.LARVA, UnitTypeId.EGG}
PROFILING_FILE = "profiling"
