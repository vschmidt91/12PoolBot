from typing import TYPE_CHECKING, Union, Optional

import numpy as np
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

def cy_pylon_matrix_covers(
    position: Union[Point2, tuple[float, float]],
    pylons: Union[Units, list[Unit]],
    height_grid: np.ndarray,
    pylon_build_progress: Optional[float] = 1.0,
) -> bool:
    """Check if a position is powered by a pylon.

    Example:
    ```py
    from cython_functions import cy_pylon_matrix_covers
    from sc2.position import Point2

    # check if start location is powered by pylon
    position: Point2 = self.start_location

    can_place_structure_here: bool = cy_pylon_matrix_covers(
        position,
        self.structures(UnitTypeId.PYLON),
        self.game_info.terrain_height.data_numpy
    )
    ```

    ```
    1.85 µs ± 8.72 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    ```

    Args:
        position: Position to check for power.
        pylons: The pylons we want to check.
        height_grid: Height grid supplied from `python-sc2` as a numpy array.
        pylon_build_progress: If less than 1.0, check near pending pylons.
            Default is 1.0.

    Returns:
        True if `position` has power, False otherwise.

    """

def cy_unit_pending(ai: "BotAI", unit_type: UnitID) -> int:
    """Check how many unit_type are pending.

    Faster unit specific alternative to `python-sc2`'s `already_pending`

    Example:
    ```py
    from cython_functions import cy_unit_pending
    from sc2.ids.unit_typeid import UnitTypeId

    num_marines_pending: int = cy_unit_pending(UnitTypeId.MARINE)
    ```
    ```
    453 ns ± 9.35 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)

    Python-sc2 `already_pending` alternative:
    2.82 µs ± 29 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    ```

    Args:
        ai: Bot object that will be running the game.
        unit_type: Unit type we want to check.

    Returns:
        How many unit_type are currently building.


    """
    ...
