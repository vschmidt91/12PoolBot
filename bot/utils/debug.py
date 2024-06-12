from dataclasses import dataclass, asdict
import os
import pickle

import numpy as np

from sc2.game_info import GameInfo


@dataclass
class GameInfoDebug:
    pathing: np.ndarray
    placement: np.ndarray
    game_area: np.ndarray
    terrain_height: np.ndarray
    vision_blockers: np.ndarray


def save_map(game_info: GameInfo, output_dir: str) -> None:
    data = GameInfoDebug(
        pathing=game_info.pathing_grid.data_numpy,
        placement=game_info.placement_grid.data_numpy,
        game_area=np.array(game_info.playable_area),
        terrain_height=game_info.terrain_height.data_numpy,
        vision_blockers=np.array(list(game_info.vision_blockers)),
    )
    output_path = os.path.join(output_dir, f"{game_info.map_name}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(game_info, f)
