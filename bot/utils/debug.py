import lzma
import os
import pickle

from sc2.game_info import GameInfo


def save_map(game_info: GameInfo, output_dir: str) -> None:
    output_path = os.path.join(output_dir, f"{game_info.map_name}.xz")
    with lzma.open(output_path, "wb") as f:
        pickle.dump(game_info, f)
