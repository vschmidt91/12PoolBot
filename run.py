import random
import sys
from os import path
from pathlib import Path
from typing import List

from sc2 import maps
from sc2.data import AIBuild, Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer

sys.path.append("ares-sc2/src/ares")
sys.path.append("ares-sc2/src")
sys.path.append("ares-sc2")

import yaml

from bot.main import TwelvePoolBot
from ladder import run_ladder_game

# change if non default setup / linux
# if having issues with this, modify `map_list` below manually
MAPS_PATH: str = "C:\\Program Files (x86)\\StarCraft II\\Maps"
CONFIG_FILE: str = "config.yml"
MAP_FILE_EXT: str = "SC2Map"
MY_BOT_NAME: str = "MyBotName"
MY_BOT_RACE: str = "MyBotRace"
MAP_VETOS: list[str] = [
    "Tier1MicroAIArena_v4",
    "Tier2MicroAIArena_v4",
]


def main():
    bot_name: str = "MyBot"
    race: Race = Race.Random

    __user_config_location__: str = path.abspath(".")
    user_config_path: str = path.join(__user_config_location__, CONFIG_FILE)
    # attempt to get race and bot name from config file if they exist
    if path.isfile(user_config_path):
        with open(user_config_path) as config_file:
            config: dict = yaml.safe_load(config_file)
            if MY_BOT_NAME in config:
                bot_name = config[MY_BOT_NAME]
            if MY_BOT_RACE in config:
                race = Race[config[MY_BOT_RACE].title()]

    bot1 = Bot(race, TwelvePoolBot(), bot_name)

    if "--LadderServer" in sys.argv:
        # Ladder game started by LadderManager
        print("Starting ladder game...")
        result, opponentid = run_ladder_game(bot1)
        print(result, " against opponent ", opponentid)
    else:
        # Local game
        map_set: set[str] = {
            p.name.replace(f".{MAP_FILE_EXT}", "")
            for p in Path(MAPS_PATH).glob(f"*.{MAP_FILE_EXT}")
            if p.is_file()
        }
        map_set = {m for m in map_set if m not in MAP_VETOS}
        map_list = list(map_set)

        random_race = random.choice([
            Race.Zerg,
            Race.Terran,
            Race.Protoss,
        ])
        print("Starting local game...")
        run_game(
            maps.get(random.choice(map_list)),
            [
                bot1,
                Computer(random_race, Difficulty.CheatInsane, ai_build=AIBuild.Rush),
            ],
            realtime=False,
        )


# Start game
if __name__ == "__main__":
    main()
