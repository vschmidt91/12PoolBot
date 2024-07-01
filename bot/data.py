from collections import Counter
from dataclasses import dataclass

import torch
from sc2.bot_ai import BotAI
from sc2.data import Result
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId

from .consts import ALL_UNITS

RESULT_VALUES: dict[Result, int] = {
    Result.Victory: +1,
    Result.Tie: 0,
    Result.Undecided: 0,
    Result.Defeat: -1,
}


@dataclass
class GameResult:
    result: Result

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([RESULT_VALUES[self.result]], dtype=torch.float)


@dataclass
class GameState:
    unit_counts: dict[UnitTypeId, int]
    enemy_unit_counts: dict[UnitTypeId, int]
    upgrades: set[UpgradeId]
    score: dict[str, float]
    time: float
    visibility: float
    creep: float

    @classmethod
    def from_bot(cls, bot: BotAI) -> "GameState":
        unit_counts = Counter(u.type_id for u in bot.units)
        enemy_unit_counts = Counter(u.type_id for u in bot.enemy_units)
        upgrades = bot.state.upgrades
        score = {str(k): float(v) for k, v in bot.state.score.summary}
        visibility = bot.state.visibility.data_numpy.astype(float).mean()
        creep = bot.state.creep.data_numpy.astype(float).mean()
        time = bot.time
        return GameState(
            unit_counts=unit_counts,
            enemy_unit_counts=enemy_unit_counts,
            upgrades=upgrades,
            score=score,
            visibility=visibility,
            creep=creep,
            time=time,
        )

    def to_tensor(self) -> torch.Tensor:
        units = torch.tensor([self.unit_counts[u] for u in ALL_UNITS], dtype=torch.float)
        enemy_units = torch.tensor([self.enemy_unit_counts[u] for u in ALL_UNITS], dtype=torch.float)
        scalars = torch.tensor(
            [
                self.visibility,
                self.creep,
                self.time,
            ],
            dtype=torch.float,
        )
        upgrades = torch.tensor([1 if u in self.upgrades else 0 for u in list(UpgradeId)])
        score = torch.tensor(list(self.score.values()), dtype=torch.float)
        tensor = torch.concatenate(
            [
                units,
                enemy_units,
                upgrades,
                score,
                scalars,
            ]
        )
        return tensor


@dataclass
class GameReplay:
    states: list[GameState]
    result: GameResult
