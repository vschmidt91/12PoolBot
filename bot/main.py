from ares import AresBot

from itertools import chain

from components.strategy import Strategy
from components.micro import Micro
from components.macro import Macro


class TwelvePoolBot(Strategy, Micro, Macro, AresBot):
    async def on_step(self, iteration: int) -> None:
        await super().on_step(iteration)

        strategy = self.decide_strategy()
        actions = chain(
            self.macro(strategy),
            self.micro(strategy),
        )
        for action in actions:
            success = await action.execute(self)
            if not success:
                raise Exception(f"Action failed: {action}")
