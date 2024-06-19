from dataclasses import dataclass

import numpy as np
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2

from .component import Component


class Tags(Component):
    _tags: set[str] = set()

    async def add_tag(self, tag: str) -> bool:
        if tag in self._tags:
            return False
        message = f"Tag:{tag}"
        await self.chat_send(message)
        return True
