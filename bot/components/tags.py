from .component import Component


class Tags(Component):
    _tags: set[str] = set()

    async def add_tag(self, tag: str) -> bool:
        if tag in self._tags:
            return False
        message = f"Tag:{tag}"
        await self.chat_send(message)
        return True
