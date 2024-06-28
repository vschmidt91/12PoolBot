from typing import Awaitable, Callable

ChatFunction = Callable[[str], Awaitable]


class Tags:
    chat_function: ChatFunction
    _tags: set[str] = set()

    def __init__(self, chat_function: ChatFunction) -> None:
        self.chat_function = chat_function

    async def add_tag(self, tag: str) -> bool:
        if tag in self._tags:
            return False
        message = f"Tag:{tag}"
        await self.chat_function(message)
        self._tags.add(tag)
        return True
