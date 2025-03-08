"""Common utilities for tests."""

from typing import Any
from unittest.mock import MagicMock

from lib.TranslationProject import TranslationProject


class AsyncMock(MagicMock):
    """Async version of MagicMock to mock async functions."""

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super(AsyncMock, self).__call__(*args, **kwargs)
