from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StagePlugin(ABC):
    name: str = ""
    requires: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()

    @abstractmethod
    def run(self, context: Any) -> dict[str, Any]:
        raise NotImplementedError
