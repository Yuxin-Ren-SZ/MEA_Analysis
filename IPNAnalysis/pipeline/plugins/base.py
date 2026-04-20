from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AnalysisPlugin(ABC):
    name: str = ""

    @abstractmethod
    def run(self, context: Any, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        raise NotImplementedError
