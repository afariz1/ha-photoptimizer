"""Abstract interface for inverter-specific control adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import ExecutionSlotCommand


class InverterControlAdapter(ABC):
    """Abstract interface for inverter-specific control adapters.

    All concrete inverter adapters (GoodWe, Growatt, etc.) must implement
    this interface to be compatible with the universal executor.
    """

    @abstractmethod
    async def async_apply(self, command: ExecutionSlotCommand) -> None:
        """Apply one normalized battery command to inverter entities/services.

        Args:
            command: Normalized battery operation command from optimizer.

        Raises:
            Any exceptions from async service calls (Home Assistant integration).
        """
