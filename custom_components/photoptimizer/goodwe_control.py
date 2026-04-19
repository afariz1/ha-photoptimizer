"""GoodWe control adapter for Photoptimizer executor."""

from __future__ import annotations

import logging

from homeassistant.core import HomeAssistant

from .inverter_interface import InverterControlAdapter
from .models import ExecutionSlotCommand, OperationMode

_LOGGER = logging.getLogger(__name__)


class GoodweControlAdapter(InverterControlAdapter):
    """Apply normalized battery commands to GoodWe entities/services."""

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        mode_entity_id: str | None,
        charge_power_entity_id: str | None,
        discharge_power_entity_id: str | None,
        max_charge_power_w: float,
        max_discharge_power_w: float,
    ) -> None:
        """Initialize the adapter with selected entity mappings."""
        self._hass = hass
        self._mode_entity_id = mode_entity_id
        self._charge_power_entity_id = charge_power_entity_id
        self._discharge_power_entity_id = discharge_power_entity_id
        self._max_charge_power_w = max(1.0, max_charge_power_w)
        self._max_discharge_power_w = max(1.0, max_discharge_power_w)

    async def async_apply(self, command: ExecutionSlotCommand) -> None:
        """Apply one battery command to GoodWe entities."""
        if command.op_mode == OperationMode.FORCED_DISCHARGE:
            percent = self._to_percent(command.p_bat_cmd, self._max_discharge_power_w)
            await self._async_set_number(self._discharge_power_entity_id, percent)
            await self._async_set_mode("eco_discharge")
            return

        if command.op_mode == OperationMode.FORCED_CHARGE:
            if not self._charge_power_entity_id:
                _LOGGER.warning(
                    "GoodWe charge command skipped: charge power entity is not configured"
                )
                await self._async_set_mode("general")
                return

            percent = self._to_percent(abs(command.p_bat_cmd), self._max_charge_power_w)
            await self._async_set_number(self._charge_power_entity_id, percent)
            await self._async_set_mode("eco_charge")
            return

        await self._async_set_mode("general")
        await self._async_set_number(self._discharge_power_entity_id, 0)
        await self._async_set_number(self._charge_power_entity_id, 0)

    async def _async_set_mode(self, option: str) -> None:
        if not self._mode_entity_id:
            return

        await self._hass.services.async_call(
            "select",
            "select_option",
            {
                "entity_id": self._mode_entity_id,
                "option": option,
            },
            blocking=True,
        )

    async def _async_set_number(self, entity_id: str | None, value: int) -> None:
        if not entity_id:
            return

        await self._hass.services.async_call(
            "number",
            "set_value",
            {
                "entity_id": entity_id,
                "value": value,
            },
            blocking=True,
        )

    def _to_percent(self, power_w: float, max_power_w: float) -> int:
        normalized = max(0.0, min(100.0, (float(power_w) / max_power_w) * 100.0))
        return int(round(normalized))
