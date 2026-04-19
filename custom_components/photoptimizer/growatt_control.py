"""Growatt control adapter for Photoptimizer executor."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging

from homeassistant.core import HomeAssistant

from .const import GROWATT_VARIANT_MIN, GROWATT_VARIANT_SPH
from .inverter_interface import InverterControlAdapter
from .models import ExecutionSlotCommand, OperationMode

_LOGGER = logging.getLogger(__name__)


class GrowattControlAdapter(InverterControlAdapter):
    """Apply normalized battery commands to Growatt entities/services."""

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        mode_entity_id: str | None,
        charge_power_entity_id: str | None,
        discharge_power_entity_id: str | None,
        ac_charge_switch_entity_id: str | None,
        growatt_device_id: str | None,
        growatt_variant: str,
        max_charge_power_w: float,
        max_discharge_power_w: float,
    ) -> None:
        """Initialize Growatt adapter with optional service targeting."""
        self._hass = hass
        self._mode_entity_id = mode_entity_id
        self._charge_power_entity_id = charge_power_entity_id
        self._discharge_power_entity_id = discharge_power_entity_id
        self._ac_charge_switch_entity_id = ac_charge_switch_entity_id
        self._growatt_device_id = growatt_device_id
        self._growatt_variant = growatt_variant
        self._max_charge_power_w = max(1.0, max_charge_power_w)
        self._max_discharge_power_w = max(1.0, max_discharge_power_w)

    async def async_apply(self, command: ExecutionSlotCommand) -> None:
        """Apply one battery command to Growatt entities/services."""
        if command.op_mode == OperationMode.FORCED_DISCHARGE:
            percent = self._to_percent(command.p_bat_cmd, self._max_discharge_power_w)
            await self._async_set_number(self._discharge_power_entity_id, percent)
            await self._async_set_mode("eco_discharge")
            await self._async_set_ac_charge(False)
            await self._async_call_variant_service(discharge_percent=percent)
            return

        if command.op_mode == OperationMode.FORCED_CHARGE:
            if not self._charge_power_entity_id:
                _LOGGER.warning(
                    "Growatt charge command skipped: charge power entity is not configured"
                )
                await self._async_set_mode("general")
                return

            percent = self._to_percent(abs(command.p_bat_cmd), self._max_charge_power_w)
            await self._async_set_number(self._charge_power_entity_id, percent)
            await self._async_set_mode("eco_charge")
            await self._async_set_ac_charge(True)
            await self._async_call_variant_service(charge_percent=percent)
            return

        await self._async_set_mode("general")
        await self._async_set_number(self._discharge_power_entity_id, 0)
        await self._async_set_number(self._charge_power_entity_id, 0)
        await self._async_set_ac_charge(False)

    async def _async_call_variant_service(
        self,
        *,
        charge_percent: int | None = None,
        discharge_percent: int | None = None,
    ) -> None:
        if not self._growatt_device_id:
            return

        now = datetime.now().replace(second=0, microsecond=0)
        end = now + timedelta(minutes=5)

        if self._growatt_variant == GROWATT_VARIANT_MIN:
            batt_mode = "load_first"
            if discharge_percent and discharge_percent > 0:
                batt_mode = "grid_first"
            elif charge_percent and charge_percent > 0:
                batt_mode = "battery_first"

            await self._hass.services.async_call(
                "growatt_server",
                "update_time_segment",
                {
                    "device_id": self._growatt_device_id,
                    "segment_id": 1,
                    "batt_mode": batt_mode,
                    "start_time": now.strftime("%H:%M"),
                    "end_time": end.strftime("%H:%M"),
                    "enabled": True,
                },
                blocking=True,
            )
            return

        if self._growatt_variant == GROWATT_VARIANT_SPH:
            if charge_percent is not None:
                await self._hass.services.async_call(
                    "growatt_server",
                    "write_ac_charge_times",
                    {
                        "device_id": self._growatt_device_id,
                        "charge_power": charge_percent,
                        "period_1_start": now.strftime("%H:%M"),
                        "period_1_end": end.strftime("%H:%M"),
                        "period_1_enabled": charge_percent > 0,
                    },
                    blocking=True,
                )

            if discharge_percent is not None:
                await self._hass.services.async_call(
                    "growatt_server",
                    "write_ac_discharge_times",
                    {
                        "device_id": self._growatt_device_id,
                        "discharge_power": discharge_percent,
                        "period_1_start": now.strftime("%H:%M"),
                        "period_1_end": end.strftime("%H:%M"),
                        "period_1_enabled": discharge_percent > 0,
                    },
                    blocking=True,
                )

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

    async def _async_set_ac_charge(self, enabled: bool) -> None:
        if not self._ac_charge_switch_entity_id:
            return

        service = "turn_on" if enabled else "turn_off"
        await self._hass.services.async_call(
            "switch",
            service,
            {
                "entity_id": self._ac_charge_switch_entity_id,
            },
            blocking=True,
        )

    def _to_percent(self, power_w: float, max_power_w: float) -> int:
        normalized = max(0.0, min(100.0, (float(power_w) / max_power_w) * 100.0))
        return int(round(normalized))
