"""Factory for creating inverter control adapters from configuration."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import (
    CONF_BATTERY_CHARGE_POWER_MAX,
    CONF_BATTERY_DISCHARGE_POWER_MAX,
    CONF_CONNECT_INVERTER_ENTITIES,
    CONF_GROWATT_AC_CHARGE_SWITCH_ENTITY,
    CONF_GROWATT_DEVICE_ID,
    CONF_GROWATT_INVERTER_VARIANT,
    CONF_INVERTER_CHARGE_POWER_ENTITY,
    CONF_INVERTER_DISCHARGE_POWER_ENTITY,
    CONF_INVERTER_MODE_ENTITY,
    CONF_INVERTER_TYPE,
    DEFAULT_BATTERY_CHARGE_POWER_MAX,
    DEFAULT_BATTERY_DISCHARGE_POWER_MAX,
    GROWATT_VARIANT_AUTO,
    INVERTER_TYPE_GOODWE,
    INVERTER_TYPE_GROWATT,
)
from .goodwe_control import GoodweControlAdapter
from .growatt_control import GrowattControlAdapter
from .inverter_interface import InverterControlAdapter
from .models import ExecutionSlotCommand

_LOGGER = logging.getLogger(__name__)


class NoopControlAdapter(InverterControlAdapter):
    """Safe fallback adapter when inverter control is not configured."""

    async def async_apply(self, command: ExecutionSlotCommand) -> None:
        """Log command without applying any control."""
        _LOGGER.info(
            "Command-only inverter action: mode=%s power_w=%s soc_target=%s grid_limit=%s",
            command.op_mode.value,
            command.p_bat_cmd,
            command.soc_target,
            command.grid_limit,
        )


def create_inverter_adapter(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> InverterControlAdapter:
    """Create an inverter control adapter for the given config entry.

    Args:
        hass: Home Assistant instance.
        entry: Configuration entry for photoptimizer integration.

    Returns:
        Concrete adapter for the configured inverter type, or NoopControlAdapter
        if the type is unsupported.
    """
    inverter_type = entry.data.get(CONF_INVERTER_TYPE)
    if not entry.data.get(CONF_CONNECT_INVERTER_ENTITIES, True):
        _LOGGER.info(
            "Inverter entity connection disabled; running in command-only mode for type '%s'",
            inverter_type,
        )
        return NoopControlAdapter()

    mode_entity = entry.data.get(CONF_INVERTER_MODE_ENTITY)
    charge_entity = entry.data.get(CONF_INVERTER_CHARGE_POWER_ENTITY)
    discharge_entity = entry.data.get(CONF_INVERTER_DISCHARGE_POWER_ENTITY)
    max_charge_w = float(
        entry.data.get(
            CONF_BATTERY_CHARGE_POWER_MAX,
            DEFAULT_BATTERY_CHARGE_POWER_MAX,
        )
    )
    max_discharge_w = float(
        entry.data.get(
            CONF_BATTERY_DISCHARGE_POWER_MAX,
            DEFAULT_BATTERY_DISCHARGE_POWER_MAX,
        )
    )

    if inverter_type == INVERTER_TYPE_GROWATT:
        return GrowattControlAdapter(
            hass,
            mode_entity_id=mode_entity,
            charge_power_entity_id=charge_entity,
            discharge_power_entity_id=discharge_entity,
            ac_charge_switch_entity_id=entry.data.get(
                CONF_GROWATT_AC_CHARGE_SWITCH_ENTITY
            ),
            growatt_device_id=entry.data.get(CONF_GROWATT_DEVICE_ID),
            growatt_variant=entry.data.get(
                CONF_GROWATT_INVERTER_VARIANT,
                GROWATT_VARIANT_AUTO,
            ),
            max_charge_power_w=max_charge_w,
            max_discharge_power_w=max_discharge_w,
        )

    if inverter_type == INVERTER_TYPE_GOODWE:
        return GoodweControlAdapter(
            hass,
            mode_entity_id=mode_entity,
            charge_power_entity_id=charge_entity,
            discharge_power_entity_id=discharge_entity,
            max_charge_power_w=max_charge_w,
            max_discharge_power_w=max_discharge_w,
        )

    _LOGGER.warning(
        "Inverter adapter not found for type '%s'; using noop fallback",
        inverter_type,
    )
    return NoopControlAdapter()
