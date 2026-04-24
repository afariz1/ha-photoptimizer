"""Switch platform for Photoptimizer integration."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import PhotoptimizerCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Photoptimizer switch entities."""
    coordinator: PhotoptimizerCoordinator = hass.data[DOMAIN][entry.entry_id]
    _LOGGER.debug("Setting up switch platform for entry_id=%s", entry.entry_id)

    async_add_entities([PhotoptimizerSwitch(coordinator, entry)])


class PhotoptimizerSwitch(
    RestoreEntity,
    CoordinatorEntity[PhotoptimizerCoordinator],
    SwitchEntity,
):
    """Representation of a Photoptimizer switch."""

    _attr_has_entity_name = True
    _attr_name = "Optimizer"

    def __init__(
        self,
        coordinator: PhotoptimizerCoordinator,
        entry: ConfigEntry,
    ) -> None:
        """Initialize the switch."""
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_optimizer_enabled"
        _LOGGER.debug("Created switch entity unique_id=%s", self._attr_unique_id)

    @property
    def is_on(self) -> bool:
        """Return whether the optimizer is enabled."""
        return self.coordinator.optimizer_enabled

    async def async_added_to_hass(self) -> None:
        """Restore the previous optimizer state after a restart."""
        await super().async_added_to_hass()
        previous_state = await self.async_get_last_state()
        if previous_state is None:
            _LOGGER.debug("No restored optimizer state available")
            return

        restored_is_on = previous_state.state == "on"
        _LOGGER.debug("Restored optimizer switch state=%s", previous_state.state)
        await self.coordinator.async_set_optimizer_enabled(restored_is_on)
        self.async_write_ha_state()

    @property
    def icon(self) -> str:
        """Icon of the entity based on state."""
        if self.is_on:
            return "mdi:solar-power"
        return "mdi:solar-power-variant-outline"

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the entity on."""
        await self.coordinator.async_set_optimizer_enabled(True)
        self.async_write_ha_state()
        _LOGGER.debug("Optimizer switch turned on")
        self.hass.async_create_task(self.coordinator.async_run_startup_ml_bootstrap())

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the entity off."""
        await self.coordinator.async_set_optimizer_enabled(False)
        self.async_write_ha_state()
        _LOGGER.debug("Optimizer switch turned off")
