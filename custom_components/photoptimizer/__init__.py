"""The Photoptimizer integration."""

from __future__ import annotations

import asyncio
import logging

from forecast_solar import ForecastSolar, ForecastSolarError

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import async_track_time_change
from homeassistant.helpers.update_coordinator import UpdateFailed

from .const import DOMAIN
from .coordinator import PhotoptimizerCoordinator

_LOGGER = logging.getLogger(__name__)

_PLATFORMS: list[Platform] = [Platform.SENSOR, Platform.SWITCH]
_MPC_QUARTER_MINUTES = [0, 15, 30, 45]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Photoptimizer."""
    if entry.entry_id in hass.data.get(DOMAIN, {}):
        _LOGGER.debug(
            "Entry already initialized, skipping duplicate setup for entry_id=%s",
            entry.entry_id,
        )
        return True

    _LOGGER.debug(
        "Starting setup for entry_id=%s title=%s", entry.entry_id, entry.title
    )

    scheduled_tasks: set[asyncio.Task[object]] = set()

    def _track_task(task: asyncio.Task[object]) -> asyncio.Task[object]:
        """Remember a scheduled task so it can be cancelled on unload."""
        scheduled_tasks.add(task)
        task.add_done_callback(scheduled_tasks.discard)
        return task

    session = async_get_clientsession(hass)

    latitude = entry.data.get("latitude") or hass.config.latitude
    longitude = entry.data.get("longitude") or hass.config.longitude
    declination = entry.data.get("tilt") or entry.data.get("declination") or 40.0
    azimuth = entry.data.get("azimuth") or 90.0
    kwp = entry.data.get("kwp") or 5.0
    api_key = entry.data.get("api_key")

    _LOGGER.debug(
        "Resolved Forecast.Solar config: lat=%s lon=%s azimuth=%s declination=%s kwp=%s api_key=%s",
        latitude,
        longitude,
        azimuth,
        declination,
        kwp,
        "set" if api_key else "unset",
    )

    if latitude is None or longitude is None:
        raise ConfigEntryNotReady("Latitude and longitude are required")

    client = ForecastSolar(
        session=session,
        latitude=float(latitude),
        longitude=float(longitude),
        declination=float(declination),
        azimuth=float(azimuth),
        kwp=float(kwp),
        damping=0,
        api_key=api_key,
    )

    try:
        _LOGGER.debug("Validating Forecast.Solar connectivity")
        await client.estimate()
        _LOGGER.debug("Forecast.Solar validation successful")
    except ForecastSolarError as err:
        _LOGGER.debug("ForecastSolar validation failed: %s", err)
        raise ConfigEntryNotReady(
            f"Unable to connect to Forecast.Solar: {err}"
        ) from err
    except Exception as err:
        _LOGGER.debug("Unexpected error during Forecast.Solar validation: %s", err)
        raise ConfigEntryNotReady(
            f"Unexpected error connecting to Forecast.Solar: {err}"
        ) from err

    coordinator = PhotoptimizerCoordinator(hass, entry, client)
    _LOGGER.debug("Running first coordinator refresh")
    await coordinator.async_config_entry_first_refresh()
    _LOGGER.debug("First coordinator refresh finished")

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = coordinator
    entry.runtime_data = coordinator

    _LOGGER.debug(
        "Forwarding setup to platforms: %s", [platform.value for platform in _PLATFORMS]
    )
    await hass.config_entries.async_forward_entry_setups(entry, _PLATFORMS)
    _LOGGER.debug("Platform setup forwarding completed")

    async def _async_handle_mpc_cycle() -> None:
        """Run one full MPC cycle in strict order: optimize, then publish."""
        _LOGGER.debug("Scheduled MPC cycle triggered")
        if not coordinator.ml_bootstrap_completed:
            _LOGGER.debug("Skipping MPC cycle until startup ML bootstrap is finished")
            return
        if not coordinator.optimizer_enabled:
            _LOGGER.debug("Optimization disabled via switch; skipping MPC cycle")
            return

        _LOGGER.debug("Scheduled MPC optimization triggered")
        try:
            await coordinator.async_run_mpc_optimization()
            _LOGGER.debug("Scheduled MPC optimization finished successfully")
        except UpdateFailed as err:
            _LOGGER.warning("EMHASS MPC optimization failed: %s", err)
            return

        _LOGGER.debug("Scheduled MPC publish triggered")
        try:
            await coordinator.async_run_mpc_publish()
            _LOGGER.debug("Scheduled MPC publish finished successfully")
        except UpdateFailed as err:
            _LOGGER.warning("EMHASS MPC publish-data failed: %s", err)

    async def _async_handle_ml_daily_refresh() -> None:
        """Run one background ML daily refresh cycle."""
        _LOGGER.debug("Scheduled ML daily refresh triggered")
        try:
            await coordinator.async_run_ml_daily_refresh()
            _LOGGER.debug("Scheduled ML daily refresh finished")
        except UpdateFailed as err:
            _LOGGER.warning("EMHASS ML daily refresh failed: %s", err)

    async def _async_handle_startup() -> None:
        """Run startup sequence: ML bootstrap only."""
        _LOGGER.debug("Startup started")
        try:
            await coordinator.async_run_startup_ml_bootstrap()
        except Exception as err:
            _LOGGER.warning("Startup ML bootstrap failed: %s", err)
        _LOGGER.debug("Startup ML bootstrap finished")

    @callback
    def _mpc_schedule_listener(_: object) -> None:
        """Schedule one MPC cycle at quarter-hour boundaries."""
        _LOGGER.debug("MPC schedule fired")
        _track_task(hass.async_create_task(_async_handle_mpc_cycle()))

    entry.async_on_unload(
        async_track_time_change(
            hass,
            _mpc_schedule_listener,
            minute=_MPC_QUARTER_MINUTES,
            second=0,
        )
    )

    @callback
    def _ml_daily_schedule_listener(_: object) -> None:
        """Schedule one ML daily refresh at local 00:05."""
        _LOGGER.debug("ML daily refresh schedule fired")
        _track_task(hass.async_create_task(_async_handle_ml_daily_refresh()))

    entry.async_on_unload(
        async_track_time_change(
            hass,
            _ml_daily_schedule_listener,
            hour=0,
            minute=5,
            second=0,
        )
    )

    _LOGGER.debug("Scheduling startup bootstrap task")
    _track_task(hass.async_create_task(_async_handle_startup()))

    @callback
    def _cancel_scheduled_tasks() -> None:
        """Cancel all outstanding scheduled tasks when entry is unloaded."""
        for task in tuple(scheduled_tasks):
            if not task.done():
                task.cancel()

    entry.async_on_unload(_cancel_scheduled_tasks)

    _LOGGER.debug("Setup finished for entry_id=%s", entry.entry_id)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    _LOGGER.debug("Unloading entry_id=%s", entry.entry_id)
    unload_ok = await hass.config_entries.async_unload_platforms(entry, _PLATFORMS)
    if unload_ok:
        hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
        entry.runtime_data = None
        _LOGGER.debug("Unload successful for entry_id=%s", entry.entry_id)
    else:
        _LOGGER.debug("Unload failed for entry_id=%s", entry.entry_id)
    return unload_ok
