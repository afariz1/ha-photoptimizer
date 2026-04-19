"""The Photoptimizer integration."""

from __future__ import annotations

from datetime import timedelta
import logging

from forecast_solar import ForecastSolar, ForecastSolarError

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.update_coordinator import UpdateFailed

from .const import DOMAIN
from .coordinator import PhotoptimizerCoordinator

_LOGGER = logging.getLogger(__name__)

_PLATFORMS: list[Platform] = [Platform.SENSOR, Platform.SWITCH]
_MPC_INTERVAL = timedelta(minutes=5)
_ML_TUNE_INTERVAL = timedelta(hours=24)


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
        if not coordinator.optimizer_enabled:
            _LOGGER.debug("Optimization disabled via switch; skipping MPC cycle")
            return

        _LOGGER.debug("Scheduled MPC optimization triggered")
        try:
            await coordinator.async_run_mpc_optimization()
            _LOGGER.debug("Scheduled MPC optimization finished successfully")
        except UpdateFailed as err:
            _LOGGER.warning("EMHASS MPC optimization failed: %s", err)

        _LOGGER.debug("Scheduled MPC publish triggered")
        try:
            await coordinator.async_run_mpc_publish()
            _LOGGER.debug("Scheduled MPC publish finished successfully")
        except UpdateFailed as err:
            _LOGGER.warning("EMHASS MPC publish-data failed: %s", err)

    async def _async_handle_ml_daily_tune() -> None:
        """Run one background ML tune cycle."""
        _LOGGER.debug("Scheduled ML daily tune triggered")
        try:
            await coordinator.async_run_ml_daily_tune()
            _LOGGER.debug("Scheduled ML daily tune finished")
        except UpdateFailed as err:
            _LOGGER.warning("EMHASS ML daily tune failed: %s", err)

    async def _async_handle_startup() -> None:
        """Run startup sequence: optimize, publish."""
        _LOGGER.debug("Startup started")
        coordinator.enable_ml_pipeline()
        await _async_handle_mpc_cycle()
        hass.async_create_task(_async_handle_ml_daily_tune())
        _LOGGER.debug("Startup bootstrap cycle finished")

    @callback
    def _mpc_schedule_listener(_: object) -> None:
        """Schedule one MPC cycle every configured interval."""
        _LOGGER.debug("MPC schedule fired")
        hass.async_create_task(_async_handle_mpc_cycle())

    entry.async_on_unload(
        async_track_time_interval(
            hass,
            _mpc_schedule_listener,
            _MPC_INTERVAL,
        )
    )

    @callback
    def _ml_tune_schedule_listener(_: object) -> None:
        """Schedule one ML tune cycle every configured interval."""
        _LOGGER.debug("ML tune schedule fired")
        hass.async_create_task(_async_handle_ml_daily_tune())

    entry.async_on_unload(
        async_track_time_interval(
            hass,
            _ml_tune_schedule_listener,
            _ML_TUNE_INTERVAL,
        )
    )

    _LOGGER.debug("Scheduling startup bootstrap task")
    startup_task = hass.async_create_task(_async_handle_startup())

    @callback
    def _cancel_startup_task() -> None:
        """Cancel startup task when entry is unloaded."""
        if not startup_task.done():
            startup_task.cancel()

    entry.async_on_unload(_cancel_startup_task)

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
