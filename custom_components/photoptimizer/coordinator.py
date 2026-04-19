"""Photoptimizer coordinator."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from functools import partial
import logging
from numbers import Real
from typing import Any, NoReturn

from forecast_solar import ForecastSolar, ForecastSolarError

from homeassistant.components.recorder import get_instance, history
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .const import (
    CONF_BATTERY_CAPACITY_KWH,
    CONF_BATTERY_CHARGE_POWER_MAX,
    CONF_BATTERY_DISCHARGE_POWER_MAX,
    CONF_BATTERY_EFFICIENCY_ROUND_TRIP,
    CONF_BATTERY_SOC_ENTITY,
    CONF_BATTERY_SOC_RESERVE_PERCENT,
    CONF_BATTERY_TARGET_SOC_PERCENT,
    CONF_CURRENT_CONSUMPTION_ENTITY,
    CONF_CURRENT_SOLAR_PRODUCTION_ENTITY,
    CONF_DEFERRABLE_LOADS,
    CONF_ELECTRICITY_PRICE_ENTITY,
    CONF_EMHASS_TOKEN,
    CONF_EMHASS_URL,
    CONF_TIMEZONE,
    CONF_WEAR_COST_PER_KWH,
    DEFAULT_BATTERY_CHARGE_POWER_MAX,
    DEFAULT_BATTERY_DISCHARGE_POWER_MAX,
    DEFAULT_BATTERY_EFFICIENCY_ROUND_TRIP,
    DEFAULT_BATTERY_SOC_RESERVE_PERCENT,
    DEFAULT_BATTERY_TARGET_SOC_PERCENT,
    DEFAULT_EMHASS_URL,
    DEFAULT_HORIZON_HOURS,
    DEFAULT_WEAR_COST_PER_KWH,
)
from .emhass_client import EmhassClient
from .executor import PhotoptimizerExecutor
from .mlforecast import MLForecastService
from .models import (
    DeferrableLoadDefinition,
    ExecutionPlan,
    OptimizationBucket,
    OptimizationInputs,
    PublishedEntityState,
)

_LOGGER = logging.getLogger(__name__)
_PV_BIAS_MIN_FACTOR = 0.6
_PV_BIAS_MAX_FACTOR = 1.4
_PV_BIAS_MIN_FORECAST_W = 50.0
_PV_BIAS_APPLY_BUCKETS = 4
_OPTIMIZATION_TIME_STEP_MINUTES = 15
_MPC_OPTIMIZATION_TIMEOUT_SECONDS = 180


class PhotoptimizerCoordinator(DataUpdateCoordinator[dict]):
    """Aggregate inputs for EMHASS and expose the combined result."""

    @staticmethod
    def _load_deferrable_loads(entry: ConfigEntry) -> list[DeferrableLoadDefinition]:
        """Return configured deferrable loads from options or setup data."""
        raw_loads = entry.options.get(CONF_DEFERRABLE_LOADS)
        if raw_loads is None:
            raw_loads = entry.data.get(CONF_DEFERRABLE_LOADS, [])

        loads: list[DeferrableLoadDefinition] = []
        for raw_load in raw_loads:
            if not isinstance(raw_load, dict):
                continue

            try:
                loads.append(
                    DeferrableLoadDefinition(
                        name=str(raw_load["name"]),
                        entity_id=str(raw_load["entity_id"]),
                        nominal_power_w=float(raw_load["nominal_power_w"]),
                        operating_minutes=int(raw_load["operating_minutes"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                _LOGGER.debug(
                    "Skipping invalid deferrable load definition: %s", raw_load
                )

        return loads

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, client: ForecastSolar
    ) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name="photoptimizer forecast",
            update_method=self._async_update_data,
            config_entry=entry,
        )
        self.client = client
        self.entry = entry
        self.deferrable_loads = self._load_deferrable_loads(entry)
        self.emhass_url = entry.data.get(CONF_EMHASS_URL, DEFAULT_EMHASS_URL)
        self.emhass_token = entry.data.get(CONF_EMHASS_TOKEN)
        self.emhass = EmhassClient(
            hass,
            self.emhass_url,
            self.emhass_token,
            battery_capacity_kwh=float(
                entry.data.get(
                    CONF_BATTERY_CAPACITY_KWH,
                    5.0,
                )
            ),
            battery_efficiency=(
                float(
                    entry.data.get(
                        CONF_BATTERY_EFFICIENCY_ROUND_TRIP,
                        DEFAULT_BATTERY_EFFICIENCY_ROUND_TRIP,
                    )
                )
                / 100.0
            ),
            battery_soc_reserve=(
                entry.data.get(
                    CONF_BATTERY_SOC_RESERVE_PERCENT,
                    DEFAULT_BATTERY_SOC_RESERVE_PERCENT,
                )
                / 100.0
            ),
            battery_target_soc=(
                entry.data.get(
                    CONF_BATTERY_TARGET_SOC_PERCENT,
                    DEFAULT_BATTERY_TARGET_SOC_PERCENT,
                )
                / 100.0
            ),
            battery_charge_power_max_w=float(
                entry.data.get(
                    CONF_BATTERY_CHARGE_POWER_MAX,
                    DEFAULT_BATTERY_CHARGE_POWER_MAX,
                )
            ),
            battery_discharge_power_max_w=float(
                entry.data.get(
                    CONF_BATTERY_DISCHARGE_POWER_MAX,
                    DEFAULT_BATTERY_DISCHARGE_POWER_MAX,
                )
            ),
            wear_cost_per_kwh=entry.data.get(
                CONF_WEAR_COST_PER_KWH,
                DEFAULT_WEAR_COST_PER_KWH,
            ),
            deferrable_loads=self.deferrable_loads,
        )
        self.ml_forecast = MLForecastService(hass, self)
        self.executor = PhotoptimizerExecutor(hass, entry)
        self._operation_lock = asyncio.Lock()
        self._last_optimization_utc: datetime | None = None
        self._last_publish_utc: datetime | None = None
        self._last_runtimeparams: dict[str, Any] = {}
        self._last_optimization_response: dict[str, Any] = {}
        self._last_publish_response: dict[str, Any] = {}
        self._last_published_entities: dict[str, PublishedEntityState] = {}
        self._last_execution_plan: ExecutionPlan | None = None
        self._last_execution_utc: datetime | None = None
        self._last_execution_applied: bool | None = None
        self._last_deferrable_loads_applied: bool | None = None
        self._optimizer_enabled: bool = True
        self._ml_forecast_enabled: bool = False
        _LOGGER.debug(
            "Coordinator initialized for entry_id=%s emhass_url=%s token=%s",
            entry.entry_id,
            self.emhass_url,
            "set" if self.emhass_token else "unset",
        )

    @property
    def optimizer_enabled(self) -> bool:
        """Return whether optimization/publish should run."""
        return self._optimizer_enabled

    async def async_set_optimizer_enabled(self, enabled: bool) -> None:
        """Enable/disable optimization runs."""
        self._optimizer_enabled = enabled

    @property
    def ml_forecast_enabled(self) -> bool:
        """Return whether ML forecasting is allowed."""
        return self._ml_forecast_enabled

    async def async_set_ml_forecast_enabled(self, enabled: bool) -> None:
        """Enable/disable ML forecasting."""
        self._ml_forecast_enabled = enabled

    def _log_step_start(self, step: str, detail: str | None = None) -> None:
        """Log start of an orchestrated integration step."""
        if detail is None:
            _LOGGER.debug("[STEP] %s", step)
            return

        _LOGGER.debug("[STEP] %s: %s", step, detail)

    def _log_step_ok(self, step: str, detail: str | None = None) -> None:
        """Log successful completion of an orchestrated integration step."""
        if detail is None:
            _LOGGER.debug("[OK] %s", step)
            return

        _LOGGER.debug("[OK] %s: %s", step, detail)

    def _log_step_error(self, step: str, err: Exception) -> None:
        """Log failed completion of an orchestrated integration step."""
        _LOGGER.warning("[ERR] %s failed: %s", step, err)

    def _raise_update_failed(self, message: str) -> NoReturn:
        """Raise an update failure from helper code paths."""
        raise UpdateFailed(message)

    def _bucket_step_minutes(self, buckets: list[OptimizationBucket]) -> int:
        """Return bucket step length in minutes, using integration default as fallback."""
        if len(buckets) < 2:
            return _OPTIMIZATION_TIME_STEP_MINUTES

        return int((buckets[1].start - buckets[0].start).total_seconds() // 60)

    def _bucket_hours(self, buckets: list[OptimizationBucket]) -> float:
        """Return bucket duration in hours."""
        return self._bucket_step_minutes(buckets) / 60.0

    def bucket_step_minutes(self, buckets: list[OptimizationBucket]) -> int:
        """Return bucket step length in minutes."""
        return self._bucket_step_minutes(buckets)

    async def async_hourly_from_load_profile(
        self, buckets: list[OptimizationBucket], profile: list[float]
    ) -> None:
        """Fill load buckets from a 24-hour profile."""
        await self._hourly_from_load_profile(buckets, profile)

    async def async_build_load_profile(self, entity_id: str | None) -> list[float]:
        """Build a 24-hour load profile from state history or return defaults."""
        return await self._build_load_profile(entity_id)

    async def async_run_startup_bootstrap(self) -> None:
        """Run one full MPC cycle during startup."""
        if not self._optimizer_enabled:
            self._log_step_ok("startup_bootstrap", "optimizer disabled, skipping")
            return

        self._log_step_start("startup_bootstrap")

        try:
            await self.async_run_mpc_optimization()
        except UpdateFailed as err:
            self._log_step_error("startup_mpc_optimization", err)
            return

        try:
            await self.async_run_mpc_publish()
        except UpdateFailed as err:
            self._log_step_error("startup_mpc_publish", err)
            return

        self._log_step_ok("startup_bootstrap")

    def _coerce_float(self, value: object) -> float | None:
        """Convert arbitrary value to float when possible."""
        if isinstance(value, str) or (
            isinstance(value, Real) and not isinstance(value, bool)
        ):
            try:
                return float(value)
            except ValueError:
                return None

        return None

    async def _async_get_state_with_startup_wait(self, entity_id: str) -> State | None:
        """Return entity state, waiting briefly during startup for late entities."""
        state = self.hass.states.get(entity_id)
        if state is not None or self.hass.is_running:
            if state is None:
                _LOGGER.debug(
                    "Entity %s unavailable (system already running)", entity_id
                )
            return state

        _LOGGER.debug("Entity %s missing during startup, waiting up to 8s", entity_id)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 8.0

        while loop.time() < deadline:
            await asyncio.sleep(0.25)
            state = self.hass.states.get(entity_id)
            if state is not None:
                _LOGGER.debug(
                    "Entity %s became available during startup wait", entity_id
                )
                return state

        _LOGGER.debug("Entity %s not available after startup wait", entity_id)
        return None

    def _build_result(
        self,
        optimization_inputs: OptimizationInputs,
        raw_pv: Any | None,
    ) -> dict[str, Any]:
        """Compose coordinator payload shared by all entities."""
        published_entities = {
            key: entity.as_dict()
            for key, entity in self._last_published_entities.items()
        }

        return {
            "timeline": [bucket.as_dict() for bucket in optimization_inputs.timeline],
            "inputs": {
                "battery_soc": optimization_inputs.battery_soc,
                "prediction_horizon": optimization_inputs.prediction_horizon,
                "optimization_time_step_minutes": optimization_inputs.optimization_time_step_minutes,
                "deferrable_loads": [
                    deferrable_load.as_dict()
                    for deferrable_load in optimization_inputs.deferrable_loads
                ],
            },
            "raw": {"forecast_solar": raw_pv},
            "emhass": {
                "runtimeparams": self._last_runtimeparams,
                "optimization_response": self._last_optimization_response,
                "publish_response": self._last_publish_response,
                "published_entities": published_entities,
                "execution_plan": (
                    None
                    if self._last_execution_plan is None
                    else self._last_execution_plan.as_dict()
                ),
                "would_apply": (
                    {
                        "battery_power_w": None,
                        "effective_at": None,
                        "source": "missing_execution_plan",
                    }
                    if self._last_execution_plan is None
                    or not self._last_execution_plan.slots
                    else {
                        "battery_power_w": self._last_execution_plan.slots[0].p_bat_cmd,
                        "effective_at": self._last_execution_plan.slots[
                            0
                        ].slot_start.isoformat(),
                        "source": self._last_execution_plan.source,
                        "valid": self._last_execution_plan.valid,
                    }
                ),
            },
            "schedule": {
                "last_optimization_utc": (
                    None
                    if self._last_optimization_utc is None
                    else self._last_optimization_utc.isoformat()
                ),
                "last_publish_utc": (
                    None
                    if self._last_publish_utc is None
                    else self._last_publish_utc.isoformat()
                ),
                "last_execution_utc": (
                    None
                    if self._last_execution_utc is None
                    else self._last_execution_utc.isoformat()
                ),
                "last_execution_applied": self._last_execution_applied,
                "last_deferrable_loads_applied": self._last_deferrable_loads_applied,
            },
        }

    async def _async_collect_inputs(
        self,
        *,
        allow_ml_forecast: bool = True,
    ) -> tuple[OptimizationInputs, Any | None]:
        """Collect timeline and plant inputs needed for EMHASS calls."""
        timeline: list[OptimizationBucket] = []
        tz_name = self.entry.data.get(CONF_TIMEZONE) or self.hass.config.time_zone
        tz = dt_util.get_time_zone(tz_name) or dt_util.UTC
        step_minutes = _OPTIMIZATION_TIME_STEP_MINUTES
        now = dt_util.now(tz).replace(second=0, microsecond=0)
        aligned_minute = (now.minute // step_minutes) * step_minutes
        now = now.replace(minute=aligned_minute)
        horizon_hours = DEFAULT_HORIZON_HOURS

        price_entity = self.entry.data.get(CONF_ELECTRICITY_PRICE_ENTITY)
        if price_entity:
            detected_price_horizon_hours = await self._detect_price_horizon_hours(
                price_entity,
                now,
                tz,
            )
            if detected_price_horizon_hours <= 0:
                self._raise_update_failed(
                    f"Price entity {price_entity} has no future spot-price data"
                )

            horizon_hours = min(horizon_hours, detected_price_horizon_hours)
            _LOGGER.debug(
                "Spot-price horizon detected from %s: detected=%s configured=%s effective=%s",
                price_entity,
                detected_price_horizon_hours,
                DEFAULT_HORIZON_HOURS,
                horizon_hours,
            )

        bucket_count = int((horizon_hours * 60) / step_minutes)
        _LOGGER.debug(
            "Collecting inputs: horizon_hours=%s step_minutes=%s timezone=%s start=%s buckets=%s",
            horizon_hours,
            step_minutes,
            tz_name,
            now.isoformat(),
            bucket_count,
        )

        for bucket_offset in range(bucket_count):
            bucket_start = now + timedelta(minutes=step_minutes * bucket_offset)
            timeline.append(
                OptimizationBucket(
                    start=bucket_start,
                    price=0.0,
                    pv=0.0,
                    load=0.0,
                )
            )

        if price_entity:
            _LOGGER.debug("Using electricity price entity: %s", price_entity)
            mapped_price_hours = await self._hourly_from_price_entity(
                price_entity, timeline
            )
            if mapped_price_hours <= 0:
                self._raise_update_failed(
                    f"Price entity {price_entity} contains no usable price points"
                )
        else:
            _LOGGER.debug("No electricity price entity configured")

        _LOGGER.debug("Using Forecast.Solar client for PV forecast")
        raw_pv = await self.client.estimate()
        await self._hourly_from_forecast_solar(timeline, raw_pv)
        await self._apply_current_solar_bias_correction(timeline)

        load_entity = self.entry.data.get(CONF_CURRENT_CONSUMPTION_ENTITY)
        if load_entity:
            if allow_ml_forecast:
                _LOGGER.debug(
                    "Using load entity for ML forecast: %s",
                    load_entity,
                )
                await self.ml_forecast.async_populate_load_from_ml_or_profile(
                    load_entity, timeline
                )
            else:
                _LOGGER.debug(
                    "ML forecast disabled for this cycle, using profile fallback"
                )
                load_profile = await self._build_load_profile(load_entity)
                await self._hourly_from_load_profile(timeline, load_profile)
        else:
            _LOGGER.debug("No load entity configured, using default profile")
            load_profile = await self._build_load_profile(None)
            await self._hourly_from_load_profile(timeline, load_profile)

        _LOGGER.debug(
            "Input collection finished with %s timeline buckets", len(timeline)
        )

        return (
            OptimizationInputs(
                timeline=timeline,
                battery_soc=self._read_battery_soc(),
                deferrable_loads=self.deferrable_loads,
                raw_forecast_solar=raw_pv,
            ),
            raw_pv,
        )

    async def _detect_price_horizon_hours(
        self,
        entity_id: str,
        start: datetime,
        tz,
    ) -> int:
        """Return contiguous available future spot-price hours from aligned start."""
        state = await self._async_get_state_with_startup_wait(entity_id)
        if state is None:
            raise UpdateFailed(f"Price entity {entity_id} not found")

        start_hour = start.replace(minute=0, second=0, microsecond=0)
        available_hours: set[datetime] = set()

        for key, value in state.attributes.items():
            if self._coerce_float(value) is None:
                continue

            dt = dt_util.parse_datetime(str(key))
            if dt is None:
                continue

            dt_local = dt_util.as_local(dt).astimezone(tz)
            hour_start = dt_local.replace(minute=0, second=0, microsecond=0)
            if hour_start < start_hour:
                continue

            available_hours.add(hour_start)

        contiguous_hours = 0
        probe = start_hour
        while probe in available_hours:
            contiguous_hours += 1
            probe += timedelta(hours=1)

        return contiguous_hours

    async def _hourly_from_price_entity(
        self, entity_id: str, buckets: list[OptimizationBucket]
    ) -> int:
        state = await self._async_get_state_with_startup_wait(entity_id)

        if state is None:
            raise UpdateFailed(f"Price entity {entity_id} not found")

        tz_name = self.entry.data.get(CONF_TIMEZONE) or self.hass.config.time_zone
        tz = dt_util.get_time_zone(tz_name) or dt_util.UTC

        hour_bucket_index: dict[datetime, list[OptimizationBucket]] = {}
        for bucket in buckets:
            hour_start = bucket.start.replace(minute=0, second=0, microsecond=0)
            hour_bucket_index.setdefault(hour_start, []).append(bucket)
        mapped_hours: set[datetime] = set()

        for key, value in state.attributes.items():
            numeric = self._coerce_float(value)
            if numeric is None:
                continue

            dt = dt_util.parse_datetime(str(key))
            if dt is None:
                continue

            dt_local = dt_util.as_local(dt).astimezone(tz)
            hour_start = dt_local.replace(minute=0, second=0, microsecond=0)
            target_buckets = hour_bucket_index.get(hour_start)
            if target_buckets:
                for bucket in target_buckets:
                    bucket.price = numeric
                mapped_hours.add(hour_start)

        _LOGGER.debug(
            "Price timeline populated from %s: mapped_hours=%s",
            entity_id,
            len(mapped_hours),
        )
        return len(mapped_hours)

    async def _hourly_from_forecast_solar(
        self, buckets: list[OptimizationBucket], raw_pv
    ) -> None:
        tz_name = self.entry.data.get(CONF_TIMEZONE) or self.hass.config.time_zone
        tz = dt_util.get_time_zone(tz_name) or dt_util.UTC

        hour_bucket_index: dict[datetime, list[OptimizationBucket]] = {}
        for bucket in buckets:
            hour_start = bucket.start.replace(minute=0, second=0, microsecond=0)
            hour_bucket_index.setdefault(hour_start, []).append(bucket)
        mapped_points = 0

        for dt_utc, wh in raw_pv.wh_period.items():
            dt_local = dt_util.as_local(dt_utc).astimezone(tz)
            hour_start = dt_local.replace(minute=0, second=0, microsecond=0)
            target_buckets = hour_bucket_index.get(hour_start)
            if target_buckets:
                per_bucket_kwh = (float(wh) / 1000.0) / len(target_buckets)
                for bucket in target_buckets:
                    bucket.pv += per_bucket_kwh
            mapped_points += 1

        _LOGGER.debug(
            "PV timeline populated from Forecast.Solar: mapped=%s", mapped_points
        )

    def _normalize_power_w(self, state: State) -> float | None:
        """Return state power normalized to Watts."""
        numeric = self._coerce_float(state.state)
        if numeric is None:
            return None

        unit = str(state.attributes.get("unit_of_measurement", "")).lower()
        if unit == "kw":
            return numeric * 1000.0
        if unit == "mw":
            return numeric * 1_000_000.0
        return numeric

    async def _apply_current_solar_bias_correction(
        self,
        buckets: list[OptimizationBucket],
    ) -> None:
        """Adjust near-term PV forecast from current measured production."""
        if not buckets:
            return

        entity_id = self.entry.data.get(CONF_CURRENT_SOLAR_PRODUCTION_ENTITY)
        if not entity_id:
            _LOGGER.debug("Skipping PV bias correction: no current solar entity")
            return

        state = await self._async_get_state_with_startup_wait(entity_id)
        if state is None:
            _LOGGER.debug(
                "Skipping PV bias correction: entity %s unavailable", entity_id
            )
            return

        actual_power_w = self._normalize_power_w(state)
        if actual_power_w is None:
            _LOGGER.debug(
                "Skipping PV bias correction: invalid state for %s (%s)",
                entity_id,
                state.state,
            )
            return

        bucket_hours = self._bucket_hours(buckets)
        if bucket_hours <= 0:
            _LOGGER.debug(
                "Skipping PV bias correction: invalid bucket_hours=%s", bucket_hours
            )
            return

        forecast_power_w = max((buckets[0].pv / bucket_hours) * 1000.0, 0.0)
        if forecast_power_w < _PV_BIAS_MIN_FORECAST_W:
            _LOGGER.debug(
                "Skipping PV bias correction: forecast %.2f W too small",
                forecast_power_w,
            )
            return

        raw_factor = max(actual_power_w, 0.0) / forecast_power_w
        factor = min(max(raw_factor, _PV_BIAS_MIN_FACTOR), _PV_BIAS_MAX_FACTOR)
        if abs(factor - 1.0) < 0.01:
            return

        affected = min(_PV_BIAS_APPLY_BUCKETS, len(buckets))
        for index in range(affected):
            buckets[index].pv = max(0.0, buckets[index].pv * factor)

        _LOGGER.debug(
            "Applied PV bias correction from %s: actual=%.2fW forecast=%.2fW factor=%.3f affected_buckets=%s",
            entity_id,
            actual_power_w,
            forecast_power_w,
            factor,
            affected,
        )

    async def _hourly_from_load_profile(
        self, buckets: list[OptimizationBucket], profile: list[float]
    ) -> None:
        """Fill load buckets from a 24-hour profile."""
        bucket_hours = self._bucket_hours(buckets)

        for bucket in buckets:
            bucket.load = profile[bucket.start.hour] * bucket_hours

        _LOGGER.debug(
            "Load timeline populated from profile with %s hourly values",
            len(profile),
        )

    async def _build_load_profile_from_state_history(
        self,
        entity_id: str,
        start: datetime,
        default_profile: list[float],
    ) -> list[float] | None:
        """Build load profile from recorder state history when statistics are absent."""
        lower_entity_id = entity_id.lower()
        try:
            state_history = await get_instance(self.hass).async_add_executor_job(
                partial(
                    history.state_changes_during_period,
                    self.hass,
                    start,
                    None,
                    lower_entity_id,
                    include_start_time_state=True,
                )
            )
        except (OSError, RuntimeError, ValueError) as err:
            _LOGGER.debug("Load state history query failed: %s", err)
            return None

        history_rows = state_history.get(lower_entity_id, [])
        _LOGGER.debug(
            "Load state history query: entity_id=%s, start=%s (UTC), rows=%s",
            entity_id,
            start,
            len(history_rows),
        )
        if not history_rows:
            return None

        slot_totals_kw: dict[datetime, float] = {}
        slot_counts: dict[datetime, int] = {}

        for hist_state in history_rows:
            value = self._coerce_float(hist_state.state)
            if value is None:
                continue

            unit = str(hist_state.attributes.get("unit_of_measurement", "")).lower()
            val = float(value)
            if unit == "w":
                val = val / 1000.0
            elif unit == "kw":
                pass
            elif unit == "mw":
                val = val * 1000.0
            elif val > 30:
                val = val / 1000.0

            changed = getattr(hist_state, "last_updated", hist_state.last_changed)
            local_changed = dt_util.as_local(changed).replace(second=0, microsecond=0)
            slot_minute = (
                local_changed.minute // _OPTIMIZATION_TIME_STEP_MINUTES
            ) * _OPTIMIZATION_TIME_STEP_MINUTES
            slot_start = local_changed.replace(minute=slot_minute)

            slot_totals_kw[slot_start] = slot_totals_kw.get(slot_start, 0.0) + val
            slot_counts[slot_start] = slot_counts.get(slot_start, 0) + 1

        if not slot_counts:
            return None

        hourly_totals = [0.0] * 24
        hourly_counts = [0] * 24
        for slot_start, total_kw in slot_totals_kw.items():
            count = slot_counts[slot_start]
            if count == 0:
                continue

            hour = slot_start.hour
            hourly_totals[hour] += total_kw / count
            hourly_counts[hour] += 1

        if not any(hourly_counts):
            return None

        profile: list[float] = []
        for hour in range(24):
            if hourly_counts[hour]:
                profile.append(hourly_totals[hour] / hourly_counts[hour])
            else:
                profile.append(default_profile[hour])

        _LOGGER.debug(
            "Load profile built from state history for %s (filled_hours=%s, filled_slots=%s)",
            entity_id,
            sum(1 for count in hourly_counts if count),
            len(slot_counts),
        )
        return profile

    async def _build_load_profile(self, entity_id: str | None) -> list[float]:
        """Build a simple 24h load profile from state history; fallback to defaults."""
        default_profile = [
            0.4,
            0.35,
            0.35,
            0.35,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.7,
            0.7,
            0.75,
            0.8,
            0.9,
            1.2,
            1.4,
            1.5,
            1.3,
            1.0,
            0.8,
            0.6,
        ]

        if not entity_id:
            _LOGGER.debug("Using default load profile (no entity configured)")
            return default_profile

        start = dt_util.utcnow() - timedelta(days=7)
        history_profile = await self._build_load_profile_from_state_history(
            entity_id,
            start,
            default_profile,
        )
        if history_profile is not None:
            return history_profile

        _LOGGER.debug(
            "No usable state history found for %s, using default load profile",
            entity_id,
        )
        return default_profile

    def _read_battery_soc(self) -> float:
        """Read and normalize the current battery SOC from Home Assistant."""
        soc_entity = self.entry.data.get(CONF_BATTERY_SOC_ENTITY)
        soc_state = self.hass.states.get(soc_entity) if soc_entity else None
        raw_soc = self._coerce_float(soc_state.state) if soc_state is not None else None
        if raw_soc is None:
            normalized_soc = 0.0
        else:
            normalized_soc = raw_soc if raw_soc <= 1.0 else raw_soc / 100.0
            normalized_soc = max(0.0, min(1.0, normalized_soc))

        _LOGGER.debug(
            "Battery SOC read from %s: raw=%s normalized=%s",
            soc_entity,
            None if soc_state is None else soc_state.state,
            normalized_soc,
        )
        return normalized_soc

    def _log_timeline(self, timeline: list[OptimizationBucket]) -> None:
        """Log the aggregated inputs passed into EMHASS."""
        _LOGGER.debug(
            "Photoptimizer inputs (%d hours):\n  %s\n%s",
            len(timeline),
            f"{'Hour':<17} {'Price':>9} {'PV (kWh)':>10} {'Load (kWh)':>11}",
            "\n".join(
                f"  {bucket.start.strftime('%Y-%m-%d %H:%M')} "
                f"{bucket.price:9.4f} "
                f"{bucket.pv:10.3f} "
                f"{bucket.load:11.3f}"
                for bucket in timeline
            ),
        )

    async def _async_update_data(self) -> dict:
        """Collect input data.

        Scheduled tasks call optimization and publish actions.
        """
        try:
            _LOGGER.debug("Coordinator refresh started")
            async with asyncio.timeout(30):
                optimization_inputs, raw_pv = await self._async_collect_inputs(
                    allow_ml_forecast=self._ml_forecast_enabled,
                )
                if _LOGGER.isEnabledFor(logging.DEBUG):
                    self._log_timeline(optimization_inputs.timeline)
                result = self._build_result(optimization_inputs, raw_pv)
                _LOGGER.debug(
                    "Coordinator refresh finished: horizon=%s step=%s",
                    optimization_inputs.prediction_horizon,
                    optimization_inputs.optimization_time_step_minutes,
                )
                return result

        except ForecastSolarError as err:
            raise UpdateFailed(f"Forecast.Solar API error: {err}") from err
        except TimeoutError as err:
            raise UpdateFailed("Coordinator refresh timed out") from err

    async def async_run_mpc_optimization(self) -> None:
        """Run one naive MPC optimization call to EMHASS."""
        async with self._operation_lock:
            self._log_step_start("mpc_optimization")
            if not self._optimizer_enabled:
                self._log_step_ok("mpc_optimization", "optimizer disabled, skipped")
                return
            try:
                async with asyncio.timeout(_MPC_OPTIMIZATION_TIMEOUT_SECONDS):
                    optimization_inputs, raw_pv = await self._async_collect_inputs(
                        allow_ml_forecast=self._ml_forecast_enabled,
                    )
                    optimization_result = (
                        await self.emhass.async_run_naive_optimization(
                            optimization_inputs
                        )
                    )
                    if optimization_result is None:
                        self._raise_update_failed("EMHASS MPC optimization failed")

                    self._last_runtimeparams = optimization_result["runtimeparams"]
                    self._last_optimization_response = optimization_result[
                        "optimization_response"
                    ]
                    self._last_optimization_utc = dt_util.utcnow()

                    if _LOGGER.isEnabledFor(logging.DEBUG):
                        self._log_timeline(optimization_inputs.timeline)

                    self.async_set_updated_data(
                        self._build_result(optimization_inputs, raw_pv)
                    )
                    self._log_step_ok(
                        "mpc_optimization",
                        f"runtimeparams_keys={sorted(self._last_runtimeparams.keys())} "
                        f"response_keys={sorted(self._last_optimization_response.keys())}",
                    )
            except ForecastSolarError as err:
                update_failed = UpdateFailed(f"Forecast.Solar API error: {err}")
                self._log_step_error("mpc_optimization", update_failed)
                raise update_failed from err
            except UpdateFailed as err:
                self._log_step_error("mpc_optimization", err)
                raise
            except TimeoutError as err:
                update_failed = UpdateFailed("MPC optimization timed out")
                self._log_step_error("mpc_optimization", update_failed)
                raise update_failed from err
            except Exception as err:
                update_failed = UpdateFailed(
                    f"MPC optimization unexpected error: {err}"
                )
                _LOGGER.exception("[ERR] mpc_optimization unexpected exception")
                raise update_failed from err

    async def async_run_mpc_publish(self) -> None:
        """Run one publish-data step for the latest MPC output."""
        async with self._operation_lock:
            self._log_step_start("mpc_publish")
            if not self._optimizer_enabled:
                self._log_step_ok("mpc_publish", "optimizer disabled, skipped")
                return
            if self._last_optimization_utc is None:
                self._log_step_ok(
                    "mpc_publish",
                    "skipped: no successful optimization available",
                )
                return
            try:
                async with asyncio.timeout(60):
                    optimization_inputs, raw_pv = await self._async_collect_inputs(
                        allow_ml_forecast=self._ml_forecast_enabled,
                    )
                    publish_result = await self.emhass.async_publish_data(
                        optimization_inputs.optimization_time_step_minutes
                    )
                    if publish_result is None:
                        self._raise_update_failed("EMHASS MPC publish-data failed")

                    self._last_publish_response = publish_result["publish_response"]
                    self._last_published_entities = publish_result["published_entities"]
                    self._last_execution_plan = publish_result["execution_plan"]
                    self._last_publish_utc = dt_util.utcnow()

                    try:
                        self._last_execution_applied = (
                            await self.executor.async_execute_plan(
                                self._last_execution_plan
                            )
                        )
                        self._last_deferrable_loads_applied = (
                            await self.executor.async_execute_deferrable_loads(
                                self._last_published_entities,
                                self.deferrable_loads,
                            )
                        )
                        self._last_execution_utc = dt_util.utcnow()
                    except (
                        HomeAssistantError,
                        RuntimeError,
                        ValueError,
                    ) as err:
                        self._last_execution_applied = False
                        self._last_deferrable_loads_applied = False
                        _LOGGER.warning("Executor apply failed: %s", err)

                    if _LOGGER.isEnabledFor(logging.DEBUG):
                        self._log_timeline(optimization_inputs.timeline)

                    self.async_set_updated_data(
                        self._build_result(optimization_inputs, raw_pv)
                    )
                    self._log_step_ok(
                        "mpc_publish",
                        f"published_entities={sorted(self._last_published_entities.keys())} "
                        f"response_keys={sorted(self._last_publish_response.keys())}",
                    )
            except ForecastSolarError as err:
                update_failed = UpdateFailed(f"Forecast.Solar API error: {err}")
                self._log_step_error("mpc_publish", update_failed)
                raise update_failed from err
            except UpdateFailed as err:
                self._log_step_error("mpc_publish", err)
                raise
            except TimeoutError as err:
                update_failed = UpdateFailed("MPC publish timed out")
                self._log_step_error("mpc_publish", update_failed)
                raise update_failed from err
            except Exception as err:
                update_failed = UpdateFailed(f"MPC publish unexpected error: {err}")
                _LOGGER.exception("[ERR] mpc_publish unexpected exception")
                raise update_failed from err
