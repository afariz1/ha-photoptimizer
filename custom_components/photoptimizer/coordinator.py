"""Photoptimizer coordinator."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from functools import partial
from numbers import Real
import time
from typing import Any, NoReturn

from forecast_solar import ForecastSolar, ForecastSolarError

from homeassistant.components.recorder import get_instance, history
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store
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
    DOMAIN,
    PV_HOURLY_FACTOR_MAX,
    PV_HOURLY_FACTOR_MIN,
    PV_HOURLY_MIN_POWER_W,
    PV_HOURLY_RATIO_MAX,
    PV_HOURLY_RATIO_MIN,
    PV_HOURLY_STORE_VERSION,
    forecast_solar_hour_wh_to_per_bucket_kwh,
    pv_hourly_ewma_update,
)
from .emhass_client import DEFAULT_ML_PREDICT_PUBLISH_ENTITY_ID, EmhassClient
from .executor import ExecutorApplyResult, PhotoptimizerExecutor
from .mlforecast import MLForecastService
from .models import (
    DeferrableLoadDefinition,
    ExecutionSlotCommand,
    ExecutionPlan,
    OptimizationBucket,
    OptimizationInputs,
    OperationMode,
    PublishedEntityState,
)

_LOGGER = logging.getLogger(__name__)
_OPTIMIZATION_TIME_STEP_MINUTES = 15
_ML_BOOTSTRAP_TIMEOUT_SECONDS = 300
_ML_BOOTSTRAP_MAX_ATTEMPTS = 3
_ML_BOOTSTRAP_RETRY_DELAY_SECONDS = 30
_LOAD_PROFILE_HISTORY_WINDOW_DAYS = 2


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
        self._last_valid_execution_plan: ExecutionPlan | None = None
        self._publish_failure_streak: int = 0
        self._last_inputs_snapshot: OptimizationInputs | None = None
        self._last_raw_pv_snapshot: Any | None = None
        self._last_execution_utc: datetime | None = None
        self._last_execution_applied: bool | None = None
        self._last_deferrable_loads_applied: bool | None = None
        self._last_valid_battery_soc: float | None = None
        self._last_pv_forecast_series: dict[str, Any] = {}
        self._pv_hourly_factors: list[float] = [1.0] * 24
        self._pv_hourly_store: Store = Store(
            hass,
            PV_HOURLY_STORE_VERSION,
            f"{DOMAIN}.pv_hourly_factors.{entry.entry_id}",
        )
        self._pv_hourly_factors_loaded: bool = False
        self._last_pv_uncorrected_kwh: list[float] | None = None
        self._last_timeline_first_start: datetime | None = None
        self._last_executor_apply_result: ExecutorApplyResult | None = None
        self._stats_window_hour: datetime | None = None
        self._stats_cycles: int = 0
        self._stats_opt_ok: int = 0
        self._stats_opt_fail: int = 0
        self._stats_pub_ok: int = 0
        self._stats_pub_fail: int = 0
        self._stats_opt_duration_ms: int = 0
        self._stats_pub_duration_ms: int = 0
        self._stats_exec_applied: int = 0
        self._stats_exec_skipped: int = 0
        self._optimizer_enabled: bool = True
        self._ml_bootstrap_completed: bool = False
        self._ml_fit_completed: bool = False
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

    @property
    def last_executor_apply_result(self) -> ExecutorApplyResult | None:
        """Outcome of the last inverter command apply attempt (after publish)."""
        return self._last_executor_apply_result

    async def async_set_optimizer_enabled(self, enabled: bool) -> None:
        """Enable/disable optimization runs."""
        self._optimizer_enabled = enabled

    @property
    def ml_bootstrap_completed(self) -> bool:
        """Return whether startup ML bootstrap reached completion state."""
        return self._ml_bootstrap_completed

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

    def _build_safe_auto_plan(self, source: str) -> ExecutionPlan:
        """Build a one-slot safe AUTO plan for emergency fallback."""
        now_utc = dt_util.utcnow()
        return ExecutionPlan(
            slots=[
                ExecutionSlotCommand(
                    slot_start=now_utc,
                    p_bat_cmd=0,
                    soc_target=0,
                    grid_limit=0,
                    op_mode=OperationMode.AUTO,
                )
            ],
            step_minutes=_OPTIMIZATION_TIME_STEP_MINUTES,
            timestamp=now_utc,
            valid=False,
            source=source,
        )

    def _execution_plan_progress(
        self, plan: ExecutionPlan, now_utc: datetime
    ) -> tuple[int, int]:
        """Return consumed and total slots for a cached execution plan."""
        total_slots = len(plan.slots)
        consumed_slots = sum(1 for slot in plan.slots if slot.slot_start <= now_utc)
        return consumed_slots, total_slots

    async def _async_recover_from_publish_failure(
        self, reason: str, *, count_failure: bool = True
    ) -> None:
        """Recover after MPC failure by reusing cached plan or falling back to AUTO."""
        if count_failure:
            self._publish_failure_streak += 1
        self._last_deferrable_loads_applied = False
        now_utc = dt_util.utcnow()
        cached_plan = self._last_valid_execution_plan

        if cached_plan is not None and cached_plan.slots:
            consumed_slots, total_slots = self._execution_plan_progress(
                cached_plan, now_utc
            )
            half_threshold = (total_slots + 1) // 2
            if consumed_slots < half_threshold:
                _LOGGER.warning(
                    "Publish failed (%s). Reusing cached execution plan: failure_streak=%s consumed_slots=%s total_slots=%s half_threshold=%s",
                    reason,
                    self._publish_failure_streak,
                    consumed_slots,
                    total_slots,
                    half_threshold,
                )
                try:
                    exec_res = await self.executor.async_execute_plan(cached_plan)
                    self._last_executor_apply_result = exec_res
                    self._last_execution_applied = (
                        exec_res == ExecutorApplyResult.APPLIED
                    )
                    self._last_execution_plan = cached_plan
                    self._last_execution_utc = dt_util.utcnow()
                except (HomeAssistantError, RuntimeError, ValueError) as err:
                    self._last_execution_applied = False
                    _LOGGER.warning(
                        "Executor apply failed while reusing cached plan: %s", err
                    )
                return

            _LOGGER.warning(
                "Publish failed (%s) and cached execution plan reached half-life: failure_streak=%s consumed_slots=%s total_slots=%s half_threshold=%s. Switching to safe AUTO mode.",
                reason,
                self._publish_failure_streak,
                consumed_slots,
                total_slots,
                half_threshold,
            )
        else:
            _LOGGER.warning(
                "Publish failed (%s) and no cached valid execution plan exists. Switching to safe AUTO mode.",
                reason,
            )

        safe_plan = self._build_safe_auto_plan("publish_failure_safe_auto")
        try:
            exec_res = await self.executor.async_execute_plan(safe_plan)
            self._last_executor_apply_result = exec_res
            self._last_execution_applied = exec_res == ExecutorApplyResult.APPLIED
            self._last_execution_plan = safe_plan
            self._last_execution_utc = dt_util.utcnow()
        except (HomeAssistantError, RuntimeError, ValueError) as err:
            self._last_execution_applied = False
            _LOGGER.warning("Executor apply failed in safe AUTO fallback: %s", err)

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

    async def async_run_startup_ml_bootstrap(self) -> None:
        """Run startup ML bootstrap with a hard timeout budget."""
        load_entity = self.entry.data.get(CONF_CURRENT_CONSUMPTION_ENTITY)
        self._ml_bootstrap_completed = False
        self._ml_fit_completed = False

        if not load_entity:
            self._ml_bootstrap_completed = True
            self._log_step_ok("startup_ml_bootstrap", "no load entity configured")
            return

        if not await self.ml_forecast.async_has_sufficient_history(load_entity):
            self._ml_bootstrap_completed = True
            self._log_step_ok("startup_ml_bootstrap", "insufficient history")
            return

        if self._operation_lock.locked():
            self._ml_bootstrap_completed = True
            self._log_step_ok("startup_ml_bootstrap", "mpc operation in progress")
            return

        try:
            await asyncio.wait_for(self._operation_lock.acquire(), timeout=0.05)
        except TimeoutError:
            self._ml_bootstrap_completed = True
            self._log_step_ok("startup_ml_bootstrap", "mpc operation in progress")
            return

        try:
            self._log_step_start("startup_ml_bootstrap", load_entity)
            last_error: Exception | None = None
            for attempt in range(1, _ML_BOOTSTRAP_MAX_ATTEMPTS + 1):
                try:
                    async with asyncio.timeout(_ML_BOOTSTRAP_TIMEOUT_SECONDS):
                        fitted = await self.ml_forecast.async_train_model(
                            load_entity,
                            force=True,
                            optimization_time_step_minutes=_OPTIMIZATION_TIME_STEP_MINUTES,
                        )
                        if not fitted:
                            raise RuntimeError("fit failed")

                        self._ml_fit_completed = True
                        prediction_horizon = (
                            DEFAULT_HORIZON_HOURS * 60
                        ) // _OPTIMIZATION_TIME_STEP_MINUTES
                        predicted = await self.ml_forecast.async_predict_load(
                            load_entity,
                            prediction_horizon,
                            _OPTIMIZATION_TIME_STEP_MINUTES,
                        )
                        if predicted is None:
                            raise RuntimeError("predict failed")

                        self._log_step_ok(
                            "startup_ml_bootstrap",
                            f"fit+predict finished points={len(predicted)} on attempt {attempt}/{_ML_BOOTSTRAP_MAX_ATTEMPTS}",
                        )
                        return
                except TimeoutError as err:
                    last_error = err
                    failure_reason = (
                        f"timeout after {_ML_BOOTSTRAP_TIMEOUT_SECONDS}s "
                        f"(attempt {attempt}/{_ML_BOOTSTRAP_MAX_ATTEMPTS})"
                    )
                except Exception as err:
                    last_error = err
                    failure_reason = (
                        f"{err} (attempt {attempt}/{_ML_BOOTSTRAP_MAX_ATTEMPTS})"
                    )

                if attempt < _ML_BOOTSTRAP_MAX_ATTEMPTS:
                    _LOGGER.warning(
                        "Startup ML bootstrap failed: %s; retrying in %ss",
                        failure_reason,
                        _ML_BOOTSTRAP_RETRY_DELAY_SECONDS,
                    )
                    await asyncio.sleep(_ML_BOOTSTRAP_RETRY_DELAY_SECONDS)
                    continue

                if last_error is not None:
                    self._log_step_error("startup_ml_bootstrap", last_error)
                else:
                    self._log_step_ok(
                        "startup_ml_bootstrap",
                        "failed after maximum retry attempts",
                    )
        finally:
            self._ml_bootstrap_completed = True
            self._operation_lock.release()

    async def async_run_ml_daily_refresh(self) -> None:
        """Run daily ML refresh at midnight: tune+predict or fit+predict."""
        load_entity = self.entry.data.get(CONF_CURRENT_CONSUMPTION_ENTITY)
        if not load_entity:
            self._log_step_ok("ml_daily_refresh", "no load entity configured")
            return

        if not await self.ml_forecast.async_has_sufficient_history(load_entity):
            self._log_step_ok("ml_daily_refresh", "insufficient history")
            return

        if self._operation_lock.locked():
            self._log_step_ok("ml_daily_refresh", "mpc operation in progress")
            return

        try:
            await asyncio.wait_for(self._operation_lock.acquire(), timeout=0.05)
        except TimeoutError:
            self._log_step_ok("ml_daily_refresh", "mpc operation in progress")
            return

        try:
            self._log_step_start("ml_daily_refresh", load_entity)
            if self._ml_fit_completed:
                tuned = await self.ml_forecast.async_tune_model_once_daily(load_entity)
                if not tuned:
                    self._raise_update_failed("EMHASS ML daily tune failed")
            else:
                fitted = await self.ml_forecast.async_train_model(
                    load_entity,
                    force=True,
                    optimization_time_step_minutes=_OPTIMIZATION_TIME_STEP_MINUTES,
                )
                if not fitted:
                    self._raise_update_failed("EMHASS ML daily fit failed")
                self._ml_fit_completed = True

            prediction_horizon = (
                DEFAULT_HORIZON_HOURS * 60
            ) // _OPTIMIZATION_TIME_STEP_MINUTES
            predicted = await self.ml_forecast.async_predict_load(
                load_entity,
                prediction_horizon,
                _OPTIMIZATION_TIME_STEP_MINUTES,
            )
            if predicted is None:
                self._raise_update_failed("EMHASS ML daily predict failed")

            self._log_step_ok(
                "ml_daily_refresh",
                f"refresh finished points={len(predicted)}",
            )
        finally:
            self._operation_lock.release()

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
        pv_series: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compose coordinator payload shared by all entities."""
        published_entities = {
            key: entity.as_dict()
            for key, entity in self._last_published_entities.items()
        }

        if pv_series is None:
            pv_series = self._last_pv_forecast_series

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
            "pv_forecast_series": pv_series,
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

    def _build_pv_power_series(
        self,
        buckets: list[OptimizationBucket],
        values_kwh: list[float],
    ) -> list[dict[str, Any]]:
        """Convert per-bucket PV energy values to chart-friendly power points."""
        if not buckets or not values_kwh:
            return []

        bucket_hours = self._bucket_hours(buckets)
        if bucket_hours <= 0:
            return []

        series: list[dict[str, Any]] = []
        for bucket, pv_kwh in zip(buckets, values_kwh, strict=False):
            power_w = max(0.0, (pv_kwh / bucket_hours) * 1000.0)
            series.append(
                {
                    "date": dt_util.as_utc(bucket.start).isoformat(),
                    "power_w": round(power_w, 2),
                }
            )

        return series

    def _build_pv_forecast_series_payload(
        self,
        buckets: list[OptimizationBucket],
    ) -> dict[str, Any]:
        """Build PV forecast series for dashboards (single hourly-corrected line)."""
        corrected_kwh = [max(0.0, b.pv) for b in buckets]
        factors_snapshot = [round(float(x), 6) for x in self._pv_hourly_factors]

        return {
            "metadata": {
                "mode": "hourly24",
                "hourly_factors": factors_snapshot,
            },
            "series": self._build_pv_power_series(buckets, corrected_kwh),
        }

    async def _async_ensure_pv_hourly_factors_loaded(self) -> None:
        """Load persisted 24 hourly PV factors once per process."""
        if self._pv_hourly_factors_loaded:
            return
        data = await self._pv_hourly_store.async_load()
        if isinstance(data, dict):
            raw = data.get("factors")
            if isinstance(raw, list) and len(raw) == 24:
                try:
                    self._pv_hourly_factors = [float(x) for x in raw]
                except (TypeError, ValueError):
                    _LOGGER.debug("Invalid stored PV hourly factors, using defaults")
        self._pv_hourly_factors_loaded = True

    async def _async_save_pv_hourly_factors(self) -> None:
        """Persist 24 hourly PV factors."""
        await self._pv_hourly_store.async_save({"factors": self._pv_hourly_factors})

    async def _async_mean_solar_power_w_interval(
        self, entity_id: str, start: datetime, end: datetime
    ) -> float | None:
        """Mean PV power (W) over [start, end) from recorder, else current sensor."""
        start_utc = dt_util.as_utc(start)
        end_utc = dt_util.as_utc(end)
        lower = entity_id.lower()
        try:
            state_history = await get_instance(self.hass).async_add_executor_job(
                partial(
                    history.state_changes_during_period,
                    self.hass,
                    start_utc,
                    end_utc,
                    lower,
                    include_start_time_state=True,
                )
            )
        except (OSError, RuntimeError, ValueError) as err:
            _LOGGER.debug("Solar state history query failed: %s", err)
            state_history = {}

        rows = state_history.get(lower, [])
        values: list[float] = []
        for st in rows:
            w = self._normalize_power_w(st)
            if w is not None:
                values.append(float(w))
        if values:
            return sum(values) / len(values)

        state = await self._async_get_state_with_startup_wait(entity_id)
        if state is None:
            return None
        return self._normalize_power_w(state)

    async def _async_pv_hourly_learn_from_completed_slot(
        self,
        timeline: list[OptimizationBucket],
        tz,
        now: datetime,
        uncorrected_pv_kwh: list[float],
    ) -> None:
        """EWMA-update the hour factor for the slot that ended just before this horizon."""
        if not timeline or not uncorrected_pv_kwh:
            return
        completed_start = now - timedelta(minutes=_OPTIMIZATION_TIME_STEP_MINUTES)
        if self._last_timeline_first_start is None or self._last_pv_uncorrected_kwh is None:
            return
        if dt_util.as_utc(self._last_timeline_first_start) != dt_util.as_utc(
            completed_start
        ):
            _LOGGER.debug(
                "PV hourly learn skipped: previous horizon start %s != completed %s",
                self._last_timeline_first_start.isoformat(),
                completed_start.isoformat(),
            )
            return

        unc0 = float(self._last_pv_uncorrected_kwh[0])
        bucket_hours = self._bucket_hours(timeline)
        if bucket_hours <= 0:
            return
        forecast_w = max((unc0 / bucket_hours) * 1000.0, 0.0)

        entity_id = self.entry.data.get(CONF_CURRENT_SOLAR_PRODUCTION_ENTITY)
        if not entity_id:
            return

        slot_end = completed_start + timedelta(minutes=_OPTIMIZATION_TIME_STEP_MINUTES)
        actual_w = await self._async_mean_solar_power_w_interval(
            entity_id, completed_start, slot_end
        )
        if actual_w is None:
            return

        if (
            forecast_w < PV_HOURLY_MIN_POWER_W
            or actual_w < PV_HOURLY_MIN_POWER_W
            or forecast_w <= 0
        ):
            return

        h = int(completed_start.astimezone(tz).hour)
        ratio = max(actual_w, 0.0) / forecast_w
        ratio = min(max(ratio, PV_HOURLY_RATIO_MIN), PV_HOURLY_RATIO_MAX)
        new_f = pv_hourly_ewma_update(self._pv_hourly_factors[h], ratio)
        new_f = min(max(new_f, PV_HOURLY_FACTOR_MIN), PV_HOURLY_FACTOR_MAX)
        if new_f != self._pv_hourly_factors[h]:
            self._pv_hourly_factors[h] = new_f
            await self._async_save_pv_hourly_factors()
            _LOGGER.debug(
                "PV hourly factor updated: hour=%s ratio=%.4f new_factor=%.4f",
                h,
                ratio,
                new_f,
            )

    def _apply_pv_hourly_factors_to_timeline(
        self,
        timeline: list[OptimizationBucket],
        uncorrected_kwh: list[float],
        tz,
    ) -> None:
        """Scale each bucket's PV kWh using the factor for its local hour."""
        for bucket, u_kwh in zip(timeline, uncorrected_kwh, strict=False):
            h = int(bucket.start.astimezone(tz).hour)
            f_h = self._pv_hourly_factors[h]
            bucket.pv = max(0.0, float(u_kwh) * f_h)

    def _log_phoptimizer_timeline(
        self, timeline: list[OptimizationBucket], tz_name: str
    ) -> None:
        """One-line JSON of the final timeline sent to EMHASS (for offline MAE, etc.)."""
        payload = {
            "entry_id": self.entry.entry_id,
            "timezone": tz_name,
            "slots": [
                {
                    "slot_start": b.start.isoformat(),
                    "pv_kwh": round(b.pv, 6),
                    "load_kwh": round(b.load, 6),
                    "price": round(b.price, 6),
                }
                for b in timeline
            ],
        }
        _LOGGER.info(
            "PHOTOPTIMIZER_TIMELINE %s",
            json.dumps(payload, separators=(",", ":")),
        )

    def _reset_hourly_stats_for_window(self, window_hour: datetime) -> None:
        """Clear accumulators for a new calendar hour window."""
        self._stats_window_hour = window_hour
        self._stats_cycles = 0
        self._stats_opt_ok = 0
        self._stats_opt_fail = 0
        self._stats_pub_ok = 0
        self._stats_pub_fail = 0
        self._stats_opt_duration_ms = 0
        self._stats_pub_duration_ms = 0
        self._stats_exec_applied = 0
        self._stats_exec_skipped = 0

    async def _async_emit_hourly_stats(self) -> None:
        """Log aggregated MPC / executor stats for the completed hour window."""
        window = self._stats_window_hour
        if window is None:
            return
        c = self._stats_cycles
        payload = {
            "entry_id": self.entry.entry_id,
            "window_hour_utc": dt_util.as_utc(window).isoformat(),
            "cycles": c,
            "optimization_ok": self._stats_opt_ok,
            "optimization_fail": self._stats_opt_fail,
            "publish_ok": self._stats_pub_ok,
            "publish_fail": self._stats_pub_fail,
            "optimization_duration_ms_sum": self._stats_opt_duration_ms,
            "publish_duration_ms_sum": self._stats_pub_duration_ms,
            "optimization_duration_ms_avg": round(self._stats_opt_duration_ms / c, 1)
            if c
            else 0.0,
            "publish_duration_ms_avg": round(self._stats_pub_duration_ms / c, 1)
            if c
            else 0.0,
            "executor_commands_applied": self._stats_exec_applied,
            "executor_commands_skipped_duplicate": self._stats_exec_skipped,
        }
        _LOGGER.info(
            "PHOTOPTIMIZER_STATS %s",
            json.dumps(payload, separators=(",", ":")),
        )

    async def async_note_mpc_wave(
        self,
        *,
        optimization_ok: bool,
        publish_ok: bool,
        optimization_duration_ms: int,
        publish_duration_ms: int,
        executor_result: ExecutorApplyResult | None,
    ) -> None:
        """Accumulate MPC wave metrics; emit PHOTOPTIMIZER_STATS once per calendar hour."""
        tz_name = self.entry.data.get(CONF_TIMEZONE) or self.hass.config.time_zone
        tz = dt_util.get_time_zone(tz_name) or dt_util.UTC
        hour_key = dt_util.now(tz).replace(minute=0, second=0, microsecond=0)

        if self._stats_window_hour is not None and hour_key > self._stats_window_hour:
            await self._async_emit_hourly_stats()
            self._reset_hourly_stats_for_window(hour_key)

        if self._stats_window_hour is None:
            self._stats_window_hour = hour_key

        self._stats_cycles += 1
        if optimization_ok:
            self._stats_opt_ok += 1
        else:
            self._stats_opt_fail += 1
        if publish_ok:
            self._stats_pub_ok += 1
        else:
            self._stats_pub_fail += 1
        self._stats_opt_duration_ms += max(0, optimization_duration_ms)
        self._stats_pub_duration_ms += max(0, publish_duration_ms)
        if executor_result == ExecutorApplyResult.APPLIED:
            self._stats_exec_applied += 1
        elif executor_result == ExecutorApplyResult.SKIPPED_DUPLICATE:
            self._stats_exec_skipped += 1

    async def _async_collect_inputs(
        self,
    ) -> tuple[OptimizationInputs, Any | None, dict[str, Any]]:
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
            mapped_price_buckets = await self._price_from_entity(price_entity, timeline)
            if mapped_price_buckets <= 0:
                self._raise_update_failed(
                    f"Price entity {price_entity} contains no usable price points"
                )
        else:
            _LOGGER.debug("No electricity price entity configured")

        _LOGGER.debug("Using Forecast.Solar client for PV forecast")
        raw_pv = await self.client.estimate()
        await self._hourly_from_forecast_solar(timeline, raw_pv)
        uncorrected_pv_kwh = [float(bucket.pv) for bucket in timeline]
        await self._async_ensure_pv_hourly_factors_loaded()
        await self._async_pv_hourly_learn_from_completed_slot(
            timeline, tz, now, uncorrected_pv_kwh
        )
        self._apply_pv_hourly_factors_to_timeline(timeline, uncorrected_pv_kwh, tz)
        pv_series = self._build_pv_forecast_series_payload(timeline)

        load_entity = self.entry.data.get(CONF_CURRENT_CONSUMPTION_ENTITY)
        if load_entity:
            _LOGGER.debug(
                "Loading consumption forecast from published ML entity with profile fallback"
            )
            loaded_from_ml_entity = await self._async_hourly_from_published_ml_forecast(
                DEFAULT_ML_PREDICT_PUBLISH_ENTITY_ID,
                timeline,
            )
            if not loaded_from_ml_entity:
                load_profile = await self._build_load_profile(load_entity)
                await self._hourly_from_load_profile(timeline, load_profile)
        else:
            _LOGGER.debug("No load entity configured, using default profile")
            load_profile = await self._build_load_profile(None)
            await self._hourly_from_load_profile(timeline, load_profile)

        _LOGGER.debug(
            "Input collection finished with %s timeline buckets", len(timeline)
        )

        self._log_phoptimizer_timeline(timeline, tz_name)
        self._last_pv_uncorrected_kwh = list(uncorrected_pv_kwh)
        self._last_timeline_first_start = timeline[0].start if timeline else None

        return (
            OptimizationInputs(
                timeline=timeline,
                battery_soc=self._read_battery_soc(),
                deferrable_loads=self.deferrable_loads,
                raw_forecast_solar=raw_pv,
            ),
            raw_pv,
            pv_series,
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

    async def _price_from_entity(
        self, entity_id: str, buckets: list[OptimizationBucket]
    ) -> int:
        state = await self._async_get_state_with_startup_wait(entity_id)

        if state is None:
            raise UpdateFailed(f"Price entity {entity_id} not found")

        tz_name = self.entry.data.get(CONF_TIMEZONE) or self.hass.config.time_zone
        tz = dt_util.get_time_zone(tz_name) or dt_util.UTC

        exact_bucket_index: dict[datetime, list[OptimizationBucket]] = {}
        hour_bucket_index: dict[datetime, list[OptimizationBucket]] = {}
        for bucket in buckets:
            exact_bucket_index.setdefault(bucket.start, []).append(bucket)
            hour_start = bucket.start.replace(minute=0, second=0, microsecond=0)
            hour_bucket_index.setdefault(hour_start, []).append(bucket)
        mapped_buckets: set[datetime] = set()
        mapped_hours: set[datetime] = set()
        points_by_hour: dict[datetime, dict[datetime, float]] = {}

        for key, value in state.attributes.items():
            numeric = self._coerce_float(value)
            if numeric is None:
                continue

            dt = dt_util.parse_datetime(str(key))
            if dt is None:
                continue

            dt_local = dt_util.as_local(dt).astimezone(tz)
            exact_start = dt_local.replace(second=0, microsecond=0)
            hour_start = dt_local.replace(minute=0, second=0, microsecond=0)
            points_by_hour.setdefault(hour_start, {})[exact_start] = numeric

        for hour_start, points in points_by_hour.items():
            # If we have sub-hour points for an hour, use exact timestamps only.
            # If all points are exactly at HH:00, treat the value as hourly
            # and propagate it across all optimization buckets in that hour.
            has_subhour_points = any(
                point.minute != 0 or point.second != 0 or point.microsecond != 0
                for point in points
            )
            if has_subhour_points:
                for exact_start, numeric in points.items():
                    target_buckets = exact_bucket_index.get(exact_start)
                    if not target_buckets:
                        continue
                    for bucket in target_buckets:
                        bucket.price = numeric
                        mapped_buckets.add(bucket.start)
                    mapped_hours.add(hour_start)
                continue

            target_buckets = hour_bucket_index.get(hour_start)
            if not target_buckets:
                continue
            numeric = next(iter(points.values()))
            for bucket in target_buckets:
                bucket.price = numeric
                mapped_buckets.add(bucket.start)
            mapped_hours.add(hour_start)

        _LOGGER.debug(
            "Price timeline populated from %s: mapped_hours=%s mapped_buckets=%s",
            entity_id,
            len(mapped_hours),
            len(mapped_buckets),
        )
        return len(mapped_buckets)

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
                n = len(target_buckets)
                per_bucket_kwh = forecast_solar_hour_wh_to_per_bucket_kwh(float(wh), n)
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

    async def _async_hourly_from_published_ml_forecast(
        self,
        entity_id: str,
        buckets: list[OptimizationBucket],
    ) -> bool:
        """Fill load buckets from an already published ML forecast sensor."""
        if not buckets:
            return True

        state = await self._async_get_state_with_startup_wait(entity_id)
        if state is None:
            _LOGGER.debug(
                "Published ML forecast entity %s not available for load ingest",
                entity_id,
            )
            return False

        _LOGGER.debug(
            "Published ML forecast entity %s state=%s available_keys=%s",
            entity_id,
            state.state,
            sorted(state.attributes.keys()),
        )

        forecasts = None
        for attr_name in ("scheduled_forecast", "forecasts"):
            candidate = state.attributes.get(attr_name)
            if isinstance(candidate, list):
                forecasts = candidate
                _LOGGER.debug(
                    "Using ML forecast attribute %s with %s rows",
                    attr_name,
                    len(candidate),
                )
                break

        if not isinstance(forecasts, list):
            _LOGGER.debug(
                "Published ML forecast entity %s did not expose forecast list attributes",
                entity_id,
            )
            return False

        first_bucket_utc = dt_util.as_utc(buckets[0].start).replace(
            second=0, microsecond=0
        )
        values_by_slot_utc: dict[datetime, float] = {}
        timestamp_keys = ("date", "datetime", "time", "start", "timestamp", "ts")
        value_keys = (
            "p_load_forecast",
            "load_power_forecast",
            "value",
            "state",
            "forecast",
            "load",
            "power",
        )

        def _parse_row_timestamp(candidate: object) -> datetime | None:
            if isinstance(candidate, datetime):
                return dt_util.as_utc(candidate)

            parsed = dt_util.parse_datetime(str(candidate))
            if parsed is None:
                return None

            return dt_util.as_utc(parsed)

        def _extract_row_numeric(candidate: object) -> float | None:
            numeric = self._coerce_float(candidate)
            if numeric is not None:
                return numeric

            if isinstance(candidate, dict):
                for key in value_keys:
                    numeric = self._coerce_float(candidate.get(key))
                    if numeric is not None:
                        return numeric

                for key, value in candidate.items():
                    if key in timestamp_keys:
                        continue
                    numeric = _extract_row_numeric(value)
                    if numeric is not None:
                        return numeric

            if isinstance(candidate, (list, tuple)):
                for item in candidate:
                    numeric = _extract_row_numeric(item)
                    if numeric is not None:
                        return numeric

            return None

        for row in forecasts:
            parsed: datetime | None = None
            numeric: float | None = None

            if isinstance(row, dict):
                for key in timestamp_keys:
                    parsed = _parse_row_timestamp(row.get(key))
                    if parsed is not None:
                        break

                if parsed is None:
                    for key, value in row.items():
                        parsed = _parse_row_timestamp(key)
                        if parsed is not None:
                            numeric = _extract_row_numeric(value)
                            break

                if numeric is None:
                    for key in value_keys:
                        numeric = _extract_row_numeric(row.get(key))
                        if numeric is not None:
                            break

                if numeric is None:
                    for key, value in row.items():
                        if key in timestamp_keys:
                            continue
                        numeric = _extract_row_numeric(value)
                        if numeric is not None:
                            break

            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                parsed = _parse_row_timestamp(row[0])
                numeric = _extract_row_numeric(row[1])

                if parsed is None and len(row) > 2:
                    for item in row:
                        if parsed is None:
                            parsed = _parse_row_timestamp(item)
                        if numeric is None:
                            numeric = _extract_row_numeric(item)

            if parsed is None or numeric is None:
                continue

            slot_utc = parsed.replace(second=0, microsecond=0)
            if slot_utc < first_bucket_utc:
                continue

            if numeric is not None:
                values_by_slot_utc[slot_utc] = numeric

        _LOGGER.debug(
            "Parsed published ML forecast for %s: parsed_points=%s first_bucket=%s",
            entity_id,
            len(values_by_slot_utc),
            first_bucket_utc.isoformat(),
        )

        if not values_by_slot_utc and forecasts:
            _LOGGER.debug(
                "Published ML forecast parse yielded no points for %s: first_row_type=%s first_row_sample=%s",
                entity_id,
                type(forecasts[0]).__name__,
                forecasts[0],
            )

        if len(values_by_slot_utc) < len(buckets):
            _LOGGER.debug(
                "Published ML forecast too short or sparse for %s: parsed_points=%s needed=%s",
                entity_id,
                len(values_by_slot_utc),
                len(buckets),
            )
            return False

        bucket_hours = self._bucket_hours(buckets)
        for bucket in buckets:
            bucket_slot_utc = dt_util.as_utc(bucket.start).replace(
                second=0, microsecond=0
            )
            bucket_value_w = values_by_slot_utc.get(bucket_slot_utc)
            if bucket_value_w is None:
                return False
            bucket.load = (bucket_value_w * bucket_hours) / 1000.0

        _LOGGER.debug(
            "Load timeline populated from published ML entity %s: points=%s",
            entity_id,
            len(buckets),
        )
        return True

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

        start = dt_util.utcnow() - timedelta(days=_LOAD_PROFILE_HISTORY_WINDOW_DAYS)
        history_profile = await self._build_load_profile_from_state_history(
            entity_id,
            start,
            default_profile,
        )
        if history_profile is not None:
            return history_profile

        _LOGGER.debug(
            "No usable state history found for %s in the last %s days, using default load profile",
            entity_id,
            _LOAD_PROFILE_HISTORY_WINDOW_DAYS,
        )
        return default_profile

    def _read_battery_soc(self) -> float:
        """Read and normalize the current battery SOC from Home Assistant."""
        soc_entity = self.entry.data.get(CONF_BATTERY_SOC_ENTITY)
        soc_state = self.hass.states.get(soc_entity) if soc_entity else None
        raw_soc = self._coerce_float(soc_state.state) if soc_state is not None else None

        if raw_soc is not None:
            normalized_soc = raw_soc if raw_soc <= 1.0 else raw_soc / 100.0
            normalized_soc = max(0.0, min(1.0, normalized_soc))
            self._last_valid_battery_soc = normalized_soc
        elif self._last_valid_battery_soc is not None:
            normalized_soc = self._last_valid_battery_soc
            _LOGGER.warning(
                "Battery SOC entity %s unavailable or invalid, reusing last valid value %s",
                soc_entity,
                normalized_soc,
            )
        else:
            raise UpdateFailed(
                f"Battery SOC entity {soc_entity} is unavailable or invalid and no last valid value exists"
            )

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
                (
                    optimization_inputs,
                    raw_pv,
                    pv_series,
                ) = await self._async_collect_inputs()
                self._last_inputs_snapshot = optimization_inputs
                self._last_raw_pv_snapshot = raw_pv
                self._last_pv_forecast_series = pv_series
                if _LOGGER.isEnabledFor(logging.DEBUG):
                    self._log_timeline(optimization_inputs.timeline)
                result = self._build_result(optimization_inputs, raw_pv, pv_series)
                _LOGGER.debug(
                    "Coordinator refresh finished: horizon=%s step=%s",
                    optimization_inputs.prediction_horizon,
                    optimization_inputs.optimization_time_step_minutes,
                )
                return result

        except ForecastSolarError as err:
            raise UpdateFailed(f"Forecast.Solar API error: {err}") from err

    async def async_run_mpc_optimization(self) -> None:
        """Run one naive MPC optimization call to EMHASS."""
        async with self._operation_lock:
            self._log_step_start("mpc_optimization")
            if not self._optimizer_enabled:
                self._log_step_ok("mpc_optimization", "optimizer disabled, skipped")
                return
            try:
                async with asyncio.timeout(90):
                    (
                        optimization_inputs,
                        raw_pv,
                        pv_series,
                    ) = await self._async_collect_inputs()
                    self._last_inputs_snapshot = optimization_inputs
                    self._last_raw_pv_snapshot = raw_pv
                    self._last_pv_forecast_series = pv_series
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
                        self._build_result(optimization_inputs, raw_pv, pv_series)
                    )
                    self._log_step_ok(
                        "mpc_optimization",
                        f"runtimeparams_keys={sorted(self._last_runtimeparams.keys())} "
                        f"response_keys={sorted(self._last_optimization_response.keys())}",
                    )
            except ForecastSolarError as err:
                update_failed = UpdateFailed(f"Forecast.Solar API error: {err}")
                self._log_step_error("mpc_optimization", update_failed)
                await self._async_recover_from_publish_failure(
                    str(update_failed), count_failure=False
                )
                raise update_failed from err
            except UpdateFailed as err:
                self._log_step_error("mpc_optimization", err)
                await self._async_recover_from_publish_failure(
                    str(err), count_failure=False
                )
                raise
            except TimeoutError as err:
                update_failed = UpdateFailed("MPC optimization timed out")
                self._log_step_error("mpc_optimization", update_failed)
                await self._async_recover_from_publish_failure(
                    str(update_failed), count_failure=False
                )
                raise update_failed from err
            except Exception as err:
                update_failed = UpdateFailed(
                    f"MPC optimization unexpected error: {err}"
                )
                _LOGGER.exception("[ERR] mpc_optimization unexpected exception")
                await self._async_recover_from_publish_failure(
                    str(update_failed), count_failure=False
                )
                raise update_failed from err

    async def async_run_mpc_publish(self) -> None:
        """Run one publish-data step for the latest MPC output."""
        async with self._operation_lock:
            self._log_step_start("mpc_publish")
            if not self._optimizer_enabled:
                self._log_step_ok("mpc_publish", "optimizer disabled, skipped")
                return
            try:
                async with asyncio.timeout(60):
                    publish_result = await self.emhass.async_publish_data(
                        _OPTIMIZATION_TIME_STEP_MINUTES
                    )
                    if publish_result is None:
                        self._raise_update_failed("EMHASS MPC publish-data failed")

                    self._last_publish_response = publish_result["publish_response"]
                    self._last_published_entities = publish_result["published_entities"]
                    self._last_execution_plan = publish_result["execution_plan"]
                    self._last_publish_utc = dt_util.utcnow()
                    self._publish_failure_streak = 0

                    if (
                        self._last_execution_plan is not None
                        and self._last_execution_plan.valid
                        and self._last_execution_plan.slots
                    ):
                        self._last_valid_execution_plan = self._last_execution_plan

                    if self._last_execution_plan is not None:
                        _LOGGER.info(
                            "Publish produced execution plan: valid=%s source=%s slots=%s step=%s",
                            self._last_execution_plan.valid,
                            self._last_execution_plan.source,
                            len(self._last_execution_plan.slots),
                            self._last_execution_plan.step_minutes,
                        )

                    try:
                        _LOGGER.debug("Executor apply phase started")
                        executor_apply_failed = False
                        exec_res = await self.executor.async_execute_plan(
                            self._last_execution_plan
                        )
                        self._last_executor_apply_result = exec_res
                        self._last_execution_applied = (
                            exec_res == ExecutorApplyResult.APPLIED
                        )
                        self._last_deferrable_loads_applied = (
                            await self.executor.async_execute_deferrable_loads(
                                self._last_published_entities,
                                self.deferrable_loads,
                            )
                        )
                        self._last_execution_utc = dt_util.utcnow()
                        _LOGGER.info(
                            "Executor apply phase finished: plan_applied=%s deferrable_loads_applied=%s",
                            self._last_execution_applied,
                            self._last_deferrable_loads_applied,
                        )
                    except (
                        HomeAssistantError,
                        RuntimeError,
                        ValueError,
                    ) as err:
                        executor_apply_failed = True
                        self._last_executor_apply_result = None
                        self._last_execution_applied = False
                        self._last_deferrable_loads_applied = False
                        _LOGGER.warning("Executor apply failed: %s", err)

                    if (
                        _LOGGER.isEnabledFor(logging.DEBUG)
                        and self._last_inputs_snapshot is not None
                    ):
                        self._log_timeline(self._last_inputs_snapshot.timeline)

                    snapshot_inputs = self._last_inputs_snapshot
                    if snapshot_inputs is None:
                        snapshot_inputs = OptimizationInputs(
                            timeline=[],
                            battery_soc=self._read_battery_soc(),
                            deferrable_loads=self.deferrable_loads,
                        )

                    self.async_set_updated_data(
                        self._build_result(snapshot_inputs, self._last_raw_pv_snapshot)
                    )
                    if executor_apply_failed:
                        _LOGGER.warning(
                            "Executor apply phase finished with failure: published_entities=%s response_keys=%s",
                            sorted(self._last_published_entities.keys()),
                            sorted(self._last_publish_response.keys()),
                        )
                    else:
                        self._log_step_ok(
                            "mpc_publish",
                            f"published_entities={sorted(self._last_published_entities.keys())} "
                            f"response_keys={sorted(self._last_publish_response.keys())}",
                        )
            except ForecastSolarError as err:
                update_failed = UpdateFailed(f"Forecast.Solar API error: {err}")
                self._log_step_error("mpc_publish", update_failed)
                await self._async_recover_from_publish_failure(str(update_failed))
                raise update_failed from err
            except UpdateFailed as err:
                self._log_step_error("mpc_publish", err)
                await self._async_recover_from_publish_failure(str(err))
                raise
            except TimeoutError as err:
                update_failed = UpdateFailed("MPC publish timed out")
                self._log_step_error("mpc_publish", update_failed)
                await self._async_recover_from_publish_failure(str(update_failed))
                raise update_failed from err
            except Exception as err:
                update_failed = UpdateFailed(f"MPC publish unexpected error: {err}")
                _LOGGER.exception("[ERR] mpc_publish unexpected exception")
                await self._async_recover_from_publish_failure(str(update_failed))
                raise update_failed from err
