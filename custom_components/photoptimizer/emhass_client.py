"""Client for EMHASS communication."""

from __future__ import annotations

import asyncio
from datetime import datetime
import logging
import math
from numbers import Real
import time
from typing import Any

from aiohttp import ClientError, ClientTimeout

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util import dt as dt_util

from .models import (
    DeferrableLoadDefinition,
    EmhassExecutionResult,
    ExecutionPlan,
    ExecutionSlotCommand,
    OperationMode,
    OptimizationInputs,
    PublishedEntityState,
)

_LOGGER = logging.getLogger(__name__)

DEFAULT_BATTERY_CHARGE_POWER_MAX = 1000.0
DEFAULT_BATTERY_DISCHARGE_POWER_MAX = 1000.0
DEFAULT_BATTERY_MAXIMUM_STATE_OF_CHARGE = 0.9
DEFAULT_ML_HISTORIC_DAYS = 9
DEFAULT_ML_MODEL_TYPE = "photoptimizer_load"
DEFAULT_ML_SKLEARN_MODEL = "RandomForestRegressor"
DEFAULT_ML_NUM_LAGS = 96
DEFAULT_ML_SPLIT_DATE_DELTA = "48h"
DEFAULT_ML_BASE_STEP_MINUTES = 15
DEFAULT_ML_PREDICT_PUBLISH_ENTITY_ID = "sensor.p_load_forecast_custom_model"
DEFAULT_ML_PREDICT_PUBLISH_UNIT = "W"
DEFAULT_ML_PREDICT_PUBLISH_NAME = "Load Power Forecast custom ML model"
DEFAULT_ML_PREDICT_READBACK_TIMEOUT_SECONDS = 20
DEFAULT_ML_PREDICT_READBACK_POLL_SECONDS = 0.5
_SLOT_POWER_THRESHOLD_W = 50.0


class EmhassClient:
    """The complete EMHASS workflow for one Photoptimizer config entry.

    The coordinator prepares neutral inputs only.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        url: str,
        token: str | None = None,
        *,
        battery_capacity_kwh: float,
        battery_efficiency: float,
        battery_soc_reserve: float,
        battery_target_soc: float,
        battery_charge_power_max_w: float,
        battery_discharge_power_max_w: float,
        wear_cost_per_kwh: float,
        deferrable_loads: list[DeferrableLoadDefinition] | None = None,
    ) -> None:
        """Store session, base URL, publish targets, and runtime defaults."""
        self._hass = hass
        self._url = url.rstrip("/")
        self._session = async_get_clientsession(hass)
        self._token = token

        self._battery_charge_power_max = (
            battery_charge_power_max_w
            if battery_charge_power_max_w > 0
            else DEFAULT_BATTERY_CHARGE_POWER_MAX
        )
        self._battery_discharge_power_max = (
            battery_discharge_power_max_w
            if battery_discharge_power_max_w > 0
            else DEFAULT_BATTERY_DISCHARGE_POWER_MAX
        )

        self._battery_nominal_energy_capacity = (
            battery_capacity_kwh * 1000.0 if battery_capacity_kwh > 0 else 5000.0
        )

        self._battery_efficiency = min(
            max(battery_efficiency, 0.01),
            1.0,
        )
        self._battery_soc_reserve = min(
            max(battery_soc_reserve, 0.0),
            DEFAULT_BATTERY_MAXIMUM_STATE_OF_CHARGE,
        )
        self._battery_target_soc = min(
            max(battery_target_soc, self._battery_soc_reserve),
            DEFAULT_BATTERY_MAXIMUM_STATE_OF_CHARGE,
        )
        self._wear_cost_per_kwh = wear_cost_per_kwh
        self._deferrable_loads = deferrable_loads or []

        self._published_entities: dict[str, dict[str, str]] = {
            "pv_forecast": {
                "entity_id": "sensor.p_pv_forecast",
                "unit_of_measurement": "W",
                "friendly_name": "PV Power Forecast",
            },
            "load_forecast": {
                "entity_id": "sensor.p_load_forecast",
                "unit_of_measurement": "W",
                "friendly_name": "Load Power Forecast",
            },
            "battery_forecast": {
                "entity_id": "sensor.p_batt_forecast",
                "unit_of_measurement": "W",
                "friendly_name": "Battery Power Forecast",
            },
            "battery_soc_forecast": {
                "entity_id": "sensor.soc_batt_forecast",
                "unit_of_measurement": "%",
                "friendly_name": "Battery SOC Forecast",
            },
            "grid_forecast": {
                "entity_id": "sensor.p_grid_forecast",
                "unit_of_measurement": "W",
                "friendly_name": "Grid Power Forecast",
            },
            "unit_load_cost": {
                "entity_id": "sensor.unit_load_cost",
                "unit_of_measurement": "currency/kWh",
                "friendly_name": "Unit Load Cost",
            },
            "unit_prod_price": {
                "entity_id": "sensor.unit_prod_price",
                "unit_of_measurement": "currency/kWh",
                "friendly_name": "Unit Prod Price",
            },
            "cost_fun": {
                "entity_id": "sensor.total_cost_fun_value",
                "unit_of_measurement": "currency",
                "friendly_name": "Total cost function value",
            },
            "optim_status": {
                "entity_id": "sensor.optim_status",
                "unit_of_measurement": "",
                "friendly_name": "EMHASS optimization status",
            },
        }

        for index, load in enumerate(self._deferrable_loads):
            self._published_entities[f"deferrable_load_{index}"] = {
                "entity_id": f"sensor.p_deferrable{index}",
                "unit_of_measurement": "W",
                "friendly_name": load.name,
            }

        _LOGGER.debug(
            "EMHASS client initialized: url=%s token=%s battery_capacity_wh=%s efficiency=%s soc_reserve=%s soc_target=%s charge_max_w=%s discharge_max_w=%s wear_cost=%s",
            self._url,
            "set" if self._token else "unset",
            self._battery_nominal_energy_capacity,
            self._battery_efficiency,
            self._battery_soc_reserve,
            self._battery_target_soc,
            self._battery_charge_power_max,
            self._battery_discharge_power_max,
            self._wear_cost_per_kwh,
        )

    def _deferrable_load_runtimeparams(self, step_minutes: int) -> dict[str, Any]:
        """Build runtime parameters for configured deferrable loads."""
        if not self._deferrable_loads:
            return {"number_of_deferrable_loads": 0}

        operating_timesteps = [
            math.ceil(load.operating_minutes / step_minutes)
            for load in self._deferrable_loads
        ]
        def_current_state = []
        for load in self._deferrable_loads:
            state = self._hass.states.get(load.entity_id)
            def_current_state.append(
                bool(state is not None and str(state.state).lower() == "on")
            )

        return {
            "number_of_deferrable_loads": len(self._deferrable_loads),
            "nominal_power_of_deferrable_loads": [
                load.nominal_power_w for load in self._deferrable_loads
            ],
            "operating_timesteps_of_each_deferrable_load": operating_timesteps,
            "def_current_state": def_current_state,
        }

    def _deferrable_load_publish_payload(self) -> list[dict[str, Any]]:
        """Build publish-data config entries for configured deferrable loads."""
        return [{} for _ in self._deferrable_loads]

    def _headers(self) -> dict[str, str]:
        """Return headers for EMHASS HTTP requests."""
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def _async_check_url(self, url: str, label: str) -> bool:
        """Lightweight reachability check before hitting EMHASS."""
        _LOGGER.debug("Checking EMHASS reachability for %s at %s", label, url)
        try:
            async with self._session.get(
                url,
                headers=self._headers(),
                timeout=ClientTimeout(total=5),
            ) as response:
                _LOGGER.debug(
                    "Reachability check status for %s: %s", label, response.status
                )
                if response.status < 500:
                    return True
                _LOGGER.error("%s unreachable (%s)", label, response.status)
        except ClientError as err:
            _LOGGER.error("%s connection error at %s: %s", label, url, err)
        except (OSError, RuntimeError, ValueError) as err:
            _LOGGER.error("%s unexpected error at %s: %s", label, url, err)
        return False

    async def _async_post_action(
        self,
        action: str,
        payload: dict[str, Any],
        *,
        timeout: int,
    ) -> dict[str, Any] | None:
        """Post one action to the EMHASS web server.

        Returns parsed JSON/dict-like response on success and ``None`` on hard failures.
        """
        endpoint = f"{self._url}/action/{action}"
        _LOGGER.debug(
            "Calling EMHASS action=%s endpoint=%s payload_keys=%s timeout=%s",
            action,
            endpoint,
            sorted(payload.keys()),
            timeout,
        )

        try:
            async with self._session.post(
                endpoint,
                json=payload,
                headers=self._headers(),
                timeout=ClientTimeout(total=timeout),
            ) as response:
                text = await response.text()

                if response.status not in (200, 201):
                    if response.status >= 500:
                        _LOGGER.error(
                            "EMHASS internal error at %s. Check the EMHASS add-on logs for the Python traceback",
                            endpoint,
                        )
                        _LOGGER.error("EMHASS error %s: %s", response.status, text)
                        return None
                    _LOGGER.warning(
                        "EMHASS %s returned %s (treating as partial success): %s",
                        action,
                        response.status,
                        text,
                    )

                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    data = await response.json()
                    if isinstance(data, dict):
                        _LOGGER.debug("EMHASS %s response keys: %s", action, list(data))
                        return data
                    return {"data": data, "status": response.status}

                _LOGGER.debug("EMHASS %s success %s: %s", action, response.status, text)
                return {"message": text, "status": response.status}
        except ClientError as err:
            _LOGGER.error("EMHASS connection error at %s: %s", endpoint, err)
            return None
        except (OSError, RuntimeError, ValueError) as err:
            _LOGGER.error("EMHASS unexpected error at %s: %s", endpoint, err)
            return None

    def _coerce_float(self, value: object) -> float | None:
        """Convert a value to float when possible."""
        if isinstance(value, str) or (
            isinstance(value, Real) and not isinstance(value, bool)
        ):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    def _extract_numeric_list(self, value: object) -> list[float]:
        """Extract numeric values from nested JSON structures."""
        if isinstance(value, list):
            result: list[float] = []
            for item in value:
                numeric = self._coerce_float(item)
                if numeric is None:
                    return []
                result.append(numeric)
            return result

        if isinstance(value, dict):
            for nested in value.values():
                extracted = self._extract_numeric_list(nested)
                if extracted:
                    return extracted

        return []

    def _extract_load_forecast_from_response(
        self,
        response: dict[str, Any],
        expected_points: int,
    ) -> list[float]:
        """Extract load forecast values from flexible EMHASS response shapes."""
        candidate_keys = (
            "load_power_forecast",
            "P_Load",
            "y_pred",
            "prediction",
            "predictions",
            "data",
        )

        for key in candidate_keys:
            if key not in response:
                continue
            values = self._extract_numeric_list(response[key])
            if len(values) >= expected_points:
                return values[:expected_points]

        values = self._extract_numeric_list(response)
        if len(values) >= expected_points:
            return values[:expected_points]

        return []

    def _ml_num_lags_for_step(self, optimization_time_step_minutes: int) -> int:
        """Return one-day lag count for a given optimization step."""
        if optimization_time_step_minutes <= 0:
            return DEFAULT_ML_NUM_LAGS

        return max(1, int((24 * 60) / optimization_time_step_minutes))

    def _extract_forecast_from_ml_publish_attributes(
        self,
        attributes: dict[str, Any],
        expected_points: int,
    ) -> list[float]:
        """Extract ML forecast values from a published Home Assistant sensor."""
        table = attributes.get("forecasts")
        if isinstance(table, list):
            schedule = self._extract_schedule(table, value_key="p_load_forecast")
            values = [value for _, value in schedule]
            if len(values) >= expected_points:
                return values[:expected_points]

            # Fallback for custom key names emitted by some EMHASS versions.
            values = []
            for row in table:
                if not isinstance(row, dict):
                    continue

                numeric: float | None = None
                for row_key, row_value in row.items():
                    if row_key == "date":
                        continue
                    numeric = self._coerce_float(row_value)
                    if numeric is not None:
                        break

                if numeric is not None:
                    values.append(numeric)

            if len(values) >= expected_points:
                return values[:expected_points]

        # Final defensive fallback for unknown attribute layouts.
        values = self._extract_numeric_list(attributes)
        if len(values) >= expected_points:
            return values[:expected_points]

        return []

    async def _async_read_ml_predict_sensor(
        self,
        *,
        entity_id: str,
        expected_points: int,
        timeout_seconds: int = DEFAULT_ML_PREDICT_READBACK_TIMEOUT_SECONDS,
        poll_seconds: float = DEFAULT_ML_PREDICT_READBACK_POLL_SECONDS,
    ) -> list[float] | None:
        """Read ML predict values from a published Home Assistant sensor."""
        deadline = time.monotonic() + timeout_seconds

        while time.monotonic() < deadline:
            state = self._hass.states.get(entity_id)
            if state is not None:
                values = self._extract_forecast_from_ml_publish_attributes(
                    dict(state.attributes),
                    expected_points,
                )
                if values:
                    _LOGGER.debug(
                        "Read ML predict values from %s: points=%s",
                        entity_id,
                        len(values),
                    )
                    return values

            await self._hass.async_block_till_done()
            await asyncio.sleep(poll_seconds)

        _LOGGER.debug(
            "Timed out waiting for ML predict sensor=%s points=%s",
            entity_id,
            expected_points,
        )
        return None

    def _build_runtimeparams(self, inputs: OptimizationInputs) -> dict[str, Any]:
        """Translate aggregated coordinator data into EMHASS runtimeparams."""
        soc_init = min(
            max(inputs.battery_soc, self._battery_soc_reserve),
            DEFAULT_BATTERY_MAXIMUM_STATE_OF_CHARGE,
        )

        runtimeparams: dict[str, Any] = {
            "pv_power_forecast": [bucket.pv * 1000.0 for bucket in inputs.timeline],
            "load_power_forecast": [bucket.load * 1000.0 for bucket in inputs.timeline],
            "load_cost_forecast": [bucket.price for bucket in inputs.timeline],
            "prod_price_forecast": [bucket.price * 0.9 for bucket in inputs.timeline],
            "set_use_pv": True,
            "set_use_battery": True,
            "battery_discharge_power_max": self._battery_discharge_power_max,
            "battery_charge_power_max": self._battery_charge_power_max,
            "battery_discharge_efficiency": self._battery_efficiency,
            "battery_charge_efficiency": self._battery_efficiency,
            "battery_nominal_energy_capacity": self._battery_nominal_energy_capacity,
            "prediction_horizon": inputs.prediction_horizon,
            "optimization_time_step": inputs.optimization_time_step_minutes,
            "soc_init": soc_init,
            "soc_final": self._battery_target_soc,
            "battery_target_state_of_charge": self._battery_target_soc,
            "battery_minimum_state_of_charge": self._battery_soc_reserve,
            "battery_maximum_state_of_charge": DEFAULT_BATTERY_MAXIMUM_STATE_OF_CHARGE,
            "continual_publish": False,
        }

        runtimeparams.update(
            self._deferrable_load_runtimeparams(inputs.optimization_time_step_minutes)
        )

        if self._wear_cost_per_kwh > 0:
            runtimeparams["weight_battery_charge"] = self._wear_cost_per_kwh
            runtimeparams["weight_battery_discharge"] = self._wear_cost_per_kwh

        _LOGGER.debug(
            "Built runtimeparams: horizon=%s step=%s keys=%s",
            inputs.prediction_horizon,
            inputs.optimization_time_step_minutes,
            sorted(runtimeparams.keys()),
        )
        return runtimeparams

    def _build_publish_payload(
        self, optimization_time_step_minutes: int
    ) -> dict[str, Any]:
        """Build the minimal publish-data payload."""
        payload: dict[str, Any] = {
            "optimization_time_step": optimization_time_step_minutes,
            "set_use_battery": True,
        }
        payload.update(
            self._deferrable_load_runtimeparams(optimization_time_step_minutes)
        )
        payload["def_load_config"] = self._deferrable_load_publish_payload()
        _LOGGER.debug(
            "Built publish payload: optimization_time_step=%s keys=%s",
            optimization_time_step_minutes,
            sorted(payload.keys()),
        )
        return payload

    def _read_published_entities(self) -> dict[str, PublishedEntityState]:
        """Read EMHASS-published entities from the local HA state machine."""
        snapshots: dict[str, PublishedEntityState] = {}

        for key, descriptor in self._published_entities.items():
            entity_id = descriptor["entity_id"]
            state = self._hass.states.get(entity_id)
            snapshots[key] = PublishedEntityState(
                entity_id=entity_id,
                state=None if state is None else state.state,
                attributes={} if state is None else dict(state.attributes),
            )

        _LOGGER.debug(
            "Read published entities snapshot: available=%s total=%s",
            sum(1 for entity in snapshots.values() if entity.state is not None),
            len(snapshots),
        )
        return snapshots

    def _log_published_entities(
        self, published_entities: dict[str, PublishedEntityState]
    ) -> None:
        """Log a compact summary of entities published by EMHASS."""
        _LOGGER.debug(
            "EMHASS published entities:\n%s",
            "\n".join(
                f"  {key}: {entity.state} ({entity.entity_id})\n    attributes: {entity.attributes}\n"
                for key, entity in published_entities.items()
            ),
        )

    def _extract_schedule(
        self,
        table: object,
        *,
        value_key: str,
    ) -> list[tuple[datetime, float]]:
        """Extract sorted schedule from flexible EMHASS dict/list shapes."""
        schedule: list[tuple[datetime, float]] = []
        now_utc = dt_util.utcnow().replace(second=0, microsecond=0)

        if isinstance(table, dict):
            for key, value in table.items():
                parsed = dt_util.parse_datetime(str(key))
                numeric = self._coerce_float(value)
                if parsed is None or numeric is None:
                    continue

                parsed_utc = dt_util.as_utc(parsed)
                if parsed_utc < now_utc:
                    continue

                schedule.append((parsed_utc, numeric))

        elif isinstance(table, list):
            for row in table:
                if not isinstance(row, dict):
                    continue

                parsed = dt_util.parse_datetime(str(row.get("date")))
                numeric = self._coerce_float(row.get(value_key))
                if numeric is None:
                    for row_key, row_value in row.items():
                        if row_key == "date":
                            continue
                        numeric = self._coerce_float(row_value)
                        if numeric is not None:
                            break

                if parsed is None or numeric is None:
                    continue

                parsed_utc = dt_util.as_utc(parsed)
                if parsed_utc < now_utc:
                    continue

                schedule.append((parsed_utc, numeric))

        schedule.sort(key=lambda item: item[0])
        return schedule

    def _build_execution_plan(
        self,
        published_entities: dict[str, PublishedEntityState],
        *,
        optimization_time_step_minutes: int,
    ) -> ExecutionPlan:
        """Build normalized execution plan from EMHASS published entities."""
        now_utc = dt_util.utcnow()
        optim_status = (
            (
                (
                    published_entities.get("optim_status")
                    or PublishedEntityState("", None, {})
                ).state
                or ""
            )
            .strip()
            .lower()
        )

        if optim_status != "optimal":
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
                step_minutes=optimization_time_step_minutes,
                timestamp=now_utc,
                valid=False,
                source="emhass_publish_status",
            )

        battery_entity = published_entities.get("battery_forecast")
        if battery_entity is None:
            return ExecutionPlan(
                slots=[],
                step_minutes=optimization_time_step_minutes,
                timestamp=now_utc,
                valid=False,
                source="emhass_publish_missing_battery",
            )

        battery_attrs = (
            battery_entity.attributes
            if isinstance(battery_entity.attributes, dict)
            else {}
        )
        battery_table = battery_attrs.get("p_batt_forecast")
        if not isinstance(battery_table, dict | list):
            battery_table = battery_attrs.get("battery_scheduled_power")
        if not isinstance(battery_table, dict | list):
            battery_table = battery_attrs

        power_schedule = self._extract_schedule(
            battery_table,
            value_key="p_batt_forecast",
        )

        soc_entity = published_entities.get("battery_soc_forecast")
        soc_schedule: list[tuple[datetime, float]] = []
        soc_target_default = 0
        if soc_entity is not None:
            soc_raw = self._coerce_float(soc_entity.state)
            if soc_raw is not None:
                if soc_raw <= 1.0:
                    soc_raw *= 100.0
                soc_target_default = int(max(0, min(100, round(soc_raw))))

            soc_attrs = (
                soc_entity.attributes if isinstance(soc_entity.attributes, dict) else {}
            )
            soc_table = soc_attrs.get("soc_batt_forecast")
            if not isinstance(soc_table, dict | list):
                soc_table = soc_attrs
            soc_schedule = self._extract_schedule(
                soc_table,
                value_key="soc_batt_forecast",
            )

        slots: list[ExecutionSlotCommand] = []
        if power_schedule:
            for index, (slot_start, power_w) in enumerate(power_schedule):
                p_bat_cmd = int(round(power_w))
                if p_bat_cmd > _SLOT_POWER_THRESHOLD_W:
                    op_mode = OperationMode.FORCED_DISCHARGE
                elif p_bat_cmd < -_SLOT_POWER_THRESHOLD_W:
                    op_mode = OperationMode.FORCED_CHARGE
                else:
                    p_bat_cmd = 0
                    op_mode = OperationMode.IDLE

                if index < len(soc_schedule):
                    soc_value = soc_schedule[index][1]
                    if soc_value <= 1.0:
                        soc_value *= 100.0
                    soc_target = int(max(0, min(100, round(soc_value))))
                else:
                    soc_target = soc_target_default

                slots.append(
                    ExecutionSlotCommand(
                        slot_start=slot_start,
                        p_bat_cmd=p_bat_cmd,
                        soc_target=soc_target,
                        grid_limit=0,
                        op_mode=op_mode,
                    )
                )
        else:
            current_power_w: float | None = self._coerce_float(battery_entity.state)
            if current_power_w is not None:
                p_bat_cmd = int(round(current_power_w))
                if p_bat_cmd > _SLOT_POWER_THRESHOLD_W:
                    op_mode = OperationMode.FORCED_DISCHARGE
                elif p_bat_cmd < -_SLOT_POWER_THRESHOLD_W:
                    op_mode = OperationMode.FORCED_CHARGE
                else:
                    p_bat_cmd = 0
                    op_mode = OperationMode.IDLE

                slots.append(
                    ExecutionSlotCommand(
                        slot_start=now_utc,
                        p_bat_cmd=p_bat_cmd,
                        soc_target=soc_target_default,
                        grid_limit=0,
                        op_mode=op_mode,
                    )
                )

        return ExecutionPlan(
            slots=slots,
            step_minutes=optimization_time_step_minutes,
            timestamp=now_utc,
            valid=bool(slots),
            source="emhass_publish_entities",
        )

    async def async_run_naive_optimization(
        self, inputs: OptimizationInputs
    ) -> dict[str, Any] | None:
        """Run only the EMHASS naive optimization step."""
        _LOGGER.debug("Starting EMHASS naive optimization")
        if not await self._async_check_url(self._url, "EMHASS base"):
            _LOGGER.debug("EMHASS naive optimization aborted: base URL unreachable")
            return None

        runtimeparams = self._build_runtimeparams(inputs)
        optimization_response = await self._async_post_action(
            "naive-mpc-optim",
            runtimeparams,
            timeout=60,
        )
        if optimization_response is None:
            _LOGGER.debug("EMHASS naive optimization failed: no response")
            return None

        _LOGGER.debug("EMHASS naive optimization finished successfully")
        return {
            "runtimeparams": runtimeparams,
            "optimization_response": optimization_response,
        }

    async def async_forecast_model_fit(
        self,
        *,
        var_model: str,
        historic_days_to_retrieve: int = DEFAULT_ML_HISTORIC_DAYS,
        model_type: str = DEFAULT_ML_MODEL_TYPE,
        optimization_time_step_minutes: int = DEFAULT_ML_BASE_STEP_MINUTES,
    ) -> dict[str, Any] | None:
        """Train EMHASS ML forecasting model for load history entity."""
        _LOGGER.debug(
            "Starting EMHASS forecast-model-fit var_model=%s historic_days=%s model_type=%s",
            var_model,
            historic_days_to_retrieve,
            model_type,
        )
        if not await self._async_check_url(self._url, "EMHASS base"):
            _LOGGER.debug("EMHASS forecast-model-fit aborted: base URL unreachable")
            return None

        payload: dict[str, Any] = {
            "var_model": var_model,
            "historic_days_to_retrieve": historic_days_to_retrieve,
            "model_type": model_type,
            "sklearn_model": DEFAULT_ML_SKLEARN_MODEL,
            "num_lags": self._ml_num_lags_for_step(optimization_time_step_minutes),
            "split_date_delta": DEFAULT_ML_SPLIT_DATE_DELTA,
            "perform_backtest": False,
            "optimization_time_step": optimization_time_step_minutes,
        }
        response = await self._async_post_action(
            "forecast-model-fit",
            payload,
            timeout=300,
        )
        if response is None:
            _LOGGER.debug("EMHASS forecast-model-fit failed: no response")
            return None

        _LOGGER.debug(
            "EMHASS forecast-model-fit finished: response_keys=%s",
            sorted(response.keys()),
        )
        return response

    async def async_forecast_model_tune(
        self,
        *,
        var_model: str,
        model_type: str = DEFAULT_ML_MODEL_TYPE,
        n_trials: int | None = None,
    ) -> dict[str, Any] | None:
        """Tune an EMHASS ML forecasting model before fit."""
        _LOGGER.debug(
            "Starting EMHASS forecast-model-tune var_model=%s model_type=%s n_trials=%s",
            var_model,
            model_type,
            n_trials,
        )
        if not await self._async_check_url(self._url, "EMHASS base"):
            _LOGGER.debug("EMHASS forecast-model-tune aborted: base URL unreachable")
            return None

        payload: dict[str, Any] = {
            "var_model": var_model,
            "model_type": model_type,
        }
        if n_trials is not None and n_trials > 0:
            payload["n_trials"] = n_trials

        response = await self._async_post_action(
            "forecast-model-tune",
            payload,
            timeout=300,
        )
        if response is None:
            _LOGGER.debug("EMHASS forecast-model-tune failed: no response")
            return None

        _LOGGER.debug(
            "EMHASS forecast-model-tune finished: response_keys=%s",
            sorted(response.keys()),
        )
        return response

    async def async_forecast_model_predict(
        self,
        *,
        var_model: str,
        prediction_horizon: int,
        optimization_time_step_minutes: int,
        model_type: str = DEFAULT_ML_MODEL_TYPE,
        publish_entity_id: str = DEFAULT_ML_PREDICT_PUBLISH_ENTITY_ID,
        publish_unit_of_measurement: str = DEFAULT_ML_PREDICT_PUBLISH_UNIT,
        publish_friendly_name: str = DEFAULT_ML_PREDICT_PUBLISH_NAME,
    ) -> list[float] | None:
        """Predict load power forecast with EMHASS ML forecaster."""
        _LOGGER.debug(
            "Starting EMHASS forecast-model-predict var_model=%s horizon=%s step=%s model_type=%s",
            var_model,
            prediction_horizon,
            optimization_time_step_minutes,
            model_type,
        )
        if not await self._async_check_url(self._url, "EMHASS base"):
            _LOGGER.debug("EMHASS forecast-model-predict aborted: base URL unreachable")
            return None

        payload: dict[str, Any] = {
            "var_model": var_model,
            "model_type": model_type,
            "prediction_horizon": prediction_horizon,
            "optimization_time_step": optimization_time_step_minutes,
            "model_predict_publish": True,
            "model_predict_entity_id": publish_entity_id,
            "model_predict_unit_of_measurement": publish_unit_of_measurement,
            "model_predict_friendly_name": publish_friendly_name,
        }
        response = await self._async_post_action(
            "forecast-model-predict",
            payload,
            timeout=60,
        )
        if response is None:
            _LOGGER.debug("EMHASS forecast-model-predict failed: no response")
            return None

        values = await self._async_read_ml_predict_sensor(
            entity_id=publish_entity_id,
            expected_points=prediction_horizon,
        )
        if not values:
            _LOGGER.debug(
                "EMHASS forecast-model-predict did not publish usable sensor values: response_keys=%s entity_id=%s",
                sorted(response.keys()),
                publish_entity_id,
            )
            return None

        _LOGGER.debug(
            "EMHASS forecast-model-predict finished with %s values",
            len(values),
        )
        return values

    async def async_publish_data(
        self, optimization_time_step_minutes: int
    ) -> dict[str, Any] | None:
        """Run only the EMHASS publish step and read back published entities."""
        _LOGGER.debug("Starting EMHASS publish-data")
        if not await self._async_check_url(self._url, "EMHASS base"):
            _LOGGER.debug("EMHASS publish-data aborted: base URL unreachable")
            return None

        publish_response = await self._async_post_action(
            "publish-data",
            self._build_publish_payload(optimization_time_step_minutes),
            timeout=30,
        )
        if publish_response is None:
            _LOGGER.debug("EMHASS publish-data failed: no response")
            return None

        await self._hass.async_block_till_done()
        published_entities = self._read_published_entities()
        execution_plan = self._build_execution_plan(
            published_entities,
            optimization_time_step_minutes=optimization_time_step_minutes,
        )
        self._log_published_entities(published_entities)
        _LOGGER.debug("EMHASS publish-data finished successfully")

        return {
            "publish_response": publish_response,
            "published_entities": published_entities,
            "execution_plan": execution_plan,
        }

    async def async_run_naive_mpc(
        self, inputs: OptimizationInputs
    ) -> EmhassExecutionResult | None:
        """Run one EMHASS naive MPC cycle and publish resulting entities.

        Execution sequence:
        1. Reachability check.
        2. POST ``naive-mpc-optim`` with runtimeparams.
        3. POST ``publish-data`` with minimal runtime parameters.
        4. Wait for HA state machine to process updates.
        5. Read and return published entity snapshots.

        Returns ``None`` if any mandatory step fails.
        """
        _LOGGER.debug("Starting full EMHASS naive MPC cycle")
        optimization_result = await self.async_run_naive_optimization(inputs)
        if optimization_result is None:
            _LOGGER.debug("Naive MPC cycle failed during optimization step")
            return None

        publish_result = await self.async_publish_data(
            inputs.optimization_time_step_minutes
        )
        if publish_result is None:
            _LOGGER.debug("Naive MPC cycle failed during publish step")
            return None

        _LOGGER.debug("Full EMHASS naive MPC cycle finished successfully")
        return EmhassExecutionResult(
            runtimeparams=optimization_result["runtimeparams"],
            optimization_response=optimization_result["optimization_response"],
            publish_response=publish_result["publish_response"],
            published_entities=publish_result["published_entities"],
            execution_plan=publish_result["execution_plan"],
        )
