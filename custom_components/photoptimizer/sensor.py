"""Platform for sensor integration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import logging
from numbers import Real

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import PERCENTAGE, UnitOfPower
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .coordinator import PhotoptimizerCoordinator

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class PhotoptimizerSensorEntityDescription(SensorEntityDescription):
    """Sensor description extended with a value extractor."""

    value_fn: Callable[[dict], StateType] = lambda _: None


def _emhass_table(index: int, field: str) -> Callable[[dict], StateType]:
    """Read a future value from a published EMHASS entity attribute table."""

    def _coerce_float(value: object) -> float | None:
        if isinstance(value, str) or (
            isinstance(value, Real) and not isinstance(value, bool)
        ):
            try:
                return float(value)
            except ValueError:
                return None

        return None

    def _sorted_schedule(table: object, value_key: str) -> list[tuple[datetime, float]]:
        schedule: list[tuple[datetime, float]] = []
        now = dt_util.utcnow().replace(second=0, microsecond=0)

        if isinstance(table, dict):
            for key, value in table.items():
                dt_value = dt_util.parse_datetime(str(key))
                numeric = _coerce_float(value)
                if dt_value is None or numeric is None:
                    continue

                dt_utc = dt_util.as_utc(dt_value)
                if dt_utc < now:
                    continue

                schedule.append((dt_utc, numeric))

        elif isinstance(table, list):
            for row in table:
                if not isinstance(row, dict):
                    continue

                dt_value = dt_util.parse_datetime(str(row.get("date")))
                numeric = _coerce_float(row.get(value_key))
                if numeric is None:
                    for row_key, row_value in row.items():
                        if row_key == "date":
                            continue
                        numeric = _coerce_float(row_value)
                        if numeric is not None:
                            break

                if dt_value is None or numeric is None:
                    continue

                dt_utc = dt_util.as_utc(dt_value)
                if dt_utc < now:
                    continue

                schedule.append((dt_utc, numeric))

        schedule.sort(key=lambda item: item[0])
        return schedule

    def _fn(data: dict) -> StateType:
        published_entities = (data.get("emhass") or {}).get("published_entities") or {}
        battery_forecast = published_entities.get("battery_forecast") or {}
        attributes = battery_forecast.get("attributes") or {}
        if isinstance(attributes, dict):
            nested = attributes.get(field)
            if isinstance(nested, dict | list):
                table = nested
            else:
                named_table = attributes.get("battery_scheduled_power")
                if isinstance(named_table, dict | list):
                    table = named_table
                else:
                    for maybe_table in attributes.values():
                        if isinstance(maybe_table, dict | list):
                            table = maybe_table
                            break
                    else:
                        table = attributes
        else:
            table = attributes

        schedule = _sorted_schedule(table, field)
        if len(schedule) <= index:
            return None

        _, value = schedule[index]
        return round(value, 2)

    return _fn


def _emhass_current_state(key: str) -> Callable[[dict], StateType]:
    """Read the current state of a published EMHASS entity."""

    def _fn(data: dict) -> StateType:
        published_entities = (data.get("emhass") or {}).get("published_entities") or {}
        entity = published_entities.get(key) or {}
        state = entity.get("state")
        try:
            return round(float(state), 2) if state is not None else None
        except TypeError, ValueError:
            return None

    return _fn


def _emhass_current_state_str(key: str) -> Callable[[dict], StateType]:
    """Read the current state of a published EMHASS entity as-is."""

    def _fn(data: dict) -> StateType:
        published_entities = (data.get("emhass") or {}).get("published_entities") or {}
        entity = published_entities.get(key) or {}
        state = entity.get("state")
        return state if state is not None else None

    return _fn


SENSOR_TYPES: tuple[PhotoptimizerSensorEntityDescription, ...] = (
    PhotoptimizerSensorEntityDescription(
        key="emhass_pv_forecast_now",
        name="Photoptimizer EMHASS PV power forecast (now)",
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfPower.WATT,
        value_fn=_emhass_current_state("pv_forecast"),
    ),
    PhotoptimizerSensorEntityDescription(
        key="emhass_load_forecast_now",
        name="Photoptimizer EMHASS load power forecast (now)",
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfPower.WATT,
        value_fn=_emhass_current_state("load_forecast"),
    ),
    PhotoptimizerSensorEntityDescription(
        key="emhass_grid_forecast_now",
        name="Photoptimizer EMHASS grid power forecast (now)",
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfPower.WATT,
        value_fn=_emhass_current_state("grid_forecast"),
    ),
    PhotoptimizerSensorEntityDescription(
        key="emhass_battery_soc_forecast_now",
        name="Photoptimizer EMHASS battery SOC forecast (now)",
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=PERCENTAGE,
        value_fn=_emhass_current_state("battery_soc_forecast"),
    ),
    PhotoptimizerSensorEntityDescription(
        key="emhass_unit_load_cost",
        name="Photoptimizer EMHASS unit load cost",
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement="currency/kWh",
        value_fn=_emhass_current_state("unit_load_cost"),
    ),
    PhotoptimizerSensorEntityDescription(
        key="emhass_unit_prod_price",
        name="Photoptimizer EMHASS unit production price",
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement="currency/kWh",
        value_fn=_emhass_current_state("unit_prod_price"),
    ),
    PhotoptimizerSensorEntityDescription(
        key="emhass_total_cost_fun_value",
        name="Photoptimizer EMHASS total cost function value",
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement="currency",
        value_fn=_emhass_current_state("cost_fun"),
    ),
    PhotoptimizerSensorEntityDescription(
        key="emhass_optim_status",
        name="Photoptimizer EMHASS optimization status",
        value_fn=_emhass_current_state_str("optim_status"),
    ),
    PhotoptimizerSensorEntityDescription(
        key="emhass_battery_power_now",
        name="Photoptimizer EMHASS battery power command (now)",
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfPower.WATT,
        value_fn=_emhass_table(0, "p_batt_forecast"),
    ),
    PhotoptimizerSensorEntityDescription(
        key="emhass_battery_power_next_hour",
        name="Photoptimizer EMHASS battery power command (next hour)",
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfPower.WATT,
        value_fn=_emhass_table(1, "p_batt_forecast"),
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Photoptimizer sensor entities."""
    coordinator: PhotoptimizerCoordinator = hass.data[DOMAIN][entry.entry_id]
    _LOGGER.debug(
        "Setting up sensor platform for entry_id=%s with %s sensors",
        entry.entry_id,
        len(SENSOR_TYPES),
    )

    entities = [
        PhotoptimizerSensor(coordinator, entry, description)
        for description in SENSOR_TYPES
    ]
    async_add_entities(entities)


class PhotoptimizerSensor(CoordinatorEntity[PhotoptimizerCoordinator], SensorEntity):
    """Representation of a Photoptimizer sensor."""

    _attr_has_entity_name = True
    entity_description: PhotoptimizerSensorEntityDescription

    def __init__(
        self,
        coordinator: PhotoptimizerCoordinator,
        entry: ConfigEntry,
        description: PhotoptimizerSensorEntityDescription,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self.entity_description = description
        self._attr_unique_id = f"{entry.entry_id}_{description.key}"
        _LOGGER.debug(
            "Created sensor entity key=%s unique_id=%s",
            description.key,
            self._attr_unique_id,
        )

    @property
    def native_value(self) -> StateType:
        """Return the sensor value by calling the description's value_fn."""
        if not self.coordinator.data:
            return None
        return self.entity_description.value_fn(self.coordinator.data)
