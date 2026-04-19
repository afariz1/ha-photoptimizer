"""Config flow for the Photoptimizer integration."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.helpers import selector

from .const import (
    CONF_AZIMUTH,
    CONF_BATTERY_CAPACITY_KWH,
    CONF_BATTERY_CHARGE_POWER_MAX,
    CONF_BATTERY_DISCHARGE_POWER_MAX,
    CONF_BATTERY_EFFICIENCY_ROUND_TRIP,
    CONF_BATTERY_SOC_ENTITY,
    CONF_BATTERY_SOC_RESERVE_PERCENT,
    CONF_BATTERY_TARGET_SOC_PERCENT,
    CONF_CONNECT_INVERTER_ENTITIES,
    CONF_CURRENT_CONSUMPTION_ENTITY,
    CONF_CURRENT_SOLAR_PRODUCTION_ENTITY,
    CONF_DECLINATION,
    CONF_DEFERRABLE_LOAD_ENTITY,
    CONF_DEFERRABLE_LOAD_NAME,
    CONF_DEFERRABLE_LOAD_NOMINAL_POWER,
    CONF_DEFERRABLE_LOAD_OPERATING_MINUTES,
    CONF_DEFERRABLE_LOADS,
    CONF_ELECTRICITY_PRICE_ENTITY,
    CONF_EMHASS_TOKEN,
    CONF_EMHASS_URL,
    CONF_GROWATT_AC_CHARGE_SWITCH_ENTITY,
    CONF_GROWATT_DEVICE_ID,
    CONF_GROWATT_INVERTER_VARIANT,
    CONF_INVERTER_CHARGE_POWER_ENTITY,
    CONF_INVERTER_DISCHARGE_POWER_ENTITY,
    CONF_INVERTER_MODE_ENTITY,
    CONF_INVERTER_TYPE,
    CONF_KWP,
    CONF_WEAR_COST_PER_KWH,
    DEFAULT_BATTERY_CHARGE_POWER_MAX,
    DEFAULT_BATTERY_DISCHARGE_POWER_MAX,
    DEFAULT_BATTERY_EFFICIENCY_ROUND_TRIP,
    DEFAULT_BATTERY_SOC_RESERVE_PERCENT,
    DEFAULT_BATTERY_TARGET_SOC_PERCENT,
    DEFAULT_EMHASS_URL,
    DEFAULT_WEAR_COST_PER_KWH,
    DOMAIN,
    GROWATT_VARIANT_AUTO,
    GROWATT_VARIANT_MIN,
    GROWATT_VARIANT_SPH,
    INVERTER_TYPE_GOODWE,
    INVERTER_TYPE_GROWATT,
)

_LOGGER = logging.getLogger(__name__)
_SENSITIVE_FIELDS = {CONF_API_KEY, CONF_EMHASS_TOKEN}


def _deferrable_load_defaults(load_index: int) -> dict[str, Any]:
    """Return suggested defaults for one deferrable load."""
    return {
        CONF_DEFERRABLE_LOAD_NAME: f"Load {load_index + 1}",
        CONF_DEFERRABLE_LOAD_NOMINAL_POWER: 1000.0,
        CONF_DEFERRABLE_LOAD_OPERATING_MINUTES: 60,
    }


def _deferrable_load_schema(defaults: dict[str, Any]) -> vol.Schema:
    """Build schema for one deferrable load."""
    return vol.Schema(
        {
            vol.Required(
                CONF_DEFERRABLE_LOAD_NAME,
                default=defaults[CONF_DEFERRABLE_LOAD_NAME],
            ): str,
            vol.Required(CONF_DEFERRABLE_LOAD_ENTITY): selector.EntitySelector(
                selector.EntitySelectorConfig(domain=["switch"], multiple=False)
            ),
            vol.Required(
                CONF_DEFERRABLE_LOAD_NOMINAL_POWER,
                default=defaults[CONF_DEFERRABLE_LOAD_NOMINAL_POWER],
            ): vol.All(vol.Coerce(float), vol.Range(min=1)),
            vol.Required(
                CONF_DEFERRABLE_LOAD_OPERATING_MINUTES,
                default=defaults[CONF_DEFERRABLE_LOAD_OPERATING_MINUTES],
            ): vol.All(vol.Coerce(int), vol.Range(min=1)),
        }
    )


def _redact_user_input(user_input: dict[str, Any]) -> dict[str, Any]:
    """Return log-safe copy of flow user input."""
    return {
        key: ("***" if key in _SENSITIVE_FIELDS and value else value)
        for key, value in user_input.items()
    }


class PhotoptimizerConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Photoptimizer config flow."""

    VERSION = 1
    MINOR_VERSION = 1

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> PhotoptimizerOptionsFlow:
        """Return the options flow handler."""
        return PhotoptimizerOptionsFlow(config_entry)

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._data: dict[str, Any] = {}
        self._deferrable_load_count = 0
        self._deferrable_load_index = 0
        self._deferrable_loads: list[dict[str, Any]] = []

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Initial step, continue directly to required user inputs."""
        _LOGGER.debug("Config flow user step started")
        return await self.async_step_electricity_price()

    async def async_step_electricity_price(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Handle electricity price configuration step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            _LOGGER.debug(
                "Electricity price step received input: %s",
                _redact_user_input(user_input),
            )
            self._data.update(user_input)
            _LOGGER.debug("Electricity price step completed")

            return await self.async_step_pv_forecast()

        _LOGGER.debug("Showing electricity price form")

        data_schema = vol.Schema(
            {
                vol.Required(CONF_ELECTRICITY_PRICE_ENTITY): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        domain=["sensor"],
                        multiple=False,
                    )
                ),
            }
        )

        return self.async_show_form(
            step_id="electricity_price",
            data_schema=data_schema,
            errors=errors,
        )

    async def async_step_pv_forecast(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """PV forecast step using Forecast.Solar settings."""
        errors: dict[str, str] = {}

        if user_input is not None:
            _LOGGER.debug(
                "PV forecast step received input: %s",
                _redact_user_input(user_input),
            )
            self._data.update(user_input)
            _LOGGER.debug("PV forecast step validating unique ID")

            unique_id = f"{user_input[CONF_LATITUDE]}_{user_input[CONF_LONGITUDE]}_{user_input[CONF_KWP]}"
            await self.async_set_unique_id(unique_id)
            self._abort_if_unique_id_configured()
            _LOGGER.debug("PV forecast step completed with unique_id=%s", unique_id)

            return await self.async_step_inverter_type()

        _LOGGER.debug("Showing PV forecast form")

        default_latitude = (
            self.hass.config.latitude
            if self.hass.config.latitude is not None
            else 49.5962536
        )
        default_longitude = (
            self.hass.config.longitude
            if self.hass.config.longitude is not None
            else 18.3395664
        )

        data_schema = vol.Schema(
            {
                vol.Required(CONF_LATITUDE, default=default_latitude): vol.Coerce(
                    float
                ),
                vol.Required(CONF_AZIMUTH, default=124): vol.Coerce(int),
                vol.Required(CONF_LONGITUDE, default=default_longitude): vol.Coerce(
                    float
                ),
                vol.Required(CONF_KWP, default=6.44): vol.Coerce(float),
                vol.Required(CONF_DECLINATION, default=40): vol.Coerce(int),
                vol.Optional(CONF_API_KEY): str,
            }
        )

        return self.async_show_form(
            step_id="pv_forecast",
            data_schema=data_schema,
            errors=errors,
        )

    async def async_step_inverter_type(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Handle inverter type selection."""
        errors: dict[str, str] = {}

        if user_input is not None:
            _LOGGER.debug(
                "Inverter type step received input: %s",
                _redact_user_input(user_input),
            )
            self._data.update(user_input)
            return await self.async_step_inverter_connection()

        data_schema = vol.Schema(
            {
                vol.Required(CONF_INVERTER_TYPE): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            selector.SelectOptionDict(
                                value=INVERTER_TYPE_GOODWE,
                                label="GoodWe",
                            ),
                            selector.SelectOptionDict(
                                value=INVERTER_TYPE_GROWATT,
                                label="Growatt",
                            ),
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
            }
        )

        return self.async_show_form(
            step_id="inverter_type",
            data_schema=data_schema,
            errors=errors,
        )

    async def async_step_inverter_connection(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Choose whether to connect inverter entities or only print commands."""
        errors: dict[str, str] = {}

        if user_input is not None:
            _LOGGER.debug(
                "Inverter connection step received input: %s",
                _redact_user_input(user_input),
            )
            self._data[CONF_CONNECT_INVERTER_ENTITIES] = user_input[
                CONF_CONNECT_INVERTER_ENTITIES
            ]
            if not user_input[CONF_CONNECT_INVERTER_ENTITIES]:
                _LOGGER.debug(
                    "Command-only mode selected; inverter control entities will be skipped"
                )
            return await self.async_step_inverter()

        data_schema = vol.Schema(
            {
                vol.Required(CONF_CONNECT_INVERTER_ENTITIES, default=True): bool,
            }
        )

        return self.async_show_form(
            step_id="inverter_connection",
            data_schema=data_schema,
            errors=errors,
        )

    async def async_step_inverter(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Handle inverter configuration step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            _LOGGER.debug(
                "Inverter step received input: %s",
                _redact_user_input(user_input),
            )
            self._data.update(user_input)
            _LOGGER.debug("Inverter data step completed")

            return await self.async_step_deferrable_load_count()

        _LOGGER.debug("Showing inverter form")

        data_fields: dict[Any, Any] = {
            vol.Required(CONF_CURRENT_SOLAR_PRODUCTION_ENTITY): selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain=["sensor"],
                    multiple=False,
                )
            ),
            vol.Required(CONF_CURRENT_CONSUMPTION_ENTITY): selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain=["sensor"],
                    multiple=False,
                )
            ),
            vol.Required(CONF_BATTERY_SOC_ENTITY): selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain=["sensor"],
                    multiple=False,
                )
            ),
            vol.Required(CONF_BATTERY_CAPACITY_KWH): vol.Coerce(float),
            vol.Required(
                CONF_BATTERY_SOC_RESERVE_PERCENT,
                default=DEFAULT_BATTERY_SOC_RESERVE_PERCENT,
            ): vol.All(vol.Coerce(float), vol.Range(min=0, max=100)),
            vol.Required(
                CONF_BATTERY_EFFICIENCY_ROUND_TRIP,
                default=DEFAULT_BATTERY_EFFICIENCY_ROUND_TRIP,
            ): vol.All(vol.Coerce(float), vol.Range(min=1, max=100)),
            vol.Required(
                CONF_BATTERY_TARGET_SOC_PERCENT,
                default=DEFAULT_BATTERY_TARGET_SOC_PERCENT,
            ): vol.All(vol.Coerce(float), vol.Range(min=0, max=100)),
            vol.Required(
                CONF_BATTERY_CHARGE_POWER_MAX,
                default=DEFAULT_BATTERY_CHARGE_POWER_MAX,
            ): vol.All(vol.Coerce(float), vol.Range(min=1)),
            vol.Required(
                CONF_BATTERY_DISCHARGE_POWER_MAX,
                default=DEFAULT_BATTERY_DISCHARGE_POWER_MAX,
            ): vol.All(vol.Coerce(float), vol.Range(min=1)),
            vol.Required(
                CONF_WEAR_COST_PER_KWH,
                default=DEFAULT_WEAR_COST_PER_KWH,
            ): vol.Coerce(float),
            vol.Required(CONF_EMHASS_URL, default=DEFAULT_EMHASS_URL): str,
            vol.Optional(CONF_EMHASS_TOKEN): selector.TextSelector(
                selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
            ),
        }

        inverter_type = self._data.get(CONF_INVERTER_TYPE)
        connect_inverter_entities = self._data.get(CONF_CONNECT_INVERTER_ENTITIES, True)

        if connect_inverter_entities:
            data_fields.update(
                {
                    vol.Required(CONF_INVERTER_MODE_ENTITY): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain=["select"],
                            multiple=False,
                        )
                    ),
                    vol.Optional(
                        CONF_INVERTER_CHARGE_POWER_ENTITY
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain=["number"],
                            multiple=False,
                        )
                    ),
                    vol.Required(
                        CONF_INVERTER_DISCHARGE_POWER_ENTITY
                    ): selector.EntitySelector(
                        selector.EntitySelectorConfig(
                            domain=["number"],
                            multiple=False,
                        )
                    ),
                }
            )

            if inverter_type == INVERTER_TYPE_GROWATT:
                data_fields.update(
                    {
                        vol.Optional(
                            CONF_GROWATT_AC_CHARGE_SWITCH_ENTITY
                        ): selector.EntitySelector(
                            selector.EntitySelectorConfig(
                                domain=["switch"],
                                multiple=False,
                            )
                        ),
                        vol.Optional(CONF_GROWATT_DEVICE_ID): selector.DeviceSelector(
                            selector.DeviceSelectorConfig(integration="growatt_server")
                        ),
                        vol.Optional(
                            CONF_GROWATT_INVERTER_VARIANT,
                            default=GROWATT_VARIANT_AUTO,
                        ): selector.SelectSelector(
                            selector.SelectSelectorConfig(
                                options=[
                                    selector.SelectOptionDict(
                                        value=GROWATT_VARIANT_AUTO,
                                        label="Auto",
                                    ),
                                    selector.SelectOptionDict(
                                        value=GROWATT_VARIANT_MIN,
                                        label="MIN",
                                    ),
                                    selector.SelectOptionDict(
                                        value=GROWATT_VARIANT_SPH,
                                        label="SPH",
                                    ),
                                ],
                                mode=selector.SelectSelectorMode.DROPDOWN,
                            )
                        ),
                    }
                )
        else:
            _LOGGER.debug(
                "Command-only mode active; not requesting inverter control entities"
            )

        data_schema = vol.Schema(data_fields)

        return self.async_show_form(
            step_id="inverter",
            data_schema=data_schema,
            errors=errors,
        )

    async def async_step_deferrable_load_count(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Select how many deferrable loads should be configured."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._deferrable_load_count = int(user_input["deferrable_load_count"])
            self._deferrable_load_index = 0
            self._deferrable_loads = []

            if self._deferrable_load_count == 0:
                self._data[CONF_DEFERRABLE_LOADS] = []
                _LOGGER.info("Creating Photoptimizer config entry")
                _LOGGER.debug(
                    "Config entry data keys: %s",
                    sorted(self._data.keys()),
                )
                return self.async_create_entry(title="Photoptimizer", data=self._data)

            return await self.async_step_deferrable_load()

        data_schema = vol.Schema(
            {
                vol.Required("deferrable_load_count", default=0): vol.All(
                    vol.Coerce(int), vol.In([0, 1, 2])
                ),
            }
        )

        return self.async_show_form(
            step_id="deferrable_load_count",
            data_schema=data_schema,
            errors=errors,
        )

    async def async_step_deferrable_load(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Configure one deferrable load definition."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._deferrable_loads.append(user_input)
            self._deferrable_load_index += 1

            if self._deferrable_load_index >= self._deferrable_load_count:
                self._data[CONF_DEFERRABLE_LOADS] = self._deferrable_loads
                _LOGGER.info("Creating Photoptimizer config entry")
                _LOGGER.debug(
                    "Config entry data keys: %s",
                    sorted(self._data.keys()),
                )
                return self.async_create_entry(title="Photoptimizer", data=self._data)

            return await self.async_step_deferrable_load()

        defaults = _deferrable_load_defaults(self._deferrable_load_index)
        data_schema = self.add_suggested_values_to_schema(
            _deferrable_load_schema(defaults),
            defaults,
        )

        return self.async_show_form(
            step_id="deferrable_load",
            data_schema=data_schema,
            errors=errors,
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Handle reconfiguration of EMHASS connection settings."""
        entry = self._get_reconfigure_entry()
        _LOGGER.debug("Reconfigure step opened for entry_id=%s", entry.entry_id)

        if user_input is not None:
            _LOGGER.debug(
                "Reconfigure step received input: %s",
                _redact_user_input(user_input),
            )
            return self.async_update_reload_and_abort(
                entry,
                data_updates={
                    CONF_EMHASS_URL: user_input[CONF_EMHASS_URL],
                    CONF_EMHASS_TOKEN: user_input.get(CONF_EMHASS_TOKEN) or None,
                },
            )

        _LOGGER.debug("Showing reconfigure form for entry_id=%s", entry.entry_id)
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(
                    {
                        vol.Required(CONF_EMHASS_URL): str,
                        vol.Optional(CONF_EMHASS_TOKEN): selector.TextSelector(
                            selector.TextSelectorConfig(
                                type=selector.TextSelectorType.PASSWORD
                            )
                        ),
                    }
                ),
                {
                    CONF_EMHASS_URL: entry.data.get(
                        CONF_EMHASS_URL,
                        DEFAULT_EMHASS_URL,
                    ),
                    CONF_EMHASS_TOKEN: entry.data.get(CONF_EMHASS_TOKEN),
                },
            ),
            errors={},
        )


class PhotoptimizerOptionsFlow(config_entries.OptionsFlowWithReload):
    """Options flow for editing deferrable loads."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize the options flow."""
        self._config_entry = config_entry
        self._existing_deferrable_loads: list[dict[str, Any]] = list(
            config_entry.options.get(
                CONF_DEFERRABLE_LOADS,
                config_entry.data.get(CONF_DEFERRABLE_LOADS, []),
            )
        )
        self._deferrable_loads: list[dict[str, Any]] = []
        self._deferrable_load_count = len(self._existing_deferrable_loads)
        self._deferrable_load_index = 0

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Start the options flow."""
        if user_input is not None:
            self._deferrable_load_count = int(user_input["deferrable_load_count"])
            self._deferrable_load_index = 0
            self._deferrable_loads = []

            if self._deferrable_load_count == 0:
                return self.async_create_entry(
                    title="",
                    data={CONF_DEFERRABLE_LOADS: []},
                )

            return await self.async_step_deferrable_load()

        data_schema = vol.Schema(
            {
                vol.Required(
                    "deferrable_load_count", default=min(self._deferrable_load_count, 2)
                ): vol.All(vol.Coerce(int), vol.In([0, 1, 2])),
            }
        )

        return self.async_show_form(step_id="init", data_schema=data_schema)

    async def async_step_deferrable_load(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Edit one deferrable load definition."""
        if user_input is not None:
            self._deferrable_loads.append(user_input)
            self._deferrable_load_index += 1

            if self._deferrable_load_index >= self._deferrable_load_count:
                return self.async_create_entry(
                    title="",
                    data={CONF_DEFERRABLE_LOADS: self._deferrable_loads},
                )

            return await self.async_step_deferrable_load()

        if self._deferrable_load_index < len(self._existing_deferrable_loads):
            existing_load = self._existing_deferrable_loads[self._deferrable_load_index]
        else:
            existing_load = {}
        defaults = {
            CONF_DEFERRABLE_LOAD_NAME: existing_load.get(
                CONF_DEFERRABLE_LOAD_NAME,
                f"Load {self._deferrable_load_index + 1}",
            ),
            CONF_DEFERRABLE_LOAD_NOMINAL_POWER: existing_load.get(
                CONF_DEFERRABLE_LOAD_NOMINAL_POWER,
                1000.0,
            ),
            CONF_DEFERRABLE_LOAD_OPERATING_MINUTES: existing_load.get(
                CONF_DEFERRABLE_LOAD_OPERATING_MINUTES,
                60,
            ),
        }
        data_schema = self.add_suggested_values_to_schema(
            _deferrable_load_schema(defaults),
            defaults,
        )

        return self.async_show_form(step_id="deferrable_load", data_schema=data_schema)
