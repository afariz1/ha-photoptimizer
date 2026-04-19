"""Constants for the Photoptimizer integration."""

DOMAIN = "photoptimizer"

CONF_TIMEZONE = "timezone"

CONF_ELECTRICITY_PRICE_ENTITY = "electricity_price_entity"

CONF_LATITUDE = "latitude"
CONF_LONGITUDE = "longitude"
CONF_AZIMUTH = "azimuth"
CONF_KWP = "kwp"
CONF_DECLINATION = "declination"
CONF_API_KEY = "api_key"

CONF_EMHASS_URL = "emhass_url"
CONF_EMHASS_TOKEN = "emhass_token"

CONF_DEFERRABLE_LOADS = "deferrable_loads"
CONF_DEFERRABLE_LOAD_NAME = "name"
CONF_DEFERRABLE_LOAD_ENTITY = "entity_id"
CONF_DEFERRABLE_LOAD_NOMINAL_POWER = "nominal_power_w"
CONF_DEFERRABLE_LOAD_OPERATING_MINUTES = "operating_minutes"

CONF_BATTERY_CAPACITY_KWH = "battery_capacity_kwh"
CONF_BATTERY_SOC_ENTITY = "battery_soc_entity"
CONF_BATTERY_SOC_RESERVE_PERCENT = "battery_soc_reserve_percent"
CONF_BATTERY_TARGET_SOC_PERCENT = "battery_target_soc_percent"
CONF_BATTERY_EFFICIENCY_ROUND_TRIP = "battery_efficiency_round_trip"
CONF_BATTERY_CHARGE_POWER_MAX = "battery_charge_power_max"
CONF_BATTERY_DISCHARGE_POWER_MAX = "battery_discharge_power_max"

CONF_MAX_INVERTER_CURRENT_AMP = "max_inverter_current_amp"
CONF_INVERTER_TYPE = "inverter_type"
CONF_CONNECT_INVERTER_ENTITIES = "connect_inverter_entities"
CONF_INVERTER_MODE_ENTITY = "inverter_mode_entity"
CONF_INVERTER_CHARGE_POWER_ENTITY = "inverter_charge_power_entity"
CONF_INVERTER_DISCHARGE_POWER_ENTITY = "inverter_discharge_power_entity"
CONF_GROWATT_AC_CHARGE_SWITCH_ENTITY = "growatt_ac_charge_switch_entity"
CONF_GROWATT_DEVICE_ID = "growatt_device_id"
CONF_GROWATT_INVERTER_VARIANT = "growatt_inverter_variant"

INVERTER_TYPE_GOODWE = "goodwe"
INVERTER_TYPE_GROWATT = "growatt"

GROWATT_VARIANT_AUTO = "auto"
GROWATT_VARIANT_MIN = "min"
GROWATT_VARIANT_SPH = "sph"

CONF_CURRENT_SOLAR_PRODUCTION_ENTITY = "current_solar_production_entity"
CONF_CURRENT_CONSUMPTION_ENTITY = "current_consumption_entity"


CONF_WEAR_COST_PER_KWH = "wear_cost_per_kwh"

DEFAULT_HORIZON_HOURS = 24
DEFAULT_BATTERY_SOC_RESERVE_PERCENT = 20.0
DEFAULT_BATTERY_TARGET_SOC_PERCENT = 60.0
DEFAULT_WEAR_COST_PER_KWH = 0.0
DEFAULT_BATTERY_EFFICIENCY_ROUND_TRIP = 98.0
DEFAULT_BATTERY_CHARGE_POWER_MAX = 5000.0
DEFAULT_BATTERY_DISCHARGE_POWER_MAX = 5000.0
DEFAULT_EMHASS_URL = "http://localhost:5000"
