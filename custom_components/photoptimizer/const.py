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

CONF_INVERTER_TYPE = "inverter_type"
CONF_INVERTER_COMMAND_ONLY = "inverter_command_only"
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

# Hourly PV correction (24 factors)
PV_HOURLY_SLOTS_PER_HOUR = 4
PV_HOURLY_W_HIST = 4.0
PV_HOURLY_RATIO_MIN = 0.3
PV_HOURLY_RATIO_MAX = 3.0
PV_HOURLY_FACTOR_MIN = 0.3
PV_HOURLY_FACTOR_MAX = 3.0
PV_HOURLY_MIN_POWER_W = 50.0
PV_HOURLY_STORE_VERSION = 1


def forecast_solar_hour_wh_to_per_bucket_kwh(wh: float, buckets_in_hour: int) -> float:
    """Spread Forecast.Solar hourly energy (Wh) across timeline buckets in that clock hour.

    Full hours use a fixed quarter-hour split (``PV_HOURLY_SLOTS_PER_HOUR``). Partial
    hours at the optimization horizon edges split ``wh`` only across buckets that exist
    in that hour (``kWh_hour / n``), avoiding artificial power inflation.
    """
    if buckets_in_hour <= 0:
        return 0.0
    kwh_hour = float(wh) / 1000.0
    if buckets_in_hour == PV_HOURLY_SLOTS_PER_HOUR:
        return kwh_hour / float(PV_HOURLY_SLOTS_PER_HOUR)
    return kwh_hour / float(buckets_in_hour)


def pv_hourly_ewma_update(
    factor: float, ratio: float, *, w_hist: float = PV_HOURLY_W_HIST
) -> float:
    """Single EWMA factor update (same weighting as the coordinator learning step)."""
    return (w_hist * factor + ratio) / (w_hist + 1.0)
