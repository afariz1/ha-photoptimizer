# Photoptimizer for Home Assistant (HACS)

Photoptimizer is a custom Home Assistant integration that optimizes PV and battery operation using EMHASS (Model Predictive Control) and forecast data.

## What The Integration Does

This integration controls the inverter (GoodWe or Growatt) based on an optimization plan. The plan is published by EMHASS, which uses load, production and spot price forecasts.

It also exposes multiple sensors for forecasts, optimization status, and battery commands.

## Operating Model

Photoptimizer runs a quarter-hour MPC loop. On each cycle it collects the latest inputs, asks EMHASS for a new plan, publishes the result to Home Assistant, and applies the current slot to the inverter.

There is also a separate daily ML refresh task that keeps the load forecast model aligned with recent history.

## Requirements

- Home Assistant with HACS
- running EMHASS
- available Home Assistant entities for:
  - electricity price (Czech Energy Spot Prices recommended for Czech Republic)
  - house consumption entity with history data
  - current solar production
  - battery SoC
  - inverter control entities
  - optional deferrable loads

## EMHASS Dependency

This integration expects EMHASS to be installed, configured, and reachable over HTTP.

During configuration, you will provide:

- URL EMHASS (`emhass_url`)
- optional token (`emhass_token`)

Without a working EMHASS instance, MPC optimization will not run.

## Known Limitations

- The integration is intentionally focused on one Photoptimizer instance per Home Assistant installation.
- Sensor and switch labels are tuned for UI clarity, but control still depends on EMHASS and the configured inverter entities.
- The daily ML refresh and the quarter-hour MPC loop are independent tasks and can fail separately.

## Quick Configuration Steps

1. Select the electricity price entity.
2. Configure Forecast.Solar parameters (latitude, longitude, azimuth, declination, kWp, optional API key).
3. Choose inverter type (GoodWe or Growatt) and map the corresponding entities.
4. Configure battery settings (capacity, SoC reserve/target, charge/discharge limits, efficiency).
5. Configure EMHASS (URL/token) and deferrable loads.
