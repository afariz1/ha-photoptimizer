# Photoptimizer for Home Assistant (HACS)

Photoptimizer is a custom Home Assistant integration that optimizes PV and battery operation using EMHASS (Model Predictive Control) and forecast data.

## What The Integration Does

This integration controls the inverter (now supports GoodWe or Growatt) based on optimization plan. The plan is published by EMHASS which uses load, production and spot prices forecasts.

It also exposes multiple sensors for forecasts, optimization status, and battery commands.

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

## Quick Configuration Steps

1. Select the electricity price entity.
2. Configure Forecast.Solar parameters (latitude, longitude, azimuth, declination, kWp, optional API key).
3. Choose inverter type (GoodWe or Growatt) and map the corresponding entities.
4. Configure battery settings (capacity, SoC reserve/target, charge/discharge limits, efficiency).
5. Configure EMHASS (URL/token) and deferrable loads.
