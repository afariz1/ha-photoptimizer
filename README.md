# Photoptimizer for Home Assistant

**Photoptimizer** is a custom Home Assistant integration that plans PV and battery operation using **EMHASS** (model predictive control) together with your forecasts for production, consumption, and electricity prices.

Repository: [https://github.com/afariz1/ha-photoptimizer](https://github.com/afariz1/ha-photoptimizer)

---

## What the Integration Does

- Runs a **quarter-hour optimization loop** that refreshes the plan, publishes sensors, and applies the active time slot to your inverter ( currently supporting **GoodWe** or **Growatt**).
- **Forecast.Solar–based PV forecast** (built into the integration configuration; no separate Forecast.Solar HA integration required for this step).
- **EMHASS** as the optimization engine (you run EMHASS separately; Photoptimizer calls it over HTTP).
- Optional **deferrable loads** (up to two switch-based devices EMHASS can schedule).
- A separate **daily ML refresh** for the load forecast (can fail independently from the MPC loop. Check logs if something looks stale).

---

## Before you install

| Requirement | Notes |
|-------------|--------|
| **Home Assistant** | 2024.6.0 or newer (see `hacs.json`). |
| **HACS** | Used to install this custom integration. |
| **Recorder** | Enabled (integration depends on it for load history used in forecasting). |
| **EMHASS** | Installed, configured, and reachable from Home Assistant (same machine or URL). Default URL in the form is `http://localhost:5000`. |
| **Entities** | Sensors for grid price (or use fixed prices), PV power, house consumption, battery SOC, and inverter control (unless you use *command-only* mode). |

Photoptimizer allows **one configuration entry** per Home Assistant instance.

---

## Installation (HACS)

1. In HACS, add this repository as a **custom repository** (category: *Integration*) if it is not listed by default.  
2. Install **Photoptimizer** and **restart Home Assistant**.  
3. Go to **Settings → Devices & services → Add integration** and search for **Photoptimizer**.

---

## Czech spot prices (recommended for CZ users)

Photoptimizer needs a **sensor** whose **attributes** contain a **future** electricity price profile: keys must be parseable as **ISO datetimes**, values must be prices (numbers or simple structures the integration can read. See code comments referencing Czech spot integrations).

A common setup in the Czech Republic:

1. Install **[Czech Energy Spot Prices](https://github.com/rnovacek/homeassistant_cz_energy_spot_prices)** (usually searchable in HACS as *Czech Energy Spot Prices* / *CZ Energy Spot Prices*).
2. Add the integration under **Settings → Devices & services** and complete its wizard (distribution, VAT/fees, currency, etc.—follow that integration’s docs).
3. Wait until its sensors have updated; in **Developer tools → States**, open the sensor you plan to use and confirm **attributes** contain **upcoming** timestamps and prices (not only the current state).
4. In Photoptimizer’s **first configuration step**, pick that **electricity price entity** (`sensor.*`).

**Alternative:** if you do not use a dynamic price sensor, you can leave the price entity empty and set a **fixed buy price (per kWh)** in the same step (and optionally a **fixed sell price**). You must provide either a price entity or a fixed buy price. Otherwise the flow will show an error.

---

## Configuration flow (step by step)

When you add the integration, the UI guides you through these steps in order:

### 1. Electricity price

- Choose a **`sensor`** for spot/dynamic prices **or** enter **fixed buy** (and optionally **fixed sell**) price per kWh.
- The sensor should expose a usable **forecast in attributes** for best results; fixed prices are simpler but ignore the market.

### 2. PV forecast (Forecast.Solar parameters)

- **Latitude / longitude** default from your Home Assistant zone—adjust if the array is elsewhere.
- **Azimuth** and **declination (tilt)** describe roof/panel orientation.
- **Peak power (kWp)** is your installed DC peak power.
- **API key** is optional; without it you use the public Forecast.Solar quota (fine for many homes; add a key if you hit limits).

### 3. Inverter type

- Select **GoodWe** or **Growatt** (controls and optional fields differ slightly).

### 4. Inverter, battery, consumption, EMHASS

One large form:

- **Current solar production** (`sensor`) — PV power now (used to align near-term production).
- **Household load** (`sensor`) — total consumption; used for **current load** and **historical ML forecast** in EMHASS.
- **Battery state of charge** (`sensor`).
- **Battery capacity (kWh)** and limits: **reserve SOC**, **target SOC** (end of horizon), **max charge/discharge power (W)**, **round-trip efficiency (%)**, optional **wear cost per kWh**.
- **EMHASS URL** and optional **token** — validated when you submit (connection/auth errors appear on the form).
- **Inverter command only** — if enabled, Photoptimizer **does not** call inverter entities; it only logs what it would do (safe for testing).
- Otherwise map **mode** (`select`) and **charge/discharge power** (`number`) entities from your inverter integration. **Growatt** users may also set AC charge switch, device, and variant (**Auto / MIN / SPH**).

### 5. Deferrable loads

- Choose **0, 1, or 2** loads.
- For each: **name**, **switch** entity, **nominal power (W)**, **minimum operating time (minutes)**.

After the last step, the integration is created.

### Changing settings later

- **Deferrable loads and price source:** **Settings → Devices & services → Photoptimizer → Configure** (options flow).
- **EMHASS URL/token only:** use the integration’s **Reconfigure** entry if shown, or remove/re-add if you change many other fields (there is a dedicated reconfigure flow in code for EMHASS).

---

## EMHASS

Photoptimizer expects EMHASS to be **running and compatible** with the actions it triggers (quarter-hour MPC and publish). Configure EMHASS according to [EMHASS documentation](https://emhass.readthedocs.io/) for your environment (Docker add-on, standalone, etc.).

Without working EMHASS, optimization will not run.

---

## Operating model (short)

1. **Quarter-hour cycle:** read inputs → call EMHASS → publish sensors → apply the current slot to the inverter.  
2. **Daily task:** refresh ML load forecast from history.  

Check **logs** (`photoptimizer` / `homeassistant.components.photoptimizer`) if plans stop updating.

---

## Known limitations

- Quality of optimization depends on **EMHASS tuning**, **entity quality**, and **price forecast horizon**.
- Limited inverter types options.

## Links

- **Issues:** [https://github.com/afariz1/ha-photoptimizer/issues](https://github.com/afariz1/ha-photoptimizer/issues)  
- **Czech spot prices (example):** [rnovacek/homeassistant_cz_energy_spot_prices](https://github.com/rnovacek/homeassistant_cz_energy_spot_prices)  
- **EMHASS:** [https://emhass.readthedocs.io/](https://emhass.readthedocs.io/)
