"""Machine learning load forecast service for Photoptimizer.

This module contains load power prediction logic using EMHASS ML forecaster.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import partial
import logging
from numbers import Real
from typing import TYPE_CHECKING

from homeassistant.components.recorder import get_instance, history
from homeassistant.util import dt as dt_util

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .coordinator import PhotoptimizerCoordinator
    from .models import OptimizationBucket

_LOGGER = logging.getLogger(__name__)

_MIN_HISTORY_ROWS_FOR_ML = 288
_HISTORY_WINDOW_DAYS_FOR_ML = 7
_ML_SLOT_MINUTES = 15


class MLForecastService:
    """Service for EMHASS ML load forecasting with profile fallback."""

    def __init__(
        self, hass: HomeAssistant, coordinator: PhotoptimizerCoordinator
    ) -> None:
        """Initialize ML forecast service."""
        self.hass = hass
        self.coordinator = coordinator
        self._last_ml_fit_utc: datetime | None = None
        _LOGGER.debug("MLForecastService initialized")

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

    async def _count_valid_state_history_slots(
        self,
        entity_id: str,
        start: datetime,
    ) -> int:
        """Return number of 15-minute slots with numeric values in state history."""
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
        except (RuntimeError, OSError, ValueError) as err:
            _LOGGER.debug("History slot query failed for %s: %s", entity_id, err)
            return 0

        history_rows = state_history.get(lower_entity_id, [])
        slots_with_data: set[datetime] = set()
        for hist_state in history_rows:
            value = self._coerce_float(hist_state.state)
            if value is None:
                continue

            changed = getattr(hist_state, "last_updated", hist_state.last_changed)
            local_changed = dt_util.as_local(changed).replace(second=0, microsecond=0)
            slot_minute = (local_changed.minute // _ML_SLOT_MINUTES) * _ML_SLOT_MINUTES
            slot_start = local_changed.replace(minute=slot_minute)
            slots_with_data.add(slot_start)

        _LOGGER.debug(
            "ML state history query for %s: rows=%s valid_slots=%s window_days=%s",
            entity_id,
            len(history_rows),
            len(slots_with_data),
            _HISTORY_WINDOW_DAYS_FOR_ML,
        )
        return len(slots_with_data)

    async def async_has_sufficient_history(self, entity_id: str) -> bool:
        """Check if entity has enough historical data for ML model training."""
        start = dt_util.utcnow() - timedelta(days=_HISTORY_WINDOW_DAYS_FOR_ML)
        valid_rows = await self._count_valid_state_history_slots(entity_id, start)
        is_sufficient = valid_rows >= _MIN_HISTORY_ROWS_FOR_ML
        _LOGGER.debug(
            "ML history sufficiency for %s (state history): valid_rows=%s threshold=%s sufficient=%s",
            entity_id,
            valid_rows,
            _MIN_HISTORY_ROWS_FOR_ML,
            is_sufficient,
        )
        return is_sufficient

    async def async_train_model(
        self,
        entity_id: str,
        *,
        force: bool = False,
    ) -> bool:
        """Train EMHASS ML model."""
        now = dt_util.utcnow()
        if (
            not force
            and self._last_ml_fit_utc is not None
            and self._last_ml_fit_utc.date() == now.date()
        ):
            _LOGGER.debug(
                "Skipping ML fit for %s; already trained today at %s",
                entity_id,
                self._last_ml_fit_utc.isoformat(),
            )
            return True

        fit_response = await self.coordinator.emhass.async_forecast_model_fit(
            var_model=entity_id
        )
        if fit_response is None:
            _LOGGER.debug("ML fit failed for %s", entity_id)
            return False

        self._last_ml_fit_utc = now
        _LOGGER.debug(
            "ML fit finished for %s at %s response_keys=%s",
            entity_id,
            now.isoformat(),
            sorted(fit_response.keys()),
        )
        return True

    async def async_predict_load(
        self,
        entity_id: str,
        prediction_horizon: int,
        optimization_time_step_minutes: int,
    ) -> list[float] | None:
        """Get ML load power forecast."""
        if prediction_horizon == 0:
            return []

        ml_forecast_w = await self.coordinator.emhass.async_forecast_model_predict(
            var_model=entity_id,
            prediction_horizon=prediction_horizon,
            optimization_time_step_minutes=optimization_time_step_minutes,
        )
        if ml_forecast_w is None:
            _LOGGER.debug("ML predict failed for %s", entity_id)
            return None

        _LOGGER.debug(
            "ML predict finished for %s with %s values",
            entity_id,
            len(ml_forecast_w),
        )
        return ml_forecast_w

    async def async_populate_load_from_ml_or_profile(
        self,
        entity_id: str,
        buckets: list[OptimizationBucket],
    ) -> None:
        """Populate load forecast in buckets using ML prediction or profile fallback."""
        if not await self.async_has_sufficient_history(entity_id):
            _LOGGER.debug(
                "Insufficient history for ML forecast (%s), using profile fallback",
                entity_id,
            )
            profile = await self.coordinator.async_build_load_profile(entity_id)
            await self.coordinator.async_hourly_from_load_profile(buckets, profile)
            return

        if not await self.async_train_model(entity_id):
            _LOGGER.debug(
                "ML fit failed for %s, using profile fallback",
                entity_id,
            )
            profile = await self.coordinator.async_build_load_profile(entity_id)
            await self.coordinator.async_hourly_from_load_profile(buckets, profile)
            return

        prediction_horizon = len(buckets)
        if prediction_horizon == 0:
            return

        step_minutes = self.coordinator.bucket_step_minutes(buckets)
        ml_forecast_w = await self.async_predict_load(
            entity_id,
            prediction_horizon,
            step_minutes,
        )
        if ml_forecast_w is None:
            _LOGGER.debug(
                "ML predict failed for %s, using profile fallback",
                entity_id,
            )
            profile = await self.coordinator.async_build_load_profile(entity_id)
            await self.coordinator.async_hourly_from_load_profile(buckets, profile)
            return

        bucket_hours = step_minutes / 60.0
        for index, bucket in enumerate(buckets):
            if index >= len(ml_forecast_w):
                break
            bucket.load = (ml_forecast_w[index] * bucket_hours) / 1000.0

        _LOGGER.debug(
            "Load timeline populated from EMHASS ML forecast for %s: points=%s",
            entity_id,
            len(ml_forecast_w),
        )

    async def async_build_load_profile(self, entity_id: str | None) -> list[float]:
        """Build a 24-hour load profile from history or return default."""
        return await self.coordinator.async_build_load_profile(entity_id)
