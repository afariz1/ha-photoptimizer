"""Executor that applies EMHASS output to configured inverter controls."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .inverter_factory import create_inverter_adapter
from .models import (
    DeferrableLoadDefinition,
    ExecutionPlan,
    ExecutionSlotCommand,
    OperationMode,
    PublishedEntityState,
)

_LOGGER = logging.getLogger(__name__)
_MAX_ACCEPTABLE_SLOT_AGE = timedelta(minutes=30)
_LOAD_POWER_THRESHOLD_W = 50.0
_FALLBACK_SIGNATURE_SLOT_START = datetime(1970, 7, 24, tzinfo=dt_util.UTC)


class PhotoptimizerExecutor:
    """Execute current-slot battery command from normalized execution plan."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize executor for one config entry."""
        self._hass = hass
        self._last_signature: tuple[datetime, int, str] | None = None
        self._last_deferrable_signatures: dict[str, bool] = {}
        self._controller = create_inverter_adapter(hass, entry)

    async def async_execute_plan(self, execution_plan: ExecutionPlan | None) -> bool:
        """Apply current command from normalized EMHASS execution plan.

        Returns True when a command was effectively sent.
        """
        if execution_plan is None:
            return False

        now_utc = dt_util.utcnow()
        command = next(
            (
                slot
                for slot in execution_plan.slots
                if timedelta(0) <= now_utc - slot.slot_start <= _MAX_ACCEPTABLE_SLOT_AGE
            ),
            None,
        )
        if (
            command is None
            and execution_plan.slots
            and execution_plan.slots[0].slot_start > now_utc
            and execution_plan.slots[0].slot_start - now_utc
            <= timedelta(minutes=execution_plan.step_minutes)
        ):
            command = execution_plan.slots[0]

        if command is None and execution_plan.slots:
            _LOGGER.warning(
                "Executor skipped stale plan slots: plan_ts=%s now=%s source=%s",
                execution_plan.timestamp.isoformat(),
                now_utc.isoformat(),
                execution_plan.source,
            )

        if command is None:
            if not execution_plan.valid:
                command = ExecutionSlotCommand(
                    slot_start=now_utc,
                    p_bat_cmd=0,
                    soc_target=0,
                    grid_limit=0,
                    op_mode=OperationMode.AUTO,
                )
            else:
                return False

        if not execution_plan.valid and command.op_mode == OperationMode.AUTO:
            signature = (
                _FALLBACK_SIGNATURE_SLOT_START,
                command.p_bat_cmd,
                command.op_mode.value,
            )
        else:
            signature = (command.slot_start, command.p_bat_cmd, command.op_mode.value)
        if signature == self._last_signature:
            return False

        await self._controller.async_apply(command)
        self._last_signature = signature
        return True

    async def async_execute_deferrable_loads(
        self,
        published_entities: dict[str, PublishedEntityState],
        deferrable_loads: list[DeferrableLoadDefinition],
    ) -> bool:
        """Apply the current EMHASS deferrable-load state to switch entities."""
        applied = False

        for index, load in enumerate(deferrable_loads):
            entity_key = f"deferrable_load_{index}"
            published_entity = published_entities.get(entity_key)
            if published_entity is None:
                continue

            state = published_entity.state
            if state is None:
                continue

            try:
                desired_on = float(state) > _LOAD_POWER_THRESHOLD_W
            except TypeError, ValueError:
                continue

            if self._last_deferrable_signatures.get(load.entity_id) == desired_on:
                continue

            service_name = "turn_on" if desired_on else "turn_off"
            await self._hass.services.async_call(
                "switch",
                service_name,
                {"entity_id": load.entity_id},
                blocking=True,
            )
            self._last_deferrable_signatures[load.entity_id] = desired_on
            applied = True

            _LOGGER.debug(
                "Applied deferrable load %s -> %s via %s",
                load.name,
                service_name,
                load.entity_id,
            )

        return applied
