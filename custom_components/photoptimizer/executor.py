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
            _LOGGER.debug("Executor received no execution plan, skipping")
            return False

        now_utc = dt_util.utcnow()
        _LOGGER.debug(
            "Executor evaluating plan: valid=%s source=%s slots=%s plan_ts=%s now=%s",
            execution_plan.valid,
            execution_plan.source,
            len(execution_plan.slots),
            execution_plan.timestamp.isoformat(),
            now_utc.isoformat(),
        )
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
                _LOGGER.debug(
                    "Executor using safe fallback command because execution plan is invalid"
                )
                command = ExecutionSlotCommand(
                    slot_start=now_utc,
                    p_bat_cmd=0,
                    soc_target=0,
                    grid_limit=0,
                    op_mode=OperationMode.AUTO,
                )
            else:
                _LOGGER.debug(
                    "Executor found no current command in a valid plan, skipping apply"
                )
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
            _LOGGER.debug(
                "Executor command unchanged, skipping apply: slot=%s mode=%s power=%s",
                command.slot_start.isoformat(),
                command.op_mode.value,
                command.p_bat_cmd,
            )
            return False

        _LOGGER.info(
            "Executor applying command: slot=%s mode=%s power=%s soc_target=%s grid_limit=%s",
            command.slot_start.isoformat(),
            command.op_mode.value,
            command.p_bat_cmd,
            command.soc_target,
            command.grid_limit,
        )
        await self._controller.async_apply(command)
        self._last_signature = signature
        _LOGGER.info("Executor command applied successfully")
        return True

    async def async_execute_deferrable_loads(
        self,
        published_entities: dict[str, PublishedEntityState],
        deferrable_loads: list[DeferrableLoadDefinition],
    ) -> bool:
        """Apply the current EMHASS deferrable-load state to switch entities."""
        applied = False

        _LOGGER.debug(
            "Executor evaluating deferrable loads: configured=%s published=%s",
            len(deferrable_loads),
            len(published_entities),
        )

        for index, load in enumerate(deferrable_loads):
            entity_key = f"deferrable_load_{index}"
            published_entity = published_entities.get(entity_key)
            if published_entity is None:
                _LOGGER.debug(
                    "Executor deferrable load missing in published entities: index=%s entity_id=%s",
                    index,
                    load.entity_id,
                )
                continue

            state = published_entity.state
            if state is None:
                _LOGGER.debug(
                    "Executor deferrable load has no state yet: index=%s entity_id=%s",
                    index,
                    load.entity_id,
                )
                continue

            try:
                desired_on = float(state) > _LOAD_POWER_THRESHOLD_W
            except (TypeError, ValueError):
                _LOGGER.debug(
                    "Executor deferrable load state not numeric: index=%s entity_id=%s state=%s",
                    index,
                    load.entity_id,
                    state,
                )
                continue

            if self._last_deferrable_signatures.get(load.entity_id) == desired_on:
                _LOGGER.debug(
                    "Executor deferrable load unchanged: entity_id=%s desired_on=%s",
                    load.entity_id,
                    desired_on,
                )
                continue

            service_name = "turn_on" if desired_on else "turn_off"
            _LOGGER.info(
                "Executor applying deferrable load: entity_id=%s service=%s published_state=%s",
                load.entity_id,
                service_name,
                state,
            )
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

        _LOGGER.debug("Executor deferrable load pass completed: applied=%s", applied)
        return applied
