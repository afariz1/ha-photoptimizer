"""Data models for Photoptimizer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


@dataclass(slots=True)
class OptimizationBucket:
    """Single aggregated timestep prepared for the optimizer."""

    start: datetime
    price: float
    pv: float
    load: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for coordinator data."""
        return asdict(self)


@dataclass(slots=True)
class DeferrableLoadDefinition:
    """User-defined deferrable load that EMHASS may schedule."""

    name: str
    entity_id: str
    nominal_power_w: float
    operating_minutes: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for coordinator data."""
        return asdict(self)


@dataclass(slots=True)
class OptimizationInputs:
    """Aggregated inputs passed from the coordinator to the EMHASS client."""

    timeline: list[OptimizationBucket]
    battery_soc: float
    deferrable_loads: list[DeferrableLoadDefinition] = field(default_factory=list)
    raw_forecast_solar: Any | None = None

    @property
    def prediction_horizon(self) -> int:
        """Return the number of timesteps sent to EMHASS."""
        return len(self.timeline)

    @property
    def optimization_time_step_minutes(self) -> int:
        """Infer the optimization step from the aggregated timeline."""
        if len(self.timeline) < 2:
            return 30

        return int(
            (self.timeline[1].start - self.timeline[0].start).total_seconds() // 60
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for coordinator data."""
        return {
            "timeline": [bucket.as_dict() for bucket in self.timeline],
            "battery_soc": self.battery_soc,
            "deferrable_loads": [
                deferrable_load.as_dict() for deferrable_load in self.deferrable_loads
            ],
            "raw_forecast_solar": self.raw_forecast_solar,
        }


@dataclass(slots=True)
class PublishedEntityState:
    """Snapshot of an EMHASS-published Home Assistant entity."""

    entity_id: str
    state: str | None
    attributes: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for coordinator data."""
        return {
            "entity_id": self.entity_id,
            "state": self.state,
            "attributes": self.attributes,
        }


@dataclass(slots=True)
class EmhassExecutionResult:
    """Complete result of one EMHASS optimization and publish cycle."""

    runtimeparams: dict[str, Any]
    optimization_response: dict[str, Any]
    publish_response: dict[str, Any]
    published_entities: dict[str, PublishedEntityState]
    execution_plan: ExecutionPlan | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for coordinator data."""
        return {
            "runtimeparams": self.runtimeparams,
            "optimization_response": self.optimization_response,
            "publish_response": self.publish_response,
            "published_entities": {
                key: entity.as_dict() for key, entity in self.published_entities.items()
            },
            "execution_plan": (
                None if self.execution_plan is None else self.execution_plan.as_dict()
            ),
        }


class OperationMode(Enum):
    """Battery operation mode determined by optimizer."""

    FORCED_CHARGE = "forced_charge"  # Battery charging (power < -50W)
    FORCED_DISCHARGE = "forced_discharge"  # Battery discharging (power > 50W)
    IDLE = "idle"  # No operation (power ~0 and price spread too low for cycling)
    AUTO = "auto"  # Safe fallback when EMHASS unavailable or any problem


@dataclass(slots=True)
class ExecutionSlotCommand:
    """Normalized battery command per optimization timestep (typically 15 min)."""

    slot_start: datetime  # Timestamp marking the start of this slot
    p_bat_cmd: int  # Battery power: positive=discharge, negative=charge, in watts
    soc_target: int  # Target state of charge at end of slot (%)
    grid_limit: int  # Grid import/export limit (W); 0=no limit/policy TBD
    op_mode: OperationMode  # Operational mode for this slot

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "slot_start": self.slot_start.isoformat() if self.slot_start else None,
            "p_bat_cmd": self.p_bat_cmd,
            "soc_target": self.soc_target,
            "grid_limit": self.grid_limit,
            "op_mode": self.op_mode.value,
        }


@dataclass(slots=True)
class ExecutionPlan:
    """Complete normalized battery plan from optimizer (collection of time slots)."""

    slots: list[ExecutionSlotCommand]  # One command per optimization timestep
    step_minutes: int  # Granularity of slots (typically 15)
    timestamp: datetime  # When this plan was generated
    valid: bool = True  # Whether plan is valid (EMHASS succeeded, data available, etc.)
    source: str = "emhass_mpc"  # Source identifier (for debugging/diagnostics)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "slots": [slot.as_dict() for slot in self.slots],
            "step_minutes": self.step_minutes,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "valid": self.valid,
            "source": self.source,
        }
