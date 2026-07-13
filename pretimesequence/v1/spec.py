from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, Mapping


class AmbiguousBarPolicy(str, Enum):
    """How OHLC bars that touch both TP and SL are treated."""

    CONSERVATIVE = "conservative"
    EXCLUDE = "exclude"


class FoldEndPolicy(str, Enum):
    """How an open position is handled at an evaluation boundary."""

    FORCE_CLOSE = "force_close"


@dataclass(frozen=True)
class CostSpec:
    """Execution costs expressed as fractions of unit notional.

    ``spread_rate`` is the full bid/ask spread. Half of it is applied on each
    market fill. ``funding_rate_per_bar`` is only a deterministic fallback for
    synthetic tests; real experiments should provide a
    ``funding_payment_rate`` column that is zero except on actual payment bars.
    A positive funding payment is paid by longs and received by shorts.
    """

    fee_rate: float = 0.0005
    spread_rate: float = 0.0
    slippage_rate: float = 0.0002
    funding_rate_per_bar: float = 0.0

    def __post_init__(self) -> None:
        for name in ("fee_rate", "spread_rate", "slippage_rate"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}.")
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
            if value >= 1:
                raise ValueError(f"{name} must be less than 1, got {value}.")
        funding_rate = float(self.funding_rate_per_bar)
        if not math.isfinite(funding_rate) or abs(funding_rate) >= 1:
            raise ValueError("funding_rate_per_bar must be between -1 and 1.")

    @property
    def half_spread_rate(self) -> float:
        return self.spread_rate / 2.0

    @property
    def estimated_round_trip_cost(self) -> float:
        return 2 * self.fee_rate + self.spread_rate + 2 * self.slippage_rate


@dataclass(frozen=True)
class StrategySpec:
    """Single v1 research contract shared by targets, splits and backtests.

    The default TP/SL distances retain the unlevered price moves implied by the
    v0 defaults (0.38/20 and 0.28/20). They are transitional, pre-registered
    candidates rather than validated profitable parameters.
    """

    base_interval_minutes: int = 1
    decision_interval_bars: int = 1
    entry_delay_bars: int = 1
    horizon_bars: int = 15
    max_holding_bars: int = 15
    take_profit_return: float = 0.019
    stop_loss_return: float = 0.014
    costs: CostSpec = field(default_factory=CostSpec)
    ambiguous_bar_policy: AmbiguousBarPolicy = AmbiguousBarPolicy.CONSERVATIVE
    fold_end_policy: FoldEndPolicy = FoldEndPolicy.FORCE_CLOSE
    purge_bars: int | None = None
    embargo_bars: int = 1
    max_concurrent_positions: int = 1
    leverage_cap: float = 20.0
    risk_fraction_per_trade: float = 0.0025

    def __post_init__(self) -> None:
        try:
            ambiguous_policy = AmbiguousBarPolicy(self.ambiguous_bar_policy)
        except ValueError as exc:
            raise ValueError(f"Unknown ambiguous_bar_policy: {self.ambiguous_bar_policy}") from exc
        try:
            fold_policy = FoldEndPolicy(self.fold_end_policy)
        except ValueError as exc:
            raise ValueError(f"Unknown fold_end_policy: {self.fold_end_policy}") from exc
        object.__setattr__(self, "ambiguous_bar_policy", ambiguous_policy)
        object.__setattr__(self, "fold_end_policy", fold_policy)

        integer_fields = {
            "base_interval_minutes": self.base_interval_minutes,
            "decision_interval_bars": self.decision_interval_bars,
            "entry_delay_bars": self.entry_delay_bars,
            "horizon_bars": self.horizon_bars,
            "max_holding_bars": self.max_holding_bars,
            "embargo_bars": self.embargo_bars,
            "max_concurrent_positions": self.max_concurrent_positions,
        }
        for name, value in integer_fields.items():
            minimum = 0 if name == "embargo_bars" else 1
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                qualifier = "non-negative" if minimum == 0 else "positive"
                raise ValueError(f"{name} must be a {qualifier} integer, got {value!r}.")

        if self.purge_bars is not None:
            if not isinstance(self.purge_bars, int) or isinstance(self.purge_bars, bool) or self.purge_bars < 0:
                raise ValueError("purge_bars must be a non-negative integer or None.")
            if self.purge_bars < self.horizon_bars:
                raise ValueError("purge_bars must cover at least horizon_bars.")

        available_path_bars = self.horizon_bars - self.entry_delay_bars + 1
        if available_path_bars <= 0:
            raise ValueError("entry_delay_bars must not exceed horizon_bars.")
        if self.max_holding_bars > available_path_bars:
            raise ValueError(
                "max_holding_bars must fit inside the entry-to-horizon path "
                f"({available_path_bars} bars available)."
            )
        for name in ("take_profit_return", "stop_loss_return"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be between 0 and 1, got {value}.")
        if self.max_concurrent_positions != 1:
            raise ValueError("The current v1 engine supports exactly one concurrent position per symbol.")
        leverage_cap = float(self.leverage_cap)
        if not math.isfinite(leverage_cap) or leverage_cap < 1:
            raise ValueError("leverage_cap must be at least 1.")
        risk_fraction = float(self.risk_fraction_per_trade)
        if not math.isfinite(risk_fraction) or not 0 < risk_fraction <= 1:
            raise ValueError("risk_fraction_per_trade must be in (0, 1].")

    @property
    def effective_purge_bars(self) -> int:
        return self.horizon_bars if self.purge_bars is None else self.purge_bars

    @property
    def default_notional_fraction(self) -> float:
        risk_per_unit_notional = self.stop_loss_return + self.costs.estimated_round_trip_cost
        return min(self.risk_fraction_per_trade / risk_per_unit_notional, self.leverage_cap)

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ambiguous_bar_policy"] = self.ambiguous_bar_policy.value
        payload["fold_end_policy"] = self.fold_end_policy.value
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StrategySpec":
        allowed = {item.name for item in fields(cls)}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown StrategySpec fields: {sorted(unknown)}")

        values = dict(payload)
        raw_costs = values.get("costs")
        if raw_costs is not None and not isinstance(raw_costs, CostSpec):
            if not isinstance(raw_costs, Mapping):
                raise ValueError("costs must be a mapping or CostSpec.")
            allowed_costs = {item.name for item in fields(CostSpec)}
            unknown_costs = set(raw_costs) - allowed_costs
            if unknown_costs:
                raise ValueError(f"Unknown CostSpec fields: {sorted(unknown_costs)}")
            values["costs"] = CostSpec(**dict(raw_costs))
        return cls(**values)
