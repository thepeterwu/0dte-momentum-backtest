from dataclasses import dataclass, field
import math
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from vol_surface.black76 import SliceAuditReport, audit_logger, process_snapshot_slice
from vol_surface.smile import SmileAuditReport, SmileSlice, fit_smile_slice

DEFAULT_DELTA_PILLARS = [-0.10, -0.25, -0.50, 0.25, 0.10]


@dataclass
class SurfaceAuditReport:
    """Comprehensive audit trail for 2D Volatility Surface assembly and arbitrage checks."""

    timestamp: str
    total_expiries_Attempted: int
    valid_slices_count: int = 0
    rejected_slices_count: int = 0
    rejected_expiries: list[dict] = field(default_factory=list)
    butterfly_arb_free: bool = True
    butterfly_violation_expiries: list[str] = field(default_factory=list)
    min_surface_g_density: float = 0.0
    calendar_delta_arb_free: bool = True
    calendar_delta_violations_count: int = 0
    calendar_delta_violation_details: list[dict] = field(default_factory=list)
    calendar_x_arb_free: bool = True
    calendar_x_violations_count: int = 0
    calendar_x_violation_details: list[dict] = field(default_factory=list)
    is_arbitrage_free: bool = True
    status: str = "INITIALIZED"
    rejection_reason: str | None = None
    slice_audits: dict[str, SliceAuditReport] = field(default_factory=dict)
    smile_audits: dict[str, SmileAuditReport] = field(default_factory=dict)


@dataclass
class VolSurface:
    """Arbitrage-Audited 2D Implied Volatility Surface in (Delta, Tau) coordinates.

    Constructed from calibrated 1D SmileSlice objects sorted by time-to-maturity tau.
    Interpolates across delta via Natural Cubic Splines / exact fixed-point inversion,
    and across maturity tau linearly in Total Implied Variance w(Delta, tau) = sigma^2 * tau.
    """

    timestamp: str
    slices: list[SmileSlice]
    delta_pillars: np.ndarray  # User-facing delta labels (e.g. [-0.10, -0.25, -0.50, 0.25, 0.10])
    call_deltas: np.ndarray  # Equivalent forward call deltas in (0, 1), sorted descending
    taus: np.ndarray  # Shape (M,): strictly increasing annualized maturities
    expiries: list[str]  # Length M: expiry date strings
    forwards: np.ndarray  # Shape (M,): implied forward prices F(tau)
    discounts: np.ndarray  # Shape (M,): implied discount factors D(tau)
    w_grid: np.ndarray  # Shape (M, D): total implied variance w(tau_i, delta_j)
    iv_grid: np.ndarray  # Shape (M, D): implied volatility sigma(tau_i, delta_j)
    x_grid: np.ndarray  # Shape (M, D): forward log-moneyness x(tau_i, delta_j)
    strike_grid: np.ndarray  # Shape (M, D): solved strikes K(tau_i, delta_j)
    surface_audit: SurfaceAuditReport

    def __post_init__(self):
        if len(self.slices) < 2:
            raise ValueError(
                f"[{self.timestamp}] VolSurface requires at least 2 valid maturity"
                f" slices (found {len(self.slices)})."
            )

        if np.any(np.diff(self.taus) <= 0):
            raise ValueError(
                f"[{self.timestamp}] Maturity vector taus must be strictly increasing."
            )

        # Build 1D Natural Cubic Splines in call-delta space for each maturity slice
        # CubicSpline requires strictly increasing x-coordinates, so we index on
        # u = 1.0 - call_delta = |put_delta| which increases strictly from 0.10 (10d Put) to 0.90 (10d Call)
        self.put_deltas = 1.0 - self.call_deltas
        if np.any(np.diff(self.put_deltas) <= 0):
            raise ValueError(
                f"[{self.timestamp}] Transformed delta axis must be strictly increasing."
            )

        self._delta_w_splines: list[CubicSpline] = []
        for i in range(len(self.slices)):
            spline_i = CubicSpline(
                self.put_deltas, self.w_grid[i, :], bc_type="natural"
            )
            self._delta_w_splines.append(spline_i)

        # Run static arbitrage audit immediately upon assembly
        self._audit_static_arbitrage()

    # =====================================================================
    # STATIC ARBITRAGE AUDITOR (BUTTERFLY + CALENDAR)
    # =====================================================================
    def _audit_static_arbitrage(self, tol: float = 1e-6) -> None:
        """Audits both Butterfly Arbitrage (g(x) >= 0) and Calendar Spread Arbitrage (dw/dtau >= 0)."""
        # 1. Butterfly Arbitrage Check across all slices
        min_g_list = []
        for s in self.slices:
            min_g_list.append(s.smile_audit.min_g_density)
            if not s.smile_audit.is_butterfly_arbitrage_free:
                self.surface_audit.butterfly_arb_free = False
                self.surface_audit.butterfly_violation_expiries.append(s.expiry)

        self.surface_audit.min_surface_g_density = float(min(min_g_list))

        # 2. Calendar Spread Check along Fixed Delta Pillars: w(Delta_j, tau_{i+1}) >= w(Delta_j, tau_i)
        n_slices, n_deltas = self.w_grid.shape
        for i in range(n_slices - 1):
            for j in range(n_deltas):
                w_short = float(self.w_grid[i, j])
                w_long = float(self.w_grid[i + 1, j])
                dw = w_long - w_short

                if dw < -tol:
                    self.surface_audit.calendar_delta_arb_free = False
                    self.surface_audit.calendar_delta_violations_count += 1
                    detail = {
                        "short_expiry": self.expiries[i],
                        "long_expiry": self.expiries[i + 1],
                        "delta_pillar": float(self.delta_pillars[j]),
                        "w_short": w_short,
                        "w_long": w_long,
                        "dw": dw,
                    }
                    self.surface_audit.calendar_delta_violation_details.append(detail)
                    audit_logger.warning(
                        f"[{self.timestamp}] Calendar Delta-Arbitrage detected between"
                        f" {self.expiries[i]} and {self.expiries[i + 1]} at"
                        f" Delta={self.delta_pillars[j]:+.2f}: w_short={w_short:.6f} >"
                        f" w_long={w_long:.6f} (dw={dw:.6f})."
                    )

        # 3. Calendar Spread Check along Fixed Forward Log-Moneyness: w(x, tau_{i+1}) >= w(x, tau_i)
        # Evaluate on overlapping interior moneyness range across adjacent slices
        for i in range(n_slices - 1):
            s_short = self.slices[i]
            s_long = self.slices[i + 1]

            x_min = max(s_short.x_left, s_long.x_left)
            x_max = min(s_short.x_right, s_long.x_right)

            if x_max > x_min:
                x_test = np.linspace(x_min, x_max, 50)
                w_short_x = np.asarray(s_short.get_total_variance(x_test), dtype=np.float64)
                w_long_x = np.asarray(s_long.get_total_variance(x_test), dtype=np.float64)
                dw_x = w_long_x - w_short_x

                min_dw_idx = int(np.argmin(dw_x))
                min_dw = float(dw_x[min_dw_idx])

                if min_dw < -tol:
                    bad_x = float(x_test[min_dw_idx])
                    self.surface_audit.calendar_x_arb_free = False
                    self.surface_audit.calendar_x_violations_count += int(np.sum(dw_x < -tol))
                    detail = {
                        "short_expiry": s_short.expiry,
                        "long_expiry": s_long.expiry,
                        "worst_x": bad_x,
                        "w_short": float(w_short_x[min_dw_idx]),
                        "w_long": float(w_long_x[min_dw_idx]),
                        "min_dw": min_dw,
                    }
                    self.surface_audit.calendar_x_violation_details.append(detail)
                    audit_logger.warning(
                        f"[{self.timestamp}] Calendar Moneyness-Arbitrage detected between"
                        f" {s_short.expiry} and {s_long.expiry} at x={bad_x:+.4f}:"
                        f" w_short={w_short_x[min_dw_idx]:.6f} >"
                        f" w_long={w_long_x[min_dw_idx]:.6f} (dw={min_dw:.6f})."
                    )

        self.surface_audit.is_arbitrage_free = (
                self.surface_audit.butterfly_arb_free
                and self.surface_audit.calendar_delta_arb_free
                and self.surface_audit.calendar_x_arb_free
        )

    # =====================================================================
    # BIVARIATE INTERPOLATION INTERFACE: (DELTA, TAU) -> W, IV
    # =====================================================================
    def _evaluate_slice_w_at_delta(
            self,
            slice_idx: int,
            target_delta: float,
            delta_method: str = "exact",
            allow_extrapolation: bool = False,
    ) -> float:
        """Evaluates total variance w(target_delta, tau_i) on maturity slice_idx."""
        call_delta = SmileSlice._normalize_to_call_delta(target_delta)
        put_delta = 1.0 - call_delta

        min_put_d = float(self.put_deltas[0])
        max_put_d = float(self.put_deltas[-1])

        if not (min_put_d - 1e-6 <= put_delta <= max_put_d + 1e-6):
            if not allow_extrapolation:
                msg = (
                    f"[{self.timestamp}] Target delta {target_delta:+.2f} (put_delta={put_delta:+.2f})"
                    f" is outside calibrated delta grid bounds [{min_put_d:+.2f}, {max_put_d:+.2f}]."
                )
                audit_logger.error(msg)
                raise ValueError(msg)
            audit_logger.warning(
                f"[{self.timestamp}] Extrapolating delta={target_delta:+.2f} outside"
                f" pillar bounds [{min_put_d:+.2f}, {max_put_d:+.2f}]."
            )

        if delta_method == "exact":
            x_star = self.slices[slice_idx].solve_x_from_delta(target_delta)
            if x_star is None:
                raise ValueError(
                    f"[{self.timestamp}] Exact delta inversion failed on expiry"
                    f" {self.expiries[slice_idx]} for delta={target_delta:+.2f}."
                )
            return float(self.slices[slice_idx].get_total_variance(x_star))

        elif delta_method == "grid_spline":
            w_val = float(self._delta_w_splines[slice_idx](put_delta))
            if w_val <= 0.0:
                raise ValueError(
                    f"[{self.timestamp}] Grid spline produced non-positive total"
                    f" variance w={w_val:.6e} at delta={target_delta:+.2f}."
                )
            return w_val

        else:
            raise ValueError(
                f"Unknown delta_method='{delta_method}'. Use 'exact' or 'grid_spline'."
            )

    def get_total_variance(
            self,
            target_delta: float,
            target_tau: float,
            delta_method: str = "exact",
            allow_extrapolation: bool = False,
    ) -> float:
        """Interpolates Total Implied Variance w(target_delta, target_tau).

        Interpolates across delta using `delta_method` ('exact' or 'grid_spline')
        and linearly in total variance across maturity `target_tau`.
        """
        if target_tau <= 0.0:
            raise ValueError(f"target_tau={target_tau} must be strictly positive.")

        tau_min = float(self.taus[0])
        tau_max = float(self.taus[-1])

        # 1. Check maturity domain bounds
        if target_tau < tau_min - 1e-9 or target_tau > tau_max + 1e-9:
            if not allow_extrapolation:
                msg = (
                    f"[{self.timestamp}] Target tau={target_tau:.6f} is outside"
                    f" calibrated maturity bounds [{tau_min:.6f}, {tau_max:.6f}]."
                )
                audit_logger.error(msg)
                raise ValueError(msg)

            audit_logger.warning(
                f"[{self.timestamp}] Extrapolating tau={target_tau:.6f} outside"
                f" calibrated maturity bounds [{tau_min:.6f}, {tau_max:.6f}]."
            )
            # Flat-volatility (linear total variance from origin) clamping at boundary
            boundary_idx = 0 if target_tau < tau_min else (len(self.taus) - 1)
            w_bound = self._evaluate_slice_w_at_delta(
                boundary_idx, target_delta, delta_method, allow_extrapolation
            )
            return w_bound * (target_tau / float(self.taus[boundary_idx]))

        # 2. Check if target_tau matches an existing maturity slice directly
        exact_matches = np.where(np.isclose(self.taus, target_tau, atol=1e-9))[0]
        if len(exact_matches) > 0:
            idx = int(exact_matches[0])
            return self._evaluate_slice_w_at_delta(
                idx, target_delta, delta_method, allow_extrapolation
            )

        # 3. Locate bracketing maturity slices: taus[k] < target_tau < taus[k + 1]
        k = int(np.searchsorted(self.taus, target_tau) - 1)
        k = max(0, min(k, len(self.taus) - 2))

        tau_k = float(self.taus[k])
        tau_k1 = float(self.taus[k + 1])

        w_k = self._evaluate_slice_w_at_delta(
            k, target_delta, delta_method, allow_extrapolation
        )
        w_k1 = self._evaluate_slice_w_at_delta(
            k + 1, target_delta, delta_method, allow_extrapolation
        )

        # Linear interpolation in total implied variance
        weight = (target_tau - tau_k) / (tau_k1 - tau_k)
        w_interp = w_k + weight * (w_k1 - w_k)

        if w_interp <= 0.0:
            raise ValueError(
                f"[{self.timestamp}] Interpolated total variance w={w_interp:.6e} <= 0."
            )

        return float(w_interp)

    def get_iv(
            self,
            target_delta: float,
            target_tau: float,
            delta_method: str = "exact",
            allow_extrapolation: bool = False,
    ) -> float:
        """Returns interpolated Black-76 Implied Volatility sigma(target_delta, target_tau)."""
        w_val = self.get_total_variance(
            target_delta=target_delta,
            target_tau=target_tau,
            delta_method=delta_method,
            allow_extrapolation=allow_extrapolation,
        )
        return math.sqrt(w_val / target_tau)

    def to_dataframe(self) -> pd.DataFrame:
        """Exports the calibrated 2D (Expiry x Delta) implied volatility surface as a DataFrame."""
        cols = [f"Delta_{d:+.2f}" for d in self.delta_pillars]
        df = pd.DataFrame(self.iv_grid, index=self.expiries, columns=cols)
        df.insert(0, "tau", self.taus)
        df.insert(1, "forward", self.forwards)
        df.insert(2, "discount", self.discounts)
        df.index.name = "expiry"
        return df

    # =====================================================================
    # FACTORY CONSTRUCTOR FROM RAW 1-MINUTE SNAPSHOT
    # =====================================================================
    @classmethod
    def from_snapshot(
            cls,
            snapshot_df: pd.DataFrame,
            delta_pillars: list[float] | np.ndarray = DEFAULT_DELTA_PILLARS,
    ) -> tuple["VolSurface | None", SurfaceAuditReport]:
        """Builds a complete 2D VolSurface from a 1-minute option chain snapshot DataFrame.

        Executes Phase 1 through Phase 5 across all expiries and logs any anomalies.
        """
        if snapshot_df.empty:
            audit = SurfaceAuditReport(timestamp="EMPTY_SNAPSHOT", total_expiries_Attempted=0)
            audit.status = "REJECTED"
            audit.rejection_reason = "EMPTY_SNAPSHOT_DATAFRAME"
            audit_logger.error("VolSurface.from_snapshot received an empty DataFrame.")
            return None, audit

        timestamp_str = str(snapshot_df["timestamp"].iloc[0])
        raw_expiries = sorted(snapshot_df["expiry"].astype(str).unique())

        surface_audit = SurfaceAuditReport(
            timestamp=timestamp_str,
            total_expiries_Attempted=len(raw_expiries),
        )

        valid_slices: list[SmileSlice] = []
        valid_grids: list[dict[str, np.ndarray]] = []

        for exp_str in raw_expiries:
            # Phase 1: Quote Cleaning & Put-Call Parity Forward Calibration
            slice_obj, slice_audit = process_snapshot_slice(snapshot_df, exp_str)
            surface_audit.slice_audits[exp_str] = slice_audit

            if slice_obj is None:
                surface_audit.rejected_slices_count += 1
                surface_audit.rejected_expiries.append({
                    "expiry": exp_str,
                    "phase": "PHASE_1_QUOTE_PREP",
                    "reason": slice_audit.rejection_reason,
                })
                continue

            # Phase 2 & 3: IV Inversion & Total Variance Cubic Spline Fitting
            smile_obj, smile_audit = fit_smile_slice(slice_obj)
            surface_audit.smile_audits[exp_str] = smile_audit

            if smile_obj is None:
                surface_audit.rejected_slices_count += 1
                surface_audit.rejected_expiries.append({
                    "expiry": exp_str,
                    "phase": "PHASE_3_SMILE_SPLINE",
                    "reason": smile_audit.rejection_reason,
                })
                continue

            # Phase 4: Strike-to-Delta Fixed-Point Grid Inversion
            grid_res = smile_obj.evaluate_delta_grid(delta_pillars)
            if grid_res is None:
                surface_audit.rejected_slices_count += 1
                surface_audit.rejected_expiries.append({
                    "expiry": exp_str,
                    "phase": "PHASE_4_DELTA_INVERSION",
                    "reason": "DELTA_GRID_EVALUATION_FAILED",
                })
                continue

            valid_slices.append(smile_obj)
            valid_grids.append(grid_res)

        surface_audit.valid_slices_count = len(valid_slices)

        # Require at least 2 valid maturities to form a 2D surface
        if len(valid_slices) < 2:
            surface_audit.status = "REJECTED"
            surface_audit.rejection_reason = (
                f"INSUFFICIENT_VALID_MATURITIES: Found {len(valid_slices)}, required >= 2"
            )
            audit_logger.error(
                f"[{timestamp_str}] Surface construction failed: {surface_audit.rejection_reason}"
            )
            return None, surface_audit

        # Sort slices strictly by increasing time-to-maturity tau
        sort_order = np.argsort([s.tau for s in valid_slices])
        sorted_slices = [valid_slices[i] for i in sort_order]
        sorted_grids = [valid_grids[i] for i in sort_order]

        taus = np.array([s.tau for s in sorted_slices], dtype=np.float64)
        expiries = [s.expiry for s in sorted_slices]
        forwards = np.array([s.forward for s in sorted_slices], dtype=np.float64)
        discounts = np.array([s.discount_factor for s in sorted_slices], dtype=np.float64)

        sorted_pillars = sorted_grids[0]["delta_pillars"]
        sorted_call_deltas = sorted_grids[0]["call_deltas"]

        w_grid = np.vstack([g["total_variance"] for g in sorted_grids])
        iv_grid = np.vstack([g["iv"] for g in sorted_grids])
        x_grid = np.vstack([g["x"] for g in sorted_grids])
        strike_grid = np.vstack([g["strikes"] for g in sorted_grids])

        try:
            surface = cls(
                timestamp=timestamp_str,
                slices=sorted_slices,
                delta_pillars=sorted_pillars,
                call_deltas=sorted_call_deltas,
                taus=taus,
                expiries=expiries,
                forwards=forwards,
                discounts=discounts,
                w_grid=w_grid,
                iv_grid=iv_grid,
                x_grid=x_grid,
                strike_grid=strike_grid,
                surface_audit=surface_audit,
            )
        except Exception as e:
            surface_audit.status = "REJECTED"
            surface_audit.rejection_reason = f"SURFACE_ASSEMBLY_ERROR: {e}"
            audit_logger.error(f"[{timestamp_str}] {surface_audit.rejection_reason}")
            return None, surface_audit

        surface_audit.status = (
            "PASSED" if surface_audit.is_arbitrage_free else "PASSED_WITH_ARB_WARNING"
        )
        audit_logger.info(
            f"[{timestamp_str}] Assembled 2D VolSurface across {len(sorted_slices)} expiries."
            f" ButterflyArbFree={surface_audit.butterfly_arb_free},"
            f" CalendarDeltaArbFree={surface_audit.calendar_delta_arb_free},"
            f" CalendarXArbFree={surface_audit.calendar_x_arb_free}."
        )

        return surface, surface_audit
