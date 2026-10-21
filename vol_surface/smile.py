from dataclasses import dataclass, field
import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.stats import norm
from vol_surface.black76 import ExpirySlice, audit_logger, invert_iv_newton_raphson


@dataclass
class SmileAuditReport:
    """Audit record for Total Variance Smile calibration."""

    expiry: str
    tau: float
    input_strikes_count: int
    valid_iv_strikes_count: int = 0
    dropped_iv_inversion_count: int = 0
    delta_grid_evaluated: bool = False
    delta_fp_fallbacks_count: int = 0
    delta_solver_failures_count: int = 0
    is_delta_grid_monotonic: bool = True
    delta_inversion_details: list[dict] = field(default_factory=list)
    raw_slope_left: float = 0.0
    raw_slope_right: float = 0.0
    clamped_slope_left: float = 0.0
    clamped_slope_right: float = 0.0
    left_wing_clamped: bool = False
    right_wing_clamped: bool = False
    min_g_density: float = 0.0
    is_butterfly_arbitrage_free: bool = True
    status: str = "INITIALIZED"
    rejection_reason: str | None = None
    iv_rejection_details: list[dict] = field(default_factory=list)


@dataclass
class SmileSlice:
    """Calibrated 1D Total Implied Variance Smile w(x) = sigma^2(x) * tau.

    Coordinate convention:
      x = ln(K / F)
      - x < 0 : OTM Puts (K < F)
      - x = 0 : ATM (K = F)
      - x > 0 : OTM Calls (K > F)

    Interpolates interior knots via a Natural Cubic Spline on w(x) and
    extrapolates wings linearly in w(x) subject to Lee's Moment Formula (|w'| <= 2).
    """

    expiry: str
    tau: float
    forward: float
    discount_factor: float
    x_market: np.ndarray  # Strictly increasing log-moneyness x = ln(K / F)
    w_market: np.ndarray  # Market total implied variance: w = sigma^2 * tau
    iv_market: np.ndarray  # Market implied volatility: sigma
    strikes: np.ndarray  # Market strikes K
    smile_audit: SmileAuditReport

    def __post_init__(self):
        if len(self.x_market) < 4:
            raise ValueError(
                f"[{self.expiry}] Need at least 4 valid strikes to fit cubic"
                f" spline (found {len(self.x_market)})."
            )

        if np.any(np.diff(self.x_market) <= 0):
            raise ValueError(
                f"[{self.expiry}] x_market must be strictly increasing."
            )

        if np.any(self.w_market <= 0):
            raise ValueError(
                f"[{self.expiry}] Total variance w_market must be strictly positive."
            )

        # 1. Fit Natural Cubic Spline directly in Total Variance space: x |-> w(x)
        # bc_type='natural' enforces w''(x_left) = 0 and w''(x_right) = 0
        self._w_spline = CubicSpline(
            self.x_market, self.w_market, bc_type="natural"
        )

        # 2. Extract boundary knots and raw first derivatives dw/dx
        self.x_left = float(self.x_market[0])
        self.x_right = float(self.x_market[-1])

        self.w_left = float(self.w_market[0])
        self.w_right = float(self.w_market[-1])

        raw_slope_left = float(self._w_spline(self.x_left, 1))
        raw_slope_right = float(self._w_spline(self.x_right, 1))

        self.smile_audit.raw_slope_left = raw_slope_left
        self.smile_audit.raw_slope_right = raw_slope_right

        # 3. Enforce Lee's Moment Formula Wing Bounds with Explicit Logging
        # Left wing (x -> -inf): w'(x) must lie in [-2.0, 0.0] so w(x) stays positive and g(x) >= 0
        if not (-2.0 <= raw_slope_left <= 0.0):
            self.slope_left = max(-2.0, min(0.0, raw_slope_left))
            self.smile_audit.left_wing_clamped = True
            audit_logger.warning(
                f"[{self.expiry}] Left wing total variance slope dw/dx={raw_slope_left:.4f}"
                f" breached Lee's no-arbitrage bound [-2.0, 0.0]; clamped to {self.slope_left:.4f}."
            )
        else:
            self.slope_left = raw_slope_left

        # Right wing (x -> +inf): w'(x) must lie in [0.0, 2.0] so w(x) stays positive and g(x) >= 0
        if not (0.0 <= raw_slope_right <= 2.0):
            self.slope_right = max(0.0, min(2.0, raw_slope_right))
            self.smile_audit.right_wing_clamped = True
            audit_logger.warning(
                f"[{self.expiry}] Right wing total variance slope dw/dx={raw_slope_right:.4f}"
                f" breached Lee's no-arbitrage bound [0.0, 2.0]; clamped to {self.slope_right:.4f}."
            )
        else:
            self.slope_right = raw_slope_right

        self.smile_audit.clamped_slope_left = self.slope_left
        self.smile_audit.clamped_slope_right = self.slope_right

        # 4. Audit Butterfly Arbitrage Density g(x) across a fine grid
        x_audit_grid = np.linspace(self.x_left - 0.05, self.x_right + 0.05, 200)
        g_vals = self.compute_density_g(x_audit_grid)
        min_g = float(np.min(g_vals))
        self.smile_audit.min_g_density = min_g

        if min_g < -1e-6:
            self.smile_audit.is_butterfly_arbitrage_free = False
            bad_x = x_audit_grid[int(np.argmin(g_vals))]
            audit_logger.warning(
                f"[{self.expiry}] Butterfly arbitrage detected in fitted spline:"
                f" min g(x) = {min_g:.6f} < 0 at log-moneyness x = {bad_x:.4f}"
                f" (K = {self.forward * math.exp(bad_x):.1f})."
            )

    def get_total_variance(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluates total implied variance w(x) = sigma^2(x) * tau at x = ln(K / F)."""
        x_arr = np.asarray(x, dtype=np.float64)
        is_scalar = x_arr.ndim == 0
        x_flat = np.atleast_1d(x_arr)

        w_vals = self._w_spline(x_flat)

        # Left linear wing (x < x_left)
        left_mask = x_flat < self.x_left
        if np.any(left_mask):
            w_vals[left_mask] = self.w_left + self.slope_left * (
                    x_flat[left_mask] - self.x_left
            )

        # Right linear wing (x > x_right)
        right_mask = x_flat > self.x_right
        if np.any(right_mask):
            w_vals[right_mask] = self.w_right + self.slope_right * (
                    x_flat[right_mask] - self.x_right
            )

        # Check for non-positive variance inside interior spline oscillations
        if np.any(w_vals <= 0.0):
            min_w = float(np.min(w_vals))
            raise ValueError(
                f"[{self.expiry}] Non-positive total variance w(x)={min_w:.6e}"
                " encountered during evaluation."
            )

        return float(w_vals[0]) if is_scalar else w_vals

    def get_iv(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluates implied volatility sigma(x) = sqrt(w(x) / tau) at x = ln(K / F)."""
        w_vals = self.get_total_variance(x)
        iv_vals = np.sqrt(w_vals / self.tau)
        return float(iv_vals) if np.ndim(iv_vals) == 0 else iv_vals

    def get_iv_from_strike(self, strike: float | np.ndarray) -> float | np.ndarray:
        """Computes x = ln(K / F) and returns implied volatility sigma(x)."""
        strike_arr = np.asarray(strike, dtype=np.float64)
        if np.any(strike_arr <= 0):
            raise ValueError("Strike must be strictly positive.")
        x = np.log(strike_arr / self.forward)
        return self.get_iv(x)

    def get_w_first_derivative(self, x: float | np.ndarray) -> float | np.ndarray:
        """Returns first derivative of total variance: w'(x) = dw / dx."""
        x_arr = np.asarray(x, dtype=np.float64)
        is_scalar = x_arr.ndim == 0
        x_flat = np.atleast_1d(x_arr)

        dw = self._w_spline(x_flat, 1)
        dw[x_flat < self.x_left] = self.slope_left
        dw[x_flat > self.x_right] = self.slope_right

        return float(dw[0]) if is_scalar else dw

    def get_w_second_derivative(self, x: float | np.ndarray) -> float | np.ndarray:
        """Returns second derivative of total variance: w''(x) = d^2w / dx^2."""
        x_arr = np.asarray(x, dtype=np.float64)
        is_scalar = x_arr.ndim == 0
        x_flat = np.atleast_1d(x_arr)

        d2w = self._w_spline(x_flat, 2)
        # Linear wings have zero second derivative
        d2w[x_flat < self.x_left] = 0.0
        d2w[x_flat > self.x_right] = 0.0

        return float(d2w[0]) if is_scalar else d2w

    def compute_density_g(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluates Gatheral's risk-neutral density function g(x).

        Static butterfly no-arbitrage requires g(x) >= 0 for all x.
        g(x) = (1 - x * w' / (2 * w))^2 - (w'^2 / 4) * (1 / w + 1 / 4) + w'' / 2
        """
        x_arr = np.asarray(x, dtype=np.float64)
        is_scalar = x_arr.ndim == 0
        x_flat = np.atleast_1d(x_arr)

        w = np.maximum(np.atleast_1d(self.get_total_variance(x_flat)), 1e-8)
        dw = np.atleast_1d(self.get_w_first_derivative(x_flat))
        d2w = np.atleast_1d(self.get_w_second_derivative(x_flat))

        term1 = (1.0 - (x_flat * dw) / (2.0 * w)) ** 2
        term2 = -0.25 * (dw ** 2) * (1.0 / w + 0.25)
        term3 = 0.5 * d2w

        g = term1 + term2 + term3
        return float(g[0]) if is_scalar else g

    # =====================================================================
    # STRIKE-TO-DELTA INVERSION METHODS (PROPERLY SCOPED IN CLASS)
    # =====================================================================
    def get_forward_delta(
            self, x: float | np.ndarray, is_call: bool = True
    ) -> float | np.ndarray:
        """Computes Black-76 forward delta at log-moneyness x = ln(K / F).

        Call delta = N(d1) in (0, 1)
        Put delta  = N(d1) - 1 in (-1, 0)
        """
        x_arr = np.asarray(x, dtype=np.float64)
        is_scalar = x_arr.ndim == 0
        x_flat = np.atleast_1d(x_arr)

        w = np.maximum(np.atleast_1d(self.get_total_variance(x_flat)), 1e-8)
        sqrt_w = np.sqrt(w)

        # Since x = ln(K / F), ln(F / K) = -x
        d1 = (-x_flat + 0.5 * w) / sqrt_w
        call_delta = norm.cdf(d1)

        res = call_delta if is_call else (call_delta - 1.0)
        return float(res[0]) if is_scalar else res

    @staticmethod
    def _normalize_to_call_delta(target_delta: float) -> float:
        """Converts signed delta or put delta into equivalent forward call delta in (0, 1).

        Convention:
          - If target_delta < 0 (e.g. -0.25 for 25-delta Put): Call delta = 1.0 + target_delta (0.75)
          - If target_delta > 0 (e.g. +0.25 for 25-delta Call): Call delta = target_delta (0.25)
        """
        if target_delta == 0.0 or abs(target_delta) >= 1.0:
            raise ValueError(
                f"target_delta={target_delta} is invalid. Must be in (-1, 0) for Puts"
                " or (0, 1) for Calls."
            )
        return (1.0 + target_delta) if target_delta < 0.0 else target_delta

    def solve_x_from_delta(
            self,
            target_delta: float,
            max_iter: int = 50,
            tol: float = 1e-7,
    ) -> float | None:
        """Solves for log-moneyness x* = ln(K* / F) matching target_delta.

        Primary method: Fixed-Point Iteration on w(x).
        Audited fallback: Bracketed Brent's method if steep skew prevents contraction.
        """
        call_delta = self._normalize_to_call_delta(target_delta)
        d1_star = float(norm.ppf(call_delta))

        # Fixed-Point Iteration starting from ATM total variance w(0)
        w_curr = float(self.get_total_variance(0.0))
        x_curr = 0.5 * w_curr - d1_star * math.sqrt(w_curr)

        fp_converged = False
        fp_failure_reason = "MAX_ITER_EXCEEDED"

        for _ in range(max_iter):
            try:
                w_next = float(self.get_total_variance(x_curr))
            except ValueError as e:
                fp_failure_reason = f"VARIANCE_DOMAIN_ERROR ({e})"
                break

            x_next = 0.5 * w_next - d1_star * math.sqrt(w_next)

            if abs(x_next - x_curr) < tol:
                x_curr = x_next
                fp_converged = True
                break

            # Guard against divergent oscillation in extreme wings
            if abs(x_next) > 1.5:
                fp_failure_reason = f"ITERATION_DIVERGED (x={x_next:.4f})"
                break

            x_curr = x_next

        # Verify delta accuracy of fixed-point solution
        if fp_converged:
            achieved_call_delta = float(self.get_forward_delta(x_curr, is_call=True))
            if abs(achieved_call_delta - call_delta) < 1e-5:
                return x_curr
            fp_failure_reason = (
                f"DELTA_RESIDUAL_TOO_LARGE (got {achieved_call_delta:.6f} vs {call_delta:.6f})"
            )

        # Logged Fallback: Bracketed Brent's Method on exact residual R(x) = 0
        self.smile_audit.delta_fp_fallbacks_count += 1
        audit_logger.warning(
            f"[{self.expiry} Delta={target_delta:+.2f}] Fixed-point iteration did not"
            f" converge ({fp_failure_reason}); falling back to Brentq root-finder."
        )

        def residual(x_val: float) -> float:
            w_val = float(self.get_total_variance(x_val))
            return -x_val + 0.5 * w_val - d1_star * math.sqrt(w_val)

        try:
            x_brent = float(brentq(residual, a=-1.0, b=1.0, xtol=tol, maxiter=100))
            return x_brent
        except Exception as e:
            self.smile_audit.delta_solver_failures_count += 1
            self.smile_audit.delta_inversion_details.append({
                "target_delta": target_delta,
                "call_delta_equiv": call_delta,
                "issue": f"DELTA_SOLVER_FAILED: {e}",
            })
            audit_logger.error(
                f"[{self.expiry} Delta={target_delta:+.2f}] Both Fixed-Point and"
                f" Brentq failed to solve log-moneyness: {e}"
            )
            return None

    def get_iv_at_delta(self, target_delta: float) -> float | None:
        """Returns implied volatility sigma(x*) at the specified target_delta."""
        x_star = self.solve_x_from_delta(target_delta)
        if x_star is None:
            return None
        return float(self.get_iv(x_star))

    def evaluate_delta_grid(
            self,
            delta_pillars: list[float] | np.ndarray,
    ) -> dict[str, np.ndarray] | None:
        """Evaluates the smile across an ordered sequence of target deltas.

        Verifies that solved log-moneyness x* increases strictly as forward call delta
        decreases (high strike K -> high x -> low call delta).
        """
        pillars = np.asarray(delta_pillars, dtype=np.float64)
        call_deltas = np.array(
            [self._normalize_to_call_delta(float(d)) for d in pillars],
            dtype=np.float64,
        )

        # Sort pillars from the lowest strike (deep OTM Put) to the highest strike (deep OTM Call)
        sort_idx = np.argsort(-call_deltas)
        sorted_pillars = pillars[sort_idx]
        sorted_call_deltas = call_deltas[sort_idx]

        solved_x = []
        solved_w = []
        solved_iv = []
        solved_strikes = []

        for d in sorted_pillars:
            x_star = self.solve_x_from_delta(float(d))
            if x_star is None:
                audit_logger.error(
                    f"[{self.expiry}] Delta grid evaluation aborted: failed at pillar {d:+.2f}."
                )
                return None

            w_star = float(self.get_total_variance(x_star))
            iv_star = math.sqrt(w_star / self.tau)
            k_star = self.forward * math.exp(x_star)

            solved_x.append(x_star)
            solved_w.append(w_star)
            solved_iv.append(iv_star)
            solved_strikes.append(k_star)

        x_arr = np.array(solved_x, dtype=np.float64)
        self.smile_audit.delta_grid_evaluated = True

        # Audit Monotonicity: As Call Delta decreases, Strike K and x = ln(K/F) must strictly increase
        if np.any(np.diff(x_arr) <= 0):
            self.smile_audit.is_delta_grid_monotonic = False
            audit_logger.error(
                f"[{self.expiry}] NON_MONOTONIC_DELTA_MAP: Solved log-moneyness x={x_arr}"
                f" is not strictly increasing across call deltas={sorted_call_deltas}."
            )
            return None

        return {
            "delta_pillars": sorted_pillars,
            "call_deltas": sorted_call_deltas,
            "put_deltas": sorted_call_deltas - 1.0,
            "x": x_arr,
            "strikes": np.array(solved_strikes, dtype=np.float64),
            "total_variance": np.array(solved_w, dtype=np.float64),
            "iv": np.array(solved_iv, dtype=np.float64),
        }


# =====================================================================
# FACTORY BUILDER FUNCTION
# =====================================================================
def fit_smile_slice(
        slice_data: ExpirySlice,
) -> tuple[SmileSlice | None, SmileAuditReport]:
    """Inverts OTM market quotes to Black-76 IVs, transforms to total variance w = sigma^2 * tau,

    and calibrates the natural cubic spline with audited Lee wing bounds.
    """
    K = slice_data.strikes
    F = slice_data.forward
    tau = slice_data.tau
    D = slice_data.discount_factor
    prices = slice_data.otm_prices
    is_call = slice_data.is_call

    smile_audit = SmileAuditReport(
        expiry=slice_data.expiry,
        tau=tau,
        input_strikes_count=len(K),
    )

    # Forward log-moneyness: x = ln(K / F)
    x_all = np.log(K / F)

    ivs = []
    valid_indices = []

    for i in range(len(K)):
        sigma = invert_iv_newton_raphson(
            price=float(prices[i]),
            F=F,
            K=float(K[i]),
            tau=tau,
            discount_factor=D,
            is_call=bool(is_call[i]),
            expiry=slice_data.expiry,
        )

        if sigma is None:
            smile_audit.dropped_iv_inversion_count += 1
            smile_audit.iv_rejection_details.append({
                "strike": float(K[i]),
                "price": float(prices[i]),
                "issue": "IV_SOLVER_OR_BOUND_FAILURE",
            })
            continue

        if not (0.02 <= sigma <= 2.50):
            smile_audit.dropped_iv_inversion_count += 1
            smile_audit.iv_rejection_details.append({
                "strike": float(K[i]),
                "price": float(prices[i]),
                "sigma": float(sigma),
                "issue": "EXTREME_IV_OUT_OF_RANGE",
            })
            audit_logger.warning(
                f"[{slice_data.expiry} K={K[i]}] Rejected extreme IV={sigma:.4f}"
                " outside [0.02, 2.50]."
            )
            continue

        ivs.append(sigma)
        valid_indices.append(i)

    smile_audit.valid_iv_strikes_count = len(valid_indices)

    if len(valid_indices) < 4:
        smile_audit.status = "REJECTED"
        smile_audit.rejection_reason = (
            f"INSUFFICIENT_VALID_IV_STRIKES: {len(valid_indices)} valid, required >= 4"
        )
        audit_logger.error(f"[{slice_data.expiry}] {smile_audit.rejection_reason}")
        return None, smile_audit

    x_valid = x_all[valid_indices]
    iv_valid = np.array(ivs, dtype=np.float64)
    strikes_valid = K[valid_indices]

    # Sort strictly by x = ln(K / F)
    sort_idx = np.argsort(x_valid)
    x_sorted = x_valid[sort_idx]
    iv_sorted = iv_valid[sort_idx]
    strikes_sorted = strikes_valid[sort_idx]

    # Transform to Total Implied Variance: w(x) = sigma^2(x) * tau
    w_sorted = (iv_sorted ** 2) * tau

    try:
        smile = SmileSlice(
            expiry=slice_data.expiry,
            tau=tau,
            forward=F,
            discount_factor=D,
            x_market=x_sorted,
            w_market=w_sorted,
            iv_market=iv_sorted,
            strikes=strikes_sorted,
            smile_audit=smile_audit,
        )
    except Exception as e:
        smile_audit.status = "REJECTED"
        smile_audit.rejection_reason = f"SPLINE_CONSTRUCTION_ERROR: {e}"
        audit_logger.error(f"[{slice_data.expiry}] {smile_audit.rejection_reason}")
        return None, smile_audit

    smile_audit.status = (
        "PASSED" if smile_audit.is_butterfly_arbitrage_free else "PASSED_WITH_ARB_WARNING"
    )
    audit_logger.info(
        f"[{slice_data.expiry}] Fitted Total Variance Smile: {len(x_sorted)} knots,"
        f" min_g={smile_audit.min_g_density:.4f},"
        f" arb_free={smile_audit.is_butterfly_arbitrage_free}"
    )

    return smile, smile_audit
