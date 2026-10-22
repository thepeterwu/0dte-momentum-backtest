from dataclasses import dataclass, field
from datetime import datetime
import logging
import math
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

NY_TZ = ZoneInfo("America/New_York")

# =====================================================================
# AUDIT LOGGER SETUP
# =====================================================================
audit_logger = logging.getLogger("DataHygieneAudit")
audit_logger.setLevel(logging.INFO)

# File handler specifically for data hygiene anomalies
if not audit_logger.handlers:
    fh = logging.FileHandler("data_hygiene_audit.log", mode="a")
    fh.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    )
    audit_logger.addHandler(fh)


@dataclass
class SliceAuditReport:
    """Detailed record of quote sanitization and parity calibration."""

    timestamp: str
    expiry: str
    tau_minutes: float
    raw_quotes_count: int
    cleaned_quotes_count: int = 0
    dropped_crossed_quotes: int = 0
    dropped_duplicate_quotes: int = 0
    dropped_min_tick_quotes: int = 0
    dropped_missing_or_negative_asks: int = 0
    dropped_non_monotonic_quotes: int = 0
    dropped_wide_spreads: int = 0
    dropped_zero_or_negative_bids: int = 0
    total_pairs_found: int = 0
    ntm_pairs_found: int = 0
    parity_r2: float = 0.0
    inferred_discount: float = 0.0
    inferred_forward: float = 0.0
    status: str = "INITIALIZED"
    rejection_reason: str | None = None
    rejected_strikes_detail: list[dict] = field(default_factory=list)


@dataclass
class ExpirySlice:
    """Sanitized market data container for an individual maturity slice."""

    expiry: str
    tau: float
    forward: float
    discount_factor: float
    strikes: np.ndarray
    otm_prices: np.ndarray
    is_call: np.ndarray
    bids: np.ndarray
    asks: np.ndarray
    audit: SliceAuditReport


# =====================================================================
# TIME-TO-MATURITY WITH VALIDATION
# =====================================================================
def compute_tau(
        timestamp_str: str, expiry_str: str, audit: SliceAuditReport
) -> float | None:
    """
    Compute time til maturity Tau in terms of years from mins (using 365 days in year)
    :param timestamp_str: time stamp string
    :param expiry_str: expiry string
    :param audit: error report
    :return: years til maturity
    """

    try:
        ts = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=NY_TZ
        )
        exp_date = datetime.strptime(expiry_str, "%Y%m%d").replace(tzinfo=NY_TZ)
        # SPXW settles at 16:00 ET on expiration date
        expiry_dt = exp_date.replace(hour=16, minute=0, second=0)
    except Exception as e:
        audit.status = "REJECTED"
        audit.rejection_reason = f"TIMESTAMP_PARSE_ERROR: {str(e)}"
        audit_logger.error(
            f"[{expiry_str}] Failed to parse timestamp {timestamp_str}: {e}"
        )
        return None

    delta = expiry_dt - ts
    total_minutes = delta.total_seconds() / 60.0
    audit.tau_minutes = total_minutes

    if total_minutes <= 10.0:
        audit.status = "REJECTED"
        audit.rejection_reason = (
            f"EXPIRING_IMMINENTLY: {total_minutes:.1f}m remaining"
        )
        audit_logger.warning(
            f"[{expiry_str}] Slice rejected: only {total_minutes:.1f} minutes to"
            " settlement."
        )
        return None

    return total_minutes / (365.0 * 1440.0)


# =====================================================================
# AUDITED PUT-CALL PARITY REGRESSION
# =====================================================================
def extract_forward_and_discount(
        calls_df: pd.DataFrame,
        puts_df: pd.DataFrame,
        spot_ref: float,
        tau: float,
        audit: SliceAuditReport,
) -> tuple[float, float] | None:
    """Calibrates put-call parity strictly on NTM pairs without silent fallbacks."""
    pairs = pd.merge(
        calls_df[["strike", "mid"]],
        puts_df[["strike", "mid"]],
        on="strike",
        suffixes=("_call", "_put"),
    )
    audit.total_pairs_found = len(pairs)

    # Restrict regression strictly to near-the-money (+/- 2% around spot)
    ntm_pairs = pairs[
        (pairs["strike"] >= spot_ref * 0.98) & (pairs["strike"] <= spot_ref * 1.02)
        ]
    audit.ntm_pairs_found = len(ntm_pairs)

    # Require at least n distinct NTM matched pairs; do not silently fall back to wings
    n = 5
    if len(ntm_pairs) < n:
        audit.status = "REJECTED"
        audit.rejection_reason = (
            f"INSUFFICIENT_NTM_PARITY_PAIRS: Found {len(ntm_pairs)} in +/-2% NTM"
            f" window (total pairs={len(pairs)}), required >= {n}"
        )
        audit_logger.error(
            f"[{audit.expiry}] {audit.rejection_reason} at spot {spot_ref:.2f}"
        )
        return None

    K = ntm_pairs["strike"].values
    y = (ntm_pairs["mid_call"] - ntm_pairs["mid_put"]).values

    # Ordinary Least Squares: y = beta_0 + beta_1 * K
    poly = np.polyfit(K, y, deg=1)
    beta_1, beta_0 = poly[0], poly[1]

    y_pred = np.polyval(poly, K)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot <= 1e-12:
        audit.status = "REJECTED"
        audit.rejection_reason = (
            "DEGENERATE_PARITY_REGRESSION: Zero variance in (C - P)"
        )
        audit_logger.error(f"[{audit.expiry}] {audit.rejection_reason}")
        return None

    r_squared = 1.0 - (ss_res / ss_tot)
    audit.parity_r2 = float(r_squared)

    discount_factor = -beta_1
    audit.inferred_discount = float(discount_factor)

    # risk-free rate bounds check between -2% to 12%
    max_rf_rate = 0.12
    min_rf_rate = -0.02
    min_valid_d = math.exp(-max_rf_rate * tau) - 0.015
    max_valid_d = math.exp(-min_rf_rate * tau) + 0.015

    if not (min_valid_d <= discount_factor <= max_valid_d):
        audit.status = "REJECTED"
        audit.rejection_reason = (
            f"DISCOUNT_BOUND_VIOLATION: D={discount_factor:.5f} not in"
            f" [{min_valid_d:.5f}, {max_valid_d:.5f}]"
        )
        audit_logger.error(
            f"[{audit.expiry}] {audit.rejection_reason} (R^2={r_squared:.4f})"
        )
        return None

    forward = beta_0 / discount_factor
    audit.inferred_forward = float(forward)

    # % diff should be ~(r-q)*tau as F ~= S(1 + (r-q)*tau). We don't expect r-q to exceed +-10%
    drift_buffer = 0.005
    max_allowed_pct_diff = 0.10 * tau + drift_buffer
    pct_diff = abs(forward - spot_ref) / spot_ref

    if pct_diff > max_allowed_pct_diff:
        audit.status = "REJECTED"
        audit.rejection_reason = (
            f"FORWARD_BOUND_VIOLATION: F={forward:.2f} vs Spot={spot_ref:.2f} (diff"
            f" {pct_diff * 100:.2f}% > limit {max_allowed_pct_diff * 100:.2f}%)"
        )
        audit_logger.error(f"[{audit.expiry}] {audit.rejection_reason}")
        return None

    if r_squared < 0.98:
        audit.status = "REJECTED"
        audit.rejection_reason = (
            f"PARITY_FIT_POOR: R^2={r_squared:.4f} < 0.98 (Quotes noisy or"
            " asynchronous)"
        )
        audit_logger.error(f"[{audit.expiry}] {audit.rejection_reason}")
        return None

    return forward, discount_factor


# =====================================================================
# MAIN PHASE 1 SLICE PROCESSOR
# =====================================================================
def process_snapshot_slice(
        snapshot_df: pd.DataFrame, expiry: str
) -> tuple[ExpirySlice | None, SliceAuditReport]:
    """Processes, cleans, and validates a maturity slice with full audit logging."""
    df_exp = snapshot_df[snapshot_df["expiry"].astype(str) == str(expiry)].copy()
    timestamp_str = (
        df_exp["timestamp"].iloc[0] if not df_exp.empty else "UNKNOWN"
    )

    audit = SliceAuditReport(
        timestamp=timestamp_str,
        expiry=str(expiry),
        tau_minutes=0.0,
        raw_quotes_count=len(df_exp),
    )

    if df_exp.empty:
        audit.status = "REJECTED"
        audit.rejection_reason = "NO_DATA_FOR_EXPIRY"
        audit_logger.warning(f"[{expiry}] No rows present in snapshot.")
        return None, audit

    tau = compute_tau(timestamp_str, str(expiry), audit)
    if tau is None:
        return None, audit

    spot_ref = df_exp["spot_price"].iloc[0]
    if spot_ref is None or math.isnan(spot_ref) or spot_ref <= 0:
        audit.status = "REJECTED"
        audit.rejection_reason = f"INVALID_SPOT_REFERENCE: {spot_ref}"
        audit_logger.error(f"[{expiry}] Invalid underlying spot: {spot_ref}")
        return None, audit

    initial_count = len(df_exp)

    # 1. Check and log duplicate (strike, right) rows
    dup_mask = df_exp.duplicated(subset=["strike", "right"], keep="first")
    for _, row in df_exp[dup_mask].iterrows():
        audit.rejected_strikes_detail.append({
            "strike": row["strike"],
            "right": row["right"],
            "bid": row["bid"],
            "ask": row["ask"],
            "issue": "DUPLICATE_STRIKE_QUOTE",
        })
    audit.dropped_duplicate_quotes = int(dup_mask.sum())
    if audit.dropped_duplicate_quotes > 0:
        audit_logger.warning(
            f"[{expiry}] Dropped {audit.dropped_duplicate_quotes} duplicate"
            " (strike, right) rows."
        )
    df_clean = df_exp[~dup_mask].copy()

    # 2. Check non-positive bids (including -1.0 sentinel)
    zero_bid_mask = (df_clean["bid"].isna()) | (df_clean["bid"] <= 0)
    for _, row in df_clean[zero_bid_mask].iterrows():
        audit.rejected_strikes_detail.append({
            "strike": row["strike"],
            "right": row["right"],
            "bid": row["bid"],
            "ask": row["ask"],
            "issue": "NON_POSITIVE_BID",
        })
    audit.dropped_zero_or_negative_bids = int(zero_bid_mask.sum())
    df_clean = df_clean[~zero_bid_mask].copy()

    # 3. Check missing or non-positive asks separately from crossed markets
    bad_ask_mask = (df_clean["ask"].isna()) | (df_clean["ask"] <= 0)
    for _, row in df_clean[bad_ask_mask].iterrows():
        audit.rejected_strikes_detail.append({
            "strike": row["strike"],
            "right": row["right"],
            "bid": row["bid"],
            "ask": row["ask"],
            "issue": "MISSING_OR_NON_POSITIVE_ASK",
        })
    audit.dropped_missing_or_negative_asks = int(bad_ask_mask.sum())
    df_clean = df_clean[~bad_ask_mask].copy()

    # 4. Check crossed markets (bid >= ask)
    crossed_mask = df_clean["bid"] >= df_clean["ask"]
    for _, row in df_clean[crossed_mask].iterrows():
        audit.rejected_strikes_detail.append({
            "strike": row["strike"],
            "right": row["right"],
            "bid": row["bid"],
            "ask": row["ask"],
            "issue": "CROSSED_MARKET",
        })
    audit.dropped_crossed_quotes = int(crossed_mask.sum())
    df_clean = df_clean[~crossed_mask].copy()

    # 5. Mid-quote and wide-spread filter
    df_clean["mid"] = 0.5 * (df_clean["bid"] + df_clean["ask"])
    df_clean["spread"] = df_clean["ask"] - df_clean["bid"]

    wide_spread_mask = (df_clean["mid"] > 1.0) & (
            df_clean["spread"] > 0.80 * df_clean["mid"]
    )
    for _, row in df_clean[wide_spread_mask].iterrows():
        audit.rejected_strikes_detail.append({
            "strike": row["strike"],
            "right": row["right"],
            "bid": row["bid"],
            "ask": row["ask"],
            "issue": "WIDE_SPREAD_OUTLIER",
        })
    audit.dropped_wide_spreads = int(wide_spread_mask.sum())
    df_clean = df_clean[~wide_spread_mask].copy()

    audit.cleaned_quotes_count = len(df_clean)

    total_dropped = initial_count - audit.cleaned_quotes_count
    if total_dropped > 0:
        audit_logger.info(
            f"[{expiry}] Sanitized quotes: {initial_count} raw ->"
            f" {audit.cleaned_quotes_count} valid. Dropped {total_dropped} issues"
            f" (Dups={audit.dropped_duplicate_quotes},"
            f" ZeroBids={audit.dropped_zero_or_negative_bids},"
            f" BadAsks={audit.dropped_missing_or_negative_asks},"
            f" Crossed={audit.dropped_crossed_quotes},"
            f" WideSpreads={audit.dropped_wide_spreads})."
        )

    calls = df_clean[df_clean["right"] == "C"]
    puts = df_clean[df_clean["right"] == "P"]

    calib_res = extract_forward_and_discount(calls, puts, spot_ref, tau, audit)
    if calib_res is None:
        return None, audit

    forward, discount_factor = calib_res

    # Partition into Liquid OTM Contracts
    otm_calls = calls[calls["strike"] >= forward].sort_values("strike").copy()
    otm_calls["is_call"] = True

    otm_puts = puts[puts["strike"] < forward].sort_values("strike").copy()
    otm_puts["is_call"] = False

    # Drop minimum-tick wing quotes (bid <= 0.05 on SPX causes flat-price variance kinks)
    min_tick_threshold = 0.05
    for sub_df in [otm_puts, otm_calls]:
        min_tick_mask = sub_df["bid"] <= min_tick_threshold
        for _, row in sub_df[min_tick_mask].iterrows():
            audit.rejected_strikes_detail.append({
                "strike": row["strike"],
                "right": row["right"],
                "bid": row["bid"],
                "ask": row["ask"],
                "issue": "MIN_TICK_WING_QUOTE",
            })
        audit.dropped_min_tick_quotes += int(min_tick_mask.sum())

    otm_puts = otm_puts[otm_puts["bid"] > min_tick_threshold].copy()
    otm_calls = otm_calls[otm_calls["bid"] > min_tick_threshold].copy()

    # Enforce Vertical Spread Monotonicity (No quote-level static arbitrage)
    # OTM Puts (K < F): mid-price must strictly increase as strike increases toward F
    valid_put_indices = []
    # Traverse puts from the highest strike (closest to ATM, most liquid) down to the lowest strike
    max_allowed_put_mid = float("inf")
    for idx, row in otm_puts.iloc[::-1].iterrows():
        if row["mid"] < max_allowed_put_mid:
            valid_put_indices.append(idx)
            max_allowed_put_mid = row["mid"]
        else:
            audit.dropped_non_monotonic_quotes += 1
            audit.rejected_strikes_detail.append({
                "strike": row["strike"],
                "right": "P",
                "bid": row["bid"],
                "ask": row["ask"],
                "issue": "NON_MONOTONIC_PUT_MID",
            })
    otm_puts = otm_puts.loc[reversed(valid_put_indices)].copy()

    # OTM Calls (K >= F): mid-price must strictly decrease as strike increases away from F
    valid_call_indices = []
    max_allowed_call_mid = float("inf")
    for idx, row in otm_calls.iterrows():
        if row["mid"] < max_allowed_call_mid:
            valid_call_indices.append(idx)
            max_allowed_call_mid = row["mid"]
        else:
            audit.dropped_non_monotonic_quotes += 1
            audit.rejected_strikes_detail.append({
                "strike": row["strike"],
                "right": "C",
                "bid": row["bid"],
                "ask": row["ask"],
                "issue": "NON_MONOTONIC_CALL_MID",
            })
    otm_calls = otm_calls.loc[valid_call_indices].copy()

    if audit.dropped_min_tick_quotes > 0 or audit.dropped_non_monotonic_quotes > 0:
        audit_logger.info(
            f"[{expiry}] OTM Arbitrage Filter: Dropped"
            f" {audit.dropped_min_tick_quotes} min-tick wing quotes and"
            f" {audit.dropped_non_monotonic_quotes} non-monotonic quotes."
        )

    otm_df = pd.concat([otm_puts, otm_calls]).sort_values("strike")

    min_clean_strikes = 5
    if len(otm_df) < min_clean_strikes:
        audit.status = "REJECTED"
        audit.rejection_reason = (
            f"INSUFFICIENT_CLEAN_STRIKES: {len(otm_df)} available, required >="
            f" {min_clean_strikes}"
        )
        audit_logger.error(f"[{expiry}] {audit.rejection_reason}")
        return None, audit

    audit.status = "PASSED"
    audit_logger.info(
        f"[{expiry}] Successfully validated: {len(otm_df)} strikes,"
        f" F={forward:.2f}, D={discount_factor:.5f}, R^2={audit.parity_r2:.4f}"
    )

    slice_obj = ExpirySlice(
        expiry=str(expiry),
        tau=tau,
        forward=forward,
        discount_factor=discount_factor,
        strikes=otm_df["strike"].values,
        otm_prices=otm_df["mid"].values,
        is_call=otm_df["is_call"].values,
        bids=otm_df["bid"].values,
        asks=otm_df["ask"].values,
        audit=audit,
    )

    return slice_obj, audit


# =====================================================================
# BLACK-76 ANALYTICAL PRICER & VEGA
# =====================================================================
SQRT_2PI = math.sqrt(2.0 * math.pi)


def black76_price(
        F: float,
        K: float,
        tau: float,
        sigma: float,
        discount_factor: float,
        is_call: bool,
) -> float:
    """Computes European option price under Black-76 forward model."""
    if sigma <= 0.0 or tau <= 0.0 or F <= 0.0 or K <= 0.0 or discount_factor <= 0.0:
        raise ValueError(
            f"Invalid Black-76 parameters: F={F}, K={K}, tau={tau}, sigma={sigma},"
            f" D={discount_factor}. All inputs must be strictly positive."
        )

    vol_sqrt_tau = sigma * math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * tau) / vol_sqrt_tau
    d2 = d1 - vol_sqrt_tau

    if is_call:
        return discount_factor * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return discount_factor * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black76_vega(
        F: float, K: float, tau: float, sigma: float, discount_factor: float
) -> float:
    """Computes analytical Black-76 Vega: dPrice / dSigma."""
    if sigma <= 0.0 or tau <= 0.0 or F <= 0.0 or K <= 0.0 or discount_factor <= 0.0:
        raise ValueError(
            f"Invalid Black-76 Vega parameters: F={F}, K={K}, tau={tau},"
            f" sigma={sigma}, D={discount_factor}. All inputs must be strictly"
            " positive."
        )

    vol_sqrt_tau = sigma * math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * tau) / vol_sqrt_tau
    n_d1 = math.exp(-0.5 * d1 * d1) / SQRT_2PI

    return discount_factor * F * math.sqrt(tau) * n_d1


# =====================================================================
# IMPLIED VOLATILITY INVERSION (BRACKET-SAFE NEWTON-RAPHSON + BRENTQ)
# =====================================================================
def invert_iv_newton_raphson(
    price: float,
    F: float,
    K: float,
    tau: float,
    discount_factor: float,
    is_call: bool,
    expiry: str = "UNKNOWN",
    max_iter: int = 25,
    tol: float = 1e-6,
) -> float | None:
    """Inverts European market price to Black-76 Implied Volatility.

    Uses a moneyness-aware initial guess and a bracket-guarded Newton-Raphson
    loop to prevent zero-vega tail underflow and overshoot on OTM wings.
    """
    right_label = "C" if is_call else "P"

    if price <= 0.0 or tau <= 0.0 or F <= 0.0 or K <= 0.0 or discount_factor <= 0.0:
        audit_logger.error(
            f"[{expiry} K={K} {right_label}] IV inversion rejected: non-positive"
            f" input (price={price}, F={F}, tau={tau}, D={discount_factor})."
        )
        return None

    # 1. No-arbitrage lower boundary (intrinsic value >= 0)
    intrinsic = max(0.0, discount_factor * ((F - K) if is_call else (K - F)))
    if price < intrinsic - 1e-4:
        audit_logger.warning(
            f"[{expiry} K={K} {right_label}] IV inversion rejected: price"
            f" {price:.4f} violates intrinsic lower bound {intrinsic:.4f}."
        )
        return None

    # 2. No-arbitrage upper boundary
    max_price = discount_factor * F if is_call else discount_factor * K
    if price >= max_price:
        audit_logger.warning(
            f"[{expiry} K={K} {right_label}] IV inversion rejected: price"
            f" {price:.4f} >= upper bound {max_price:.4f}."
        )
        return None

    # 3. Moneyness-Aware Initial Volatility Guess using ATM Brenner-Subrahmanyam estimate
    abs_x = abs(math.log(F / K))
    norm_price = price / (discount_factor * math.sqrt(F * K))
    sigma_atm = SQRT_2PI / math.sqrt(tau) * (price / (discount_factor * F))

    # OTM Asymptotic Tail estimate (prevents starting in the zero-vega tail)
    if abs_x > 1e-4:
        clamped_np = max(1e-12, min(0.45, norm_price))
        sigma_otm = abs_x / math.sqrt(-2.0 * tau * math.log(clamped_np))
        # Ensure initial guess puts |d1| <= 3.0 so initial Vega is always well-conditioned
        sigma_min_vega = abs_x / (3.0 * math.sqrt(tau))
        sigma_0 = max(sigma_atm, sigma_otm, sigma_min_vega)
    else:
        sigma_0 = sigma_atm

    sigma_low = 0.005
    sigma_high = 4.0
    sigma = max(0.02, min(sigma_0, 2.5))

    # 4. Bracket-Guarded Newton-Raphson Loop (rtsafe hybrid)
    nr_failure_reason = "MAX_ITER_EXCEEDED"
    for _ in range(max_iter):
        p = black76_price(F, K, tau, sigma, discount_factor, is_call)
        diff = p - price

        if abs(diff) < tol:
            return float(sigma)

        # Tighten the monotonic search bracket [sigma_low, sigma_high]
        if diff > 0.0:
            sigma_high = min(sigma_high, sigma)
        else:
            sigma_low = max(sigma_low, sigma)

        v = black76_vega(F, K, tau, sigma, discount_factor)

        if v > 1e-8:
            sigma_next = sigma - (diff / v)
        else:
            sigma_next = -1.0  # Force bisection step below

        # If Newton step jumps outside the known bracket [sigma_low, sigma_high],
        # take a damped bisection step inside the bracket instead of aborting
        if not (sigma_low < sigma_next < sigma_high):
            sigma_next = 0.5 * (sigma_low + sigma_high)

        if abs(sigma_next - sigma) < 1e-8:
            return float(sigma_next)

        sigma = sigma_next

    # 5. Logged Fallback: Brent's Method
    audit_logger.info(
        f"[{expiry} K={K} {right_label}] Newton-Raphson did not converge"
        f" ({nr_failure_reason}); falling back to Brent's method."
    )

    def objective(s: float) -> float:
        return black76_price(F, K, tau, s, discount_factor, is_call) - price

    try:
        sigma_brent = brentq(
            objective, a=sigma_low, b=sigma_high, xtol=tol, maxiter=30
        )
        return float(sigma_brent)
    except Exception as e:
        audit_logger.error(
            f"[{expiry} K={K} {right_label}] IV root-finding failed in both"
            f" Newton-Raphson and Brentq (price={price:.4f}, F={F:.2f}): {e}"
        )
        return None
