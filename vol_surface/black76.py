from dataclasses import dataclass, field
from datetime import datetime
import logging
import math
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

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
    dropped_zero_or_negative_bids: int = 0
    dropped_wide_spreads: int = 0
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
        audit: "SliceAuditReport",
) -> tuple[float, float] | None:
    """Calibrates put-call parity without unverified fallbacks.

  Rejects slice if economic bounds are breached.
  """

    # Ensure 1-to-1 pairing per strike
    c_unique = calls_df.drop_duplicates(subset=["strike"])
    p_unique = puts_df.drop_duplicates(subset=["strike"])

    pairs = pd.merge(
        c_unique[["strike", "mid"]],
        p_unique[["strike", "mid"]],
        on="strike",
        suffixes=("_call", "_put"),
    )
    audit.ntm_pairs_found = len(pairs)

    # Require at least 3 distinct NTM matched pairs for reliable regression
    if len(pairs) < 3:
        audit.status = "REJECTED"
        audit.rejection_reason = (
            f"INSUFFICIENT_PARITY_PAIRS: Found {len(pairs)}, required >= 3"
        )
        audit_logger.error(
            f"[{audit.expiry}] {audit.rejection_reason} at spot {spot_ref:.2f}"
        )
        return None

    # Restrict regression to near-the-money (+/- 2% around spot)
    ntm_pairs = pairs[
        (pairs["strike"] >= spot_ref * 0.98) & (pairs["strike"] <= spot_ref * 1.02)
        ]
    if len(ntm_pairs) < 3:
        ntm_pairs = pairs  # Fall back to all pairs if local NTM density is thin

    K = ntm_pairs["strike"].values
    y = (ntm_pairs["mid_call"] - ntm_pairs["mid_put"]).values

    # Ordinary Least Squares: y = beta_0 + beta_1 * K
    # Put-Call Parity: C - P = D*F - D*K => beta_1 = -D, beta_0 = D*F
    poly = np.polyfit(K, y, deg=1)
    beta_1, beta_0 = poly[0], poly[1]

    # Compute R^2 of parity fit
    y_pred = np.polyval(poly, K)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    audit.parity_r2 = float(r_squared)

    discount_factor = -beta_1
    audit.inferred_discount = float(discount_factor)

    # Strict validation of Discount Factor (implied risk-free rate)
    # Rate range [-2%, +12%] plus a 1.5% tick-discretization margin for short tenors
    # to protect against noise causing DISCOUNT_BOUND_VIOLATION
    min_valid_d = math.exp(-0.12 * tau) - 0.015
    max_valid_d = math.exp(0.02 * tau) + 0.015

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

    # Strict validation of Forward Price
    # Forward should not deviate from spot by more than a reasonable carry range for tau
    # Max theoretical drift over short tenors: +/- (10% * tau + 0.5% buffer)
    max_allowed_pct_diff = 0.10 * tau + 0.0075
    pct_diff = abs(forward - spot_ref) / spot_ref

    if pct_diff > max_allowed_pct_diff:
        audit.status = "REJECTED"
        audit.rejection_reason = (
            f"FORWARD_BOUND_VIOLATION: F={forward:.2f} vs Spot={spot_ref:.2f} (diff"
            f" {pct_diff * 100:.2f}% > limit {max_allowed_pct_diff * 100:.2f}%)"
        )
        audit_logger.error(f"[{audit.expiry}] {audit.rejection_reason}")
        return None

    # Parity Linearity Check
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
    """Processes, cleans, and validates a maturity slice.

  Logs every data anomaly and drops the slice if assumptions are violated.
  """
    df_exp = snapshot_df[snapshot_df["expiry"] == expiry].copy()
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

    # Compute time-to-maturity
    tau = compute_tau(timestamp_str, str(expiry), audit)
    if tau is None:
        return None, audit

    spot_ref = df_exp["spot_price"].iloc[0]
    if spot_ref is None or math.isnan(spot_ref) or spot_ref <= 0:
        audit.status = "REJECTED"
        audit.rejection_reason = f"INVALID_SPOT_REFERENCE: {spot_ref}"
        audit_logger.error(f"[{expiry}] Invalid underlying spot: {spot_ref}")
        return None, audit

    # Detailed Quote Sanitization & Issue Tracking
    initial_count = len(df_exp)

    # Check non-positive bids (including -1.0 sentinel)
    zero_bid_mask = (df_exp["bid"].isna()) | (df_exp["bid"] <= 0)
    for _, row in df_exp[zero_bid_mask].iterrows():
        audit.rejected_strikes_detail.append({
            "strike": row["strike"],
            "right": row["right"],
            "bid": row["bid"],
            "ask": row["ask"],
            "issue": "NON_POSITIVE_BID",
        })
    audit.dropped_zero_or_negative_bids = int(zero_bid_mask.sum())
    df_clean = df_exp[~zero_bid_mask].copy()

    # Check crossed markets (bid >= ask)
    crossed_mask = (df_clean["ask"].isna()) | (
            df_clean["bid"] >= df_clean["ask"]
    )
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

    # Mid-quote and wide-spread filter
    df_clean["mid"] = 0.5 * (df_clean["bid"] + df_clean["ask"])
    df_clean["spread"] = df_clean["ask"] - df_clean["bid"]

    # Exclude extreme wide spreads (e.g., spread > 0.8 * mid for mid > 1.0)
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

    # Log cleaning summary
    total_dropped = (
            audit.dropped_zero_or_negative_bids
            + audit.dropped_crossed_quotes
            + audit.dropped_wide_spreads
    )
    if total_dropped > 0:
        audit_logger.info(
            f"[{expiry}] Sanitized quotes: {initial_count} raw ->"
            f" {audit.cleaned_quotes_count} valid. Dropped {total_dropped} issues"
            f" (ZeroBids={audit.dropped_zero_or_negative_bids},"
            f" Crossed={audit.dropped_crossed_quotes},"
            f" WideSpreads={audit.dropped_wide_spreads})."
        )

    # Calibrate Forward & Discount via Put-Call Parity
    calls = df_clean[df_clean["right"] == "C"]
    puts = df_clean[df_clean["right"] == "P"]

    calib_res = extract_forward_and_discount(calls, puts, spot_ref, tau, audit)
    if calib_res is None:
        # Slice is rejected because economic parameters failed validation
        return None, audit

    forward, discount_factor = calib_res

    # Partition into Liquid OTM Contracts
    otm_calls = calls[calls["strike"] >= forward].copy()
    otm_calls["is_call"] = True

    otm_puts = puts[puts["strike"] < forward].copy()
    otm_puts["is_call"] = False

    otm_df = (
        pd.concat([otm_puts, otm_calls])
        .sort_values("strike")
        .drop_duplicates(subset=["strike"])
    )

    # Require a critical mass of strikes to form a smile
    if len(otm_df) < 5:
        audit.status = "REJECTED"
        audit.rejection_reason = (
            f"INSUFFICIENT_CLEAN_STRIKES: {len(otm_df)} available, required >= 5"
        )
        audit_logger.error(f"[{expiry}] {audit.rejection_reason}")
        return None, audit

    audit.status = "PASSED"
    audit_logger.info(
        f"[{expiry}] Successfully validated: {len(otm_df)} strikes, F={forward:.2f},"
        f" D={discount_factor:.5f}, R^2={audit.parity_r2:.4f}"
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
