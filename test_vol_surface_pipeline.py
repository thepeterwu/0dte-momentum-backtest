"""test_pipeline.py

Verification script for Phase 1 through Phase 4.
Selects an active post-09:45 ET market snapshot to verify the calibration pipeline.
"""

import math
import os
import numpy as np
import pandas as pd

from vol_surface.black76 import black76_price, process_snapshot_slice
from vol_surface.smile import fit_smile_slice


def generate_mock_snapshot() -> pd.DataFrame:
    print("\n[INFO] Generating synthetic SPX option snapshot for testing...")
    spot = 5800.0
    tau = 3.0 / 365.0
    rate = 0.045
    div = 0.015
    fwd = spot * math.exp((rate - div) * tau)
    discount = math.exp(-rate * tau)

    strikes = np.arange(5500, 6125, 25)
    records = []

    for k in strikes:
        x = math.log(k / fwd)
        skew_vol = 0.16 - 0.25 * x + 0.15 * (x ** 2)
        call_p = black76_price(fwd, k, tau, skew_vol, discount, is_call=True)
        put_p = black76_price(fwd, k, tau, skew_vol, discount, is_call=False)

        records.append({
            "timestamp": "2026-10-20 09:46:00",
            "underlying_symbol": "SPX",
            "spot_price": spot,
            "expiry": "20261021",
            "strike": float(k),
            "right": "C",
            "bid": max(0.05, round(call_p - 0.20, 2)),
            "ask": round(call_p + 0.20, 2),
            "last": round(call_p, 2),
            "volume": 100,
        })
        records.append({
            "timestamp": "2026-10-20 09:46:00",
            "underlying_symbol": "SPX",
            "spot_price": spot,
            "expiry": "20261021",
            "strike": float(k),
            "right": "P",
            "bid": max(0.05, round(put_p - 0.20, 2)),
            "ask": round(put_p + 0.20, 2),
            "last": round(put_p, 2),
            "volume": 100,
        })

    return pd.DataFrame(records)


def main():
    csv_file = "spx_options_surface.csv"

    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 100:
        print(f"[INFO] Found market data file '{csv_file}'. Loading...")
        raw_df = pd.read_csv(csv_file)

        # Filter strictly for post-09:45 timestamps (after the 15-minute delay catches up)
        ts_col = raw_df["timestamp"].astype(str)
        # Extract HH:MM
        time_part = ts_col.str.slice(11, 16)
        active_rth_df = raw_df[time_part >= "09:46"]

        if not active_rth_df.empty:
            # Pick the timestamp with the highest number of non-zero bids
            bid_counts = (
                active_rth_df[active_rth_df["bid"] > 0]
                .groupby("timestamp")["strike"]
                .count()
            )
            chosen_ts = bid_counts.idxmax()
        else:
            # Fallback to the last timestamp in the CSV
            chosen_ts = raw_df["timestamp"].iloc[-1]

        snapshot_df = raw_df[raw_df["timestamp"] == chosen_ts].copy()
        print(
            f"[INFO] Evaluating active snapshot timestamp: {chosen_ts} ("
            f"{len(snapshot_df)} rows)"
        )
    else:
        snapshot_df = generate_mock_snapshot()

    # Normalize expiry values to strings
    expiries = sorted(snapshot_df["expiry"].astype(str).unique())
    print(f"[INFO] Found {len(expiries)} expiries to test: {expiries}")

    for exp in expiries:
        print("\n" + "=" * 75)
        print(f" TESTING MATURITY SLICE: {exp} ")
        print("=" * 75)

        # Phase 1: Quote sanitization & Forward extraction
        slice_obj, slice_audit = process_snapshot_slice(snapshot_df, str(exp))

        if slice_obj is None:
            print(f"[REJECTED] Slice {exp} failed validation!")
            print(f"           Reason: {slice_audit.rejection_reason}")
            print(f"           Raw Quotes: {slice_audit.raw_quotes_count}")
            print(f"           Clean Quotes: {slice_audit.cleaned_quotes_count}")
            continue

        print(
            f"[PHASE 1 OK] Tau: {slice_obj.tau:.5f} ({slice_audit.tau_minutes:.1f}m)"
        )
        print(f"             Forward F: {slice_obj.forward:.2f}")
        print(
            f"             Discount Factor D: {slice_obj.discount_factor:.5f} (R^2"
            f" = {slice_audit.parity_r2:.4f})"
        )
        print(f"             Clean OTM Strikes: {len(slice_obj.strikes)}")

        # Phase 2 & 3: Total Variance Spline w(x)
        smile, smile_audit = fit_smile_slice(slice_obj)

        if smile is None:
            print(f"[REJECTED] Smile fitting failed for {exp}!")
            print(f"           Reason: {smile_audit.rejection_reason}")
            continue

        print(
            f"[PHASE 2 & 3 OK] Spline Fitted: {len(smile.x_market)} knots in"
            " total variance w(x)"
        )
        print(
            f"                 Left Wing Slope: {smile.slope_left:.4f} (Clamped:"
            f" {smile_audit.left_wing_clamped})"
        )
        print(
            f"                 Right Wing Slope: {smile.slope_right:.4f} (Clamped:"
            f" {smile_audit.right_wing_clamped})"
        )
        print(
            f"                 Gatheral Min Density g(x):"
            f" {smile_audit.min_g_density:.6f}"
        )
        print(
            f"                 Butterfly Arbitrage-Free:"
            f" {smile_audit.is_butterfly_arbitrage_free}"
        )

        # Phase 4: Fixed-Point Delta Inversion
        target_pillars = [-0.10, -0.25, -0.50, 0.25, 0.10]
        grid_results = smile.evaluate_delta_grid(target_pillars)

        if grid_results is None:
            print(f"[REJECTED] Phase 4 Delta Inversion failed for {exp}!")
            continue

        print("\n[PHASE 4 OK] Calibrated Delta-Grid Summary:")
        print("-" * 75)
        print(
            f"{'Target Delta':^14} | {'Strike':^10} | {'Moneyness x':^12} | {'Total Var w':^12} | {'IV (%)':^8} | {'Round-Trip Err':^14}"
        )
        print("-" * 75)

        for d, k, x, w, iv in zip(
                grid_results["delta_pillars"],
                grid_results["strikes"],
                grid_results["x"],
                grid_results["total_variance"],
                grid_results["iv"],
        ):
            achieved_delta = smile.get_forward_delta(x, is_call=(d > 0))
            delta_err = abs(achieved_delta - d)
            print(
                f"{d:^+14.2f} | {k:^10.2f} | {x:^+12.4f} | {w:^12.5f} |"
                f" {iv * 100:^8.2f} | {delta_err:^14.2e}"
            )

        print("-" * 75)


if __name__ == "__main__":
    main()
