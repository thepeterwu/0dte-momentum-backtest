"""vol_run_surface_pipeline.py

End-to-end execution script for the SPX Implied Volatility Surface pipeline.
Supports:
  1. Single-minute surface calibration, arbitrary (delta, tau) interpolation, & 3D/2D plotting.
  2. Multi-minute dynamic time-slider animation to scrub through intraday surface evolution.
"""

import argparse
import os
import pandas as pd
from vol_surface.surface import DEFAULT_DELTA_PILLARS, VolSurface
from vol_surface.visualization import (
    plot_dynamic_surface_slider,
    plot_single_surface_dashboard,
    plot_slice_diagnostics,
)


# =====================================================================
# USER CONFIGURATION (Edit directly for PyCharm one-click Run)
# =====================================================================

CSV_FILE_PATH = "spx_options_surface.csv"

# 1. Single-Minute Inspection Mode:
# Specify "HH:MM" (e.g. "10:46") or full timestamp "YYYY-MM-DD HH:MM:SS"
TARGET_MINUTE = "11:02"
SHOW_SLICE_DIAGNOSTICS = True

# 2. Dynamic Time-Slider Animation Mode:
RUN_DYNAMIC_SLIDER = True
SLIDER_START_TIME = "09:46"   # First post-delay active RTH minute
SLIDER_END_TIME = "16:15"     # End of time window
SLIDER_STEP_MINUTES = 1       # Sample every Nth minute (1 = every minute, 5 = every 5m)


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def find_matching_timestamp(df: pd.DataFrame, target_str: str) -> str:
    """Finds the exact or closest matching active timestamp in the CSV."""
    all_ts = sorted(df["timestamp"].astype(str).unique())

    # Exact match
    if target_str in all_ts:
        return target_str

    # Substring match (e.g. "10:46" matches "2026-10-20 10:46:00")
    matches = [ts for ts in all_ts if target_str in ts]
    if matches:
        return matches[0]

    # Fallback: find closest active timestamp after 09:45
    active_df = df[df["bid"].notna() & (df["bid"] > 0)]
    active_ts = sorted(active_df["timestamp"].astype(str).unique())
    rth_ts = [ts for ts in active_ts if ts[11:16] >= "09:46"]

    chosen = rth_ts[0] if rth_ts else all_ts[-1]
    print(
        f"[WARNING] Target minute '{target_str}' not found in CSV. "
        f"Selected active timestamp '{chosen}' instead."
    )
    return chosen


def run_single_minute_analysis(
    raw_df: pd.DataFrame,
    target_minute: str,
    delta_pillars: list[float] = DEFAULT_DELTA_PILLARS,
    show_diagnostics: bool = True,
) -> VolSurface | None:
    """Calibrates, prints, interpolates, and plots the VolSurface for a single minute."""
    chosen_ts = find_matching_timestamp(raw_df, target_minute)
    snapshot_df = raw_df[raw_df["timestamp"].astype(str) == chosen_ts].copy()

    print("\n" + "=" * 82)
    print(f" CALIBRATING SINGLE-MINUTE VOLATILITY SURFACE: {chosen_ts} ")
    print("=" * 82)

    surface, audit = VolSurface.from_snapshot(snapshot_df, delta_pillars=delta_pillars)

    if surface is None:
        print(f"[ERROR] Surface calibration failed at {chosen_ts}: {audit.rejection_reason}")
        return None

    # 1. Print Calibrated 2D Implied Volatility Grid (in %)
    df_grid = surface.to_dataframe()
    delta_cols = [c for c in df_grid.columns if c.startswith("Delta_")]
    df_display = df_grid.copy()
    df_display["DTE_Days"] = (df_display["tau"] * 365.0).round(2)
    df_display["forward"] = df_display["forward"].round(2)
    df_display["discount"] = df_display["discount"].round(5)
    for c in delta_cols:
        df_display[c] = (df_display[c] * 100.0).round(2)

    cols_order = ["DTE_Days", "forward", "discount"] + delta_cols
    print("\n[CALIBRATED 2D IMPLIED VOLATILITY GRID (%)]")
    print("-" * 82)
    print(df_display[cols_order].to_string())
    print("-" * 82)

    # 2. Demonstrate Arbitrary (Delta, Tau) Surface Interpolation
    print("\n[ARBITRARY (DELTA, TAU) SURFACE INTERPOLATION QUERIES]")
    sample_queries = [
        (-0.25, 2.5 / 365.0, "25-Delta Put  @ 2.5 DTE"),
        (-0.50, 3.5 / 365.0, "50-Delta ATM  @ 3.5 DTE"),
        (+0.20, 5.0 / 365.0, "20-Delta Call @ 5.0 DTE"),
        (-0.15, 6.0 / 365.0, "15-Delta Put  @ 6.0 DTE"),
    ]
    for d_q, tau_q, desc in sample_queries:
        # Clamp query tau inside available surface bounds for demonstration
        tau_clamped = max(float(surface.taus[0]), min(float(surface.taus[-1]), tau_q))
        iv_exact = surface.get_iv(target_delta=d_q, target_tau=tau_clamped, delta_method="exact")
        iv_spline = surface.get_iv(target_delta=d_q, target_tau=tau_clamped, delta_method="grid_spline")
        print(
            f"  {desc:<26} (tau={tau_clamped * 365:4.2f}d) -> "
            f"Exact IV: {iv_exact * 100:6.2f}% | Grid-Spline IV: {iv_spline * 100:6.2f}%"
        )

    # 3. Print Arbitrage Summary Status (Informational)
    print("\n[ARBITRAGE AUDIT SUMMARY (Informational)]")
    print(
        f"  Valid Expiries: {audit.valid_slices_count}/{audit.total_expiries_Attempted} | "
        f"Butterfly Arb-Free: {audit.butterfly_arb_free} (min g={audit.min_surface_g_density:+.4f}) | "
        f"Calendar Delta Arb-Free: {audit.calendar_delta_arb_free} | "
        f"Calendar Moneyness Arb-Free: {audit.calendar_x_arb_free}"
    )

    # 4. Launch Interactive Visuals
    safe_ts = chosen_ts.replace(":", "-").replace(" ", "_")
    plot_single_surface_dashboard(
        surface,
        delta_method="exact",
        show_fig=True,
        save_html=f"vol_surface_{safe_ts}.html",
    )

    if show_diagnostics:
        plot_slice_diagnostics(
            surface,
            show_fig=True,
            save_html=f"vol_diagnostics_{safe_ts}.html",
        )

    return surface


def run_dynamic_slider_analysis(
    raw_df: pd.DataFrame,
    start_time: str = "09:46",
    end_time: str = "16:15",
    step_minutes: int = 1,
    delta_pillars: list[float] = DEFAULT_DELTA_PILLARS,
) -> list[VolSurface]:
    """Calibrates surfaces across a time window and launches the interactive Time-Slider."""
    all_ts = sorted(raw_df["timestamp"].astype(str).unique())

    # Filter timestamps to [start_time, end_time] based on HH:MM
    filtered_ts = []
    for ts in all_ts:
        hh_mm = ts[11:16] if len(ts) >= 16 else ts
        if start_time <= hh_mm <= end_time:
            filtered_ts.append(ts)

    # Apply step_minutes stride
    sampled_ts = filtered_ts[:: max(1, step_minutes)]

    print("\n" + "=" * 82)
    print(
        f" BUILDING DYNAMIC TIME-SLIDER SURFACES ({start_time} to {end_time}, step={step_minutes}m) "
    )
    print(f" Total candidate timestamps to calibrate: {len(sampled_ts)}")
    print("=" * 82)

    surfaces: list[VolSurface] = []
    for ts in sampled_ts:
        snap_df = raw_df[raw_df["timestamp"].astype(str) == ts]
        surf, audit = VolSurface.from_snapshot(snap_df, delta_pillars=delta_pillars)
        if surf is not None:
            surfaces.append(surf)
            print(
                f"  [OK] {ts} | Expiries: {len(surf.slices)} | "
                f"F₀={surf.forwards[0]:.2f} | "
                f"Front 50Δ IV={surf.iv_grid[0, 2] * 100:.2f}%"
            )
        else:
            print(f"  [SKIP] {ts} | Reason: {audit.rejection_reason}")

    if not surfaces:
        print("[ERROR] No valid surfaces could be calibrated in the requested time window.")
        return []

    print(f"\n[INFO] Successfully calibrated {len(surfaces)} surfaces. Launching time slider...")
    plot_dynamic_surface_slider(
        surfaces,
        n_delta=25,
        n_tau=25,
        delta_method="grid_spline",
        show_fig=True,
        save_html="vol_surface_time_slider.html",
    )

    return surfaces


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="SPX Volatility Surface Pipeline Runner")
    parser.add_argument("--csv", type=str, default=CSV_FILE_PATH, help="Path to options CSV")
    parser.add_argument(
        "--minute",
        type=str,
        default=TARGET_MINUTE,
        help="Target minute to plot (e.g., '10:46' or '2026-10-20 10:46:00')",
    )
    parser.add_argument(
        "--slider",
        action="store_true",
        default=RUN_DYNAMIC_SLIDER,
        help="Build and launch the dynamic multi-minute time slider",
    )
    parser.add_argument("--start", type=str, default=SLIDER_START_TIME, help="Slider start HH:MM")
    parser.add_argument("--end", type=str, default=SLIDER_END_TIME, help="Slider end HH:MM")
    parser.add_argument("--step", type=int, default=SLIDER_STEP_MINUTES, help="Minute step size")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Could not find market data file: '{args.csv}'")

    print(f"[INFO] Loading market data from '{args.csv}'...")
    raw_df = pd.read_csv(args.csv)
    print(
        f"[INFO] Loaded {len(raw_df):,} rows across "
        f"{raw_df['timestamp'].nunique()} distinct minute snapshots."
    )

    # 1. Run Single-Minute Surface Calibration & Dashboard
    if args.minute:
        run_single_minute_analysis(
            raw_df=raw_df,
            target_minute=args.minute,
            show_diagnostics=SHOW_SLICE_DIAGNOSTICS,
        )

    # 2. Run Multi-Minute Dynamic Time-Slider Animation
    if args.slider:
        run_dynamic_slider_analysis(
            raw_df=raw_df,
            start_time=args.start,
            end_time=args.end,
            step_minutes=args.step,
        )


if __name__ == "__main__":
    main()