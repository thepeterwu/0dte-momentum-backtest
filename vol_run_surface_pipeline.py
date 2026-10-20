# run_surface_pipeline.py
import pandas as pd
from vol_surface.surface import VolSurface

# 1. Load a single 1-minute snapshot from your CSV
df = pd.read_csv("data/spx_options_surface.csv")
snapshot_df = df[df["timestamp"] == "2026-10-20 10:00:00"]

# 2. Fit the complete 2D surface across all expiries
surface = VolSurface.from_snapshot(
    snapshot_df,
    delta_grid=[-0.10, -0.25, -0.50, 0.25, 0.10],
    strike_step=25,
)

# 3. Query IV for any arbitrary (delta, tau) point
target_iv = surface.get_iv(delta=-0.25, tau=2 / 365)
print(f"Interpolated 25-Delta Put IV (2-DTE): {target_iv * 100:.2f}%")

# 4. Explicit Arbitrage Audit
arb_report = surface.audit_arbitrage()
print(
    f"Arbitrage Free: {arb_report.is_arbitrage_free} | Violations:"
    f" {arb_report.violation_count}"
)

# 5. Visual Diagnostics
surface.plot_slice(expiry="20261022")  # 2D Smile with Bid/Ask bars
surface.plot_3d()  # 3D Interactive Vol Surface