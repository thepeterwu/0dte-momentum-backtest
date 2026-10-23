"""visualisation.py

Interactive 2D and 3D Plotly visualizers for the SPX Implied Volatility Surface,
including a dynamic minute-by-minute time-slider animation.
"""

import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from vol_surface.surface import VolSurface
except ModuleNotFoundError:
    from surface import VolSurface


def _u_to_signed_delta(u_val: float) -> float:
    """Converts continuous grid coordinate u = 1 - CallDelta in (0, 1) to signed OTM trader delta.

    - u = 0.10 -> -0.10 (10d Put)
    - u = 0.25 -> -0.25 (25d Put)
    - u = 0.50 -> -0.50 (50d ATM)
    - u = 0.75 -> +0.25 (25d Call)
    - u = 0.90 -> +0.10 (10d Call)
    """
    return -u_val if u_val <= 0.50 else (1.0 - u_val)


def evaluate_dense_surface_mesh(
    surface: VolSurface,
    n_delta: int = 25,
    n_tau: int = 25,
    delta_method: str = "grid_spline",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates a dense (n_tau x n_delta) mesh of Implied Volatility (%) for 3D plotting."""
    u_min = float(1.0 - surface.call_deltas[0])
    u_max = float(1.0 - surface.call_deltas[-1])

    u_Axis = np.linspace(u_min, u_max, n_delta)
    tau_axis = np.linspace(float(surface.taus[0]), float(surface.taus[-1]), n_tau)
    dte_axis = tau_axis * 365.0

    iv_mesh_pct = np.zeros((n_tau, n_delta), dtype=np.float64)

    for i, tau_val in enumerate(tau_axis):
        for j, u_val in enumerate(u_Axis):
            signed_d = _u_to_signed_delta(float(u_val))
            iv_val = surface.get_iv(
                target_delta=signed_d,
                target_tau=float(tau_val),
                delta_method=delta_method,
                allow_extrapolation=True,
            )
            iv_mesh_pct[i, j] = iv_val * 100.0

    return u_Axis, dte_axis, iv_mesh_pct


# =====================================================================
# 1. SINGLE-MINUTE 3D + 2D INTERACTIVE DASHBOARD
# =====================================================================
def plot_single_surface_dashboard(
    surface: VolSurface,
    delta_method: str = "exact",
    show_fig: bool = True,
    save_html: str | None = None,
) -> go.Figure:
    """Renders a 2-panel dashboard for a specific minute snapshot:

    Left: 3D Implied Volatility Surface with calibrated pillar knots.
    Right: 2D Implied Volatility Smiles in log-moneyness x = ln(K/F) vs Market IVs.
    """
    u_axis, dte_axis, iv_mesh_pct = evaluate_dense_surface_mesh(
        surface, n_delta=25, n_tau=25, delta_method=delta_method
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        column_widths=[0.55, 0.45],
        subplot_titles=(
            f"3D Volatility Surface — {surface.timestamp}",
            "Cross-Sectional Smiles (Spline Curve vs. Market Quotes)",
        ),
        horizontal_spacing=0.06,
    )

    # 1A. Dense 3D Surface Mesh
    fig.add_trace(
        go.Surface(
            x=u_axis,
            y=dte_axis,
            z=iv_mesh_pct,
            colorscale="Viridis",
            opacity=0.92,
            name="Fitted Surface",
            colorbar=dict(
                title="IV (%)",
                len=0.75,
                thickness=14,
                x=0.48,
            ),
            hovertemplate=(
                "Delta Axis: %{x:.2f}<br>"
                "Maturity: %{y:.2f} DTE<br>"
                "Implied Vol: %{z:.2f}%<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # 1B. Overlay exact calibrated (Delta_j, Tau_i) anchor pillars in 3D
    u_pillars = 1.0 - surface.call_deltas
    u_grid_2d, dte_grid_2d = np.meshgrid(u_pillars, surface.taus * 365.0)

    fig.add_trace(
        go.Scatter3d(
            x=u_grid_2d.flatten(),
            y=dte_grid_2d.flatten(),
            z=(surface.iv_grid * 100.0).flatten(),
            mode="markers",
            marker=dict(size=4, color="#ffffff", line=dict(color="#000000", width=1)),
            name="Calibrated Pillars",
            hovertemplate=(
                "Pillar u=%{x:.2f}<br>DTE=%{y:.2f}d<br>IV=%{z:.2f}%<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # 2. Right Panel: 2D Smiles across Expiries (Market Dots + Continuous Spline)
    palette = ["#00e5ff", "#00e676", "#ffea00", "#ff9100", "#ff5252", "#d500f9", "#2979ff"]

    for idx, s in enumerate(surface.slices):
        color = palette[idx % len(palette)]
        dte = s.tau * 365.0
        label = f"{s.expiry} ({dte:.1f}d)"

        # Map market strikes to u = 1 - CallDelta = |PutDelta|
        u_mkt = 1.0 - np.asarray(
            s.get_forward_delta(s.x_market, is_call=True), dtype=np.float64
        )

        # Market IV dots
        fig.add_trace(
            go.Scatter(
                x=u_mkt,
                y=s.iv_market * 100.0,
                mode="markers",
                marker=dict(size=6, color=color, symbol="circle-open"),
                name=f"{label} Mkt",
                legendgroup=s.expiry,
                showlegend=False,
                customdata=u_mkt - 1.0,
                hovertemplate=(
                    f"<b>{s.expiry} Market</b><br>"
                    "Put Δ: %{customdata:+.3f}<br>"
                    "Market IV: %{y:.2f}%<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

        # Fitted Continuous Spline Curve mapped to Delta
        x_dense = np.linspace(s.x_left - 0.015, s.x_right + 0.015, 160)
        u_dense = 1.0 - np.asarray(
            s.get_forward_delta(x_dense, is_call=True), dtype=np.float64
        )
        mask = (u_dense >= 0.02) & (u_dense <= 0.98)
        iv_dense_pct = np.asarray(s.get_iv(x_dense[mask]), dtype=np.float64) * 100.0

        fig.add_trace(
            go.Scatter(
                x=u_dense[mask],
                y=iv_dense_pct,
                mode="lines",
                line=dict(color=color, width=2),
                name=label,
                legendgroup=s.expiry,
                customdata=u_dense[mask] - 1.0,
                hovertemplate=(
                    f"<b>{s.expiry} Spline</b><br>"
                    "Put Δ: %{customdata:+.2f}<br>"
                    "Spline IV: %{y:.2f}%<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    # Custom tick labels on 3D Delta Axis
    tick_u = [0.10, 0.25, 0.50, 0.75, 0.90]
    tick_labels = ["10Δ Put", "25Δ Put", "50Δ ATM", "25Δ Call", "10Δ Call"]

    fig.update_layout(
        title=dict(
            text=(
                f"SPX Implied Volatility Surface | Snapshot: {surface.timestamp} | "
                f"Spot Forward F₀={surface.forwards[0]:.2f}"
            ),
            font=dict(size=14, color="#eceff1"),
            x=0.01,
        ),
        template="plotly_dark",
        height=720,
        margin=dict(l=25, r=25, t=65, b=35),
        scene=dict(
            xaxis=dict(
                title="Option Delta",
                tickmode="array",
                tickvals=tick_u,
                ticktext=tick_labels,
            ),
            yaxis=dict(title="Maturity (DTE Days)"),
            zaxis=dict(title="Implied Volatility (%)"),
            camera=dict(eye=dict(x=-1.55, y=-1.45, z=0.85)),
        ),
        xaxis=dict(
            title="Option Delta (Put Δ from -0.10 to -0.90)",
            range=[0.0, 1.0],
            tickmode="array",
            tickvals=np.round(np.arange(0.1, 1.0, 0.1), 1),
            ticktext=["-0.1", "-0.2", "-0.3", "-0.4", "-0.5", "-0.6", "-0.7", "-0.8", "-0.9"],
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.10)",
            minor=dict(dtick=0.05, showgrid=True, gridcolor="rgba(255, 255, 255, 0.04)"),
        ),
        yaxis=dict(title="Implied Volatility (%)"),
        legend=dict(
            orientation="v",
            x=1.01,
            y=0.98,
            font=dict(size=10),
        ),
    )

    if save_html:
        fig.write_html(save_html)
        print(f"[INFO] Saved single-surface dashboard to '{save_html}'.")

    if show_fig:
        fig.show()

    return fig


# =====================================================================
# 2. SLICE DIAGNOSTICS (SMILE + GATHERAL DENSITY g(x))
# =====================================================================
def plot_slice_diagnostics(
    surface: VolSurface,
    show_fig: bool = True,
    save_html: str | None = None,
) -> go.Figure:
    """Plots a 2-row diagnostic chart across all expiries against Option Delta:

    Row 1: Fitted Volatility Smiles, Raw Market Quotes, & Solved Delta Pillars.
    Row 2: Gatheral's Risk-Neutral Density g(Delta) with the g = 0 no-arbitrage floor.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.60, 0.40],
        subplot_titles=(
            f"Calibrated Volatility Smiles in Delta Space ({surface.timestamp})",
            "Gatheral Butterfly Density Condition g(Δ) (Arbitrage-Free requires g ≥ 0)",
        ),
    )

    palette = ["#00e5ff", "#00e676", "#ffea00", "#ff9100", "#ff5252", "#d500f9"]

    # Standardized u = 1 - CallDelta = |PutDelta| coordinates for the solved pillars
    u_pillars = 1.0 - surface.call_deltas

    for idx, s in enumerate(surface.slices):
        color = palette[idx % len(palette)]
        dte = s.tau * 365.0

        # Evaluate dense log-moneyness grid and map to continuous delta coordinate u = 1 - CallDelta
        x_dense = np.linspace(s.x_left - 0.015, s.x_right + 0.015, 200)
        call_delta_dense = np.asarray(
            s.get_forward_delta(x_dense, is_call=True), dtype=np.float64
        )
        u_dense = 1.0 - call_delta_dense

        # Filter to the [0.02, 0.98] delta band so extreme asymptotic tails don't bunch up at 0 and 1
        band_mask = (u_dense >= 0.02) & (u_dense <= 0.98)
        u_plot = u_dense[band_mask]
        x_plot = x_dense[band_mask]

        iv_dense_pct = np.asarray(s.get_iv(x_plot), dtype=np.float64) * 100.0
        g_dense = np.asarray(s.compute_density_g(x_plot), dtype=np.float64)

        # Also map observed discrete market quotes to their delta coordinates
        u_mkt = 1.0 - np.asarray(
            s.get_forward_delta(s.x_market, is_call=True), dtype=np.float64
        )

        # Row 1A: Fitted Spline Curve in Delta Space
        fig.add_trace(
            go.Scatter(
                x=u_plot,
                y=iv_dense_pct,
                mode="lines",
                line=dict(color=color, width=2.2),
                name=f"{s.expiry} ({dte:.1f}d)",
                legendgroup=s.expiry,
                customdata=np.column_stack([u_plot - 1.0, x_plot]),
                hovertemplate=(
                    f"<b>{s.expiry} Spline</b><br>"
                    "Put Δ: %{customdata[0]:+.2f} (u=%{x:.2f})<br>"
                    "x = ln(K/F): %{customdata[1]:+.4f}<br>"
                    "Spline IV: %{y:.2f}%<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        # Row 1B: Raw Market Quotes (Open Circles) at their floating market deltas
        fig.add_trace(
            go.Scatter(
                x=u_mkt,
                y=s.iv_market * 100.0,
                mode="markers",
                marker=dict(size=5.5, color=color, symbol="circle-open"),
                name=f"{s.expiry} Market Quotes",
                legendgroup=s.expiry,
                showlegend=False,
                customdata=np.column_stack([u_mkt - 1.0, s.strikes]),
                hovertemplate=(
                    f"<b>{s.expiry} Market Quote</b><br>"
                    "Strike K: %{customdata[1]:.0f}<br>"
                    "Put Δ: %{customdata[0]:+.3f}<br>"
                    "Market IV: %{y:.2f}%<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        # Row 1C: Solved Standardized Delta Pillars (Diamonds aligned vertically!)
        fig.add_trace(
            go.Scatter(
                x=u_pillars,
                y=surface.iv_grid[idx, :] * 100.0,
                mode="markers",
                marker=dict(
                    size=8.5,
                    color=color,
                    symbol="diamond",
                    line=dict(width=1.2, color="#ffffff"),
                ),
                name=f"{s.expiry} Delta Pillars",
                legendgroup=s.expiry,
                showlegend=False,
                customdata=np.column_stack([
                    surface.delta_pillars,
                    surface.strike_grid[idx, :],
                    surface.x_grid[idx, :],
                ]),
                hovertemplate=(
                    f"<b>{s.expiry} Anchor Pillar</b><br>"
                    "Target Δ: %{customdata[0]:+.2f}<br>"
                    "Solved Strike K*: %{customdata[1]:.2f}<br>"
                    "Solved x*: %{customdata[2]:+.4f}<br>"
                    "Interpolated IV: %{y:.2f}%<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        # Row 2: Gatheral Density g(x) plotted against Delta
        fig.add_trace(
            go.Scatter(
                x=u_plot,
                y=g_dense,
                mode="lines",
                line=dict(color=color, width=1.8),
                name=f"{s.expiry} g(Δ)",
                legendgroup=s.expiry,
                showlegend=False,
                customdata=u_plot - 1.0,
                hovertemplate=(
                    f"<b>{s.expiry} Density</b><br>"
                    "Put Δ: %{customdata:+.2f}<br>"
                    "g(x): %{y:+.4f}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    # Zero floor line for butterfly no-arbitrage
    fig.add_hline(
        y=0.0,
        line_dash="dash",
        line_color="#ff5252",
        line_width=1.5,
        annotation_text="No-Arbitrage Floor g = 0",
        annotation_position="bottom right",
        row=2,
        col=1,
    )

    # 0.1 Subgrid Tick Configuration (u from 0.1 to 0.9)
    tick_vals = np.round(np.arange(0.1, 1.0, 0.1), 1)
    # Trader OTM labels with Put Delta in parentheses:
    tick_labels = [
        "10Δ Put<br>(-0.10)",
        "20Δ Put<br>(-0.20)",
        "30Δ Put<br>(-0.30)",
        "40Δ Put<br>(-0.40)",
        "50Δ ATM<br>(-0.50)",
        "40Δ Call<br>(-0.60)",
        "30Δ Call<br>(-0.70)",
        "20Δ Call<br>(-0.80)",
        "10Δ Call<br>(-0.90)",
    ]

    axis_style = dict(
        range=[0.0, 1.0],
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_labels,
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.10)",
        gridwidth=1,
        minor=dict(
            dtick=0.05,  # 0.05 minor subgrid lines (captures 25d Put at 0.25 and 25d Call at 0.75)
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.04)",
            gridwidth=0.5,
        ),
        zeroline=False,
    )

    fig.update_layout(
        template="plotly_dark",
        height=780,
        margin=dict(l=55, r=30, t=60, b=55),
        xaxis=axis_style,
        xaxis2=dict(
            **axis_style,
            title="Option Delta (OTM Trader Convention & Unified Put Delta Δp)",
        ),
        yaxis=dict(title="Implied Volatility (%)"),
        yaxis2=dict(title="Gatheral Density g(x)"),
    )

    if save_html:
        fig.write_html(save_html)
        print(f"[INFO] Saved slice diagnostics to '{save_html}'.")

    if show_fig:
        fig.show()

    return fig


# =====================================================================
# 3. DYNAMIC MINUTE-BY-MINUTE TIME-SLIDER SURFACE ANIMATION
# =====================================================================
def plot_dynamic_surface_slider(
    surfaces: list[VolSurface],
    n_delta: int = 25,
    n_tau: int = 25,
    delta_method: str = "grid_spline",
    show_fig: bool = True,
    save_html: str | None = None,
) -> go.Figure:
    """Builds an interactive Plotly dashboard with a draggable time slider and

    Play/Pause controls to scrub through the evolving 3D Volatility Surface
    and 2D Skew/Term-Structure minute by minute.
    """
    if not surfaces:
        raise ValueError("No VolSurface objects provided to plot_dynamic_surface_slider.")

    print(f"[INFO] Precomputing 3D surface frames across {len(surfaces)} timestamps...")

    meshes = []
    global_iv_min = float("inf")
    global_iv_max = float("-inf")
    global_dte_max = float("-inf")

    for surf in surfaces:
        u_ax, dte_ax, iv_mesh = evaluate_dense_surface_mesh(
            surf, n_delta=n_delta, n_tau=n_tau, delta_method=delta_method
        )
        meshes.append((u_ax, dte_ax, iv_mesh))
        global_iv_min = min(global_iv_min, float(np.min(iv_mesh)), float(np.min(surf.iv_grid * 100.0)))
        global_iv_max = max(global_iv_max, float(np.max(iv_mesh)), float(np.max(surf.iv_grid * 100.0)))
        global_dte_max = max(global_dte_max, float(np.max(dte_ax)))

    # Pad Z-axis bounds slightly and lock across all frames so surface shifts are visible
    z_range = [max(0.0, math.floor(global_iv_min - 1.0)), math.ceil(global_iv_max + 1.0)]

    # Locate indices for 25d Put (-0.25), 50d ATM (-0.50), and 25d Call (+0.25)
    pillars = surfaces[0].delta_pillars
    idx_25p = int(np.argmin(np.abs(pillars - (-0.25))))
    idx_50a = int(np.argmin(np.abs( np.abs(pillars) - 0.50 )))
    idx_25c = int(np.argmin(np.abs(pillars - 0.25)))

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        column_widths=[0.58, 0.42],
        subplot_titles=(
            "3D Implied Volatility Surface Evolution",
            "Term Structure Evolution (25Δ Put vs. 50Δ ATM vs. 25Δ Call)",
        ),
        horizontal_spacing=0.06,
    )

    # Initial Frame (t = 0)
    s0 = surfaces[0]
    u0, dte0, iv0 = meshes[0]
    u_pillars0 = 1.0 - s0.call_deltas
    u_grid0, dte_grid0 = np.meshgrid(u_pillars0, s0.taus * 365.0)

    # Trace 0: 3D Surface
    fig.add_trace(
        go.Surface(
            x=u0,
            y=dte0,
            z=iv0,
            colorscale="Viridis",
            cmin=z_range[0],
            cmax=z_range[1],
            opacity=0.92,
            colorbar=dict(title="IV (%)", len=0.72, thickness=14, x=0.51),
            name="Vol Surface",
        ),
        row=1,
        col=1,
    )

    # Trace 1: 3D Pillar Knots
    fig.add_trace(
        go.Scatter3d(
            x=u_grid0.flatten(),
            y=dte_grid0.flatten(),
            z=(s0.iv_grid * 100.0).flatten(),
            mode="markers",
            marker=dict(size=3.5, color="#ffffff"),
            name="Pillar Knots",
        ),
        row=1,
        col=1,
    )

    # Trace 2: 25-Delta Put Term Structure
    fig.add_trace(
        go.Scatter(
            x=s0.taus * 365.0,
            y=s0.iv_grid[:, idx_25p] * 100.0,
            mode="lines+markers",
            line=dict(color="#ff5252", width=2.2),
            marker=dict(size=7),
            name="25Δ Put IV",
        ),
        row=1,
        col=2,
    )

    # Trace 3: 50-Delta ATM Term Structure
    fig.add_trace(
        go.Scatter(
            x=s0.taus * 365.0,
            y=s0.iv_grid[:, idx_50a] * 100.0,
            mode="lines+markers",
            line=dict(color="#00e5ff", width=2.5),
            marker=dict(size=7),
            name="50Δ ATM IV",
        ),
        row=1,
        col=2,
    )

    # Trace 4: 25-Delta Call Term Structure
    fig.add_trace(
        go.Scatter(
            x=s0.taus * 365.0,
            y=s0.iv_grid[:, idx_25c] * 100.0,
            mode="lines+markers",
            line=dict(color="#00e676", width=2.2),
            marker=dict(size=7),
            name="25Δ Call IV",
        ),
        row=1,
        col=2,
    )

    # Construct Animation Frames & Slider Steps
    frames = []
    slider_steps = []

    for k, (surf, (uk, dtek, ivk)) in enumerate(zip(surfaces, meshes)):
        u_pk = 1.0 - surf.call_deltas
        ug_k, dteg_k = np.meshgrid(u_pk, surf.taus * 365.0)
        dte_pts = surf.taus * 365.0

        # Short label for slider bar (HH:MM)
        short_time = surf.timestamp[11:16] if len(surf.timestamp) >= 16 else surf.timestamp
        frame_name = surf.timestamp

        frame_title = (
            f"SPX Vol Surface Dynamics | Timestamp: {surf.timestamp} | "
            f"Forward F₀: {surf.forwards[0]:.2f} | "
            f"50Δ ATM Front IV: {surf.iv_grid[0, idx_50a] * 100:.2f}%"
        )

        frames.append(
            go.Frame(
                name=frame_name,
                data=[
                    go.Surface(x=uk, y=dtek, z=ivk, cmin=z_range[0], cmax=z_range[1]),
                    go.Scatter3d(
                        x=ug_k.flatten(),
                        y=dteg_k.flatten(),
                        z=(surf.iv_grid * 100.0).flatten(),
                    ),
                    go.Scatter(x=dte_pts, y=surf.iv_grid[:, idx_25p] * 100.0),
                    go.Scatter(x=dte_pts, y=surf.iv_grid[:, idx_50a] * 100.0),
                    go.Scatter(x=dte_pts, y=surf.iv_grid[:, idx_25c] * 100.0),
                ],
                traces=[0, 1, 2, 3, 4],
                layout=go.Layout(title_text=frame_title),
            )
        )

        slider_steps.append(
            dict(
                method="animate",
                args=[
                    [frame_name],
                    dict(
                        mode="immediate",
                        frame=dict(duration=150, redraw=True),
                        transition=dict(duration=0),
                    ),
                ],
                label=short_time,
            )
        )

    fig.frames = frames

    tick_u = [0.10, 0.25, 0.50, 0.75, 0.90]
    tick_labels = ["10Δ Put", "25Δ Put", "50Δ ATM", "25Δ Call", "10Δ Call"]

    fig.update_layout(
        title=dict(
            text=(
                f"SPX Vol Surface Dynamics | Timestamp: {s0.timestamp} | "
                f"Forward F₀: {s0.forwards[0]:.2f} | "
                f"50Δ ATM Front IV: {s0.iv_grid[0, idx_50a] * 100:.2f}%"
            ),
            font=dict(size=13.5, color="#eceff1"),
            x=0.01,
        ),
        template="plotly_dark",
        height=780,
        margin=dict(l=25, r=25, t=65, b=110),
        scene=dict(
            xaxis=dict(
                title="Option Delta",
                tickmode="array",
                tickvals=tick_u,
                ticktext=tick_labels,
                range=[0.08, 0.92],
            ),
            yaxis=dict(
                title="Maturity (DTE Days)",
                range=[0.0, global_dte_max * 1.05],
            ),
            zaxis=dict(
                title="Implied Volatility (%)",
                range=z_range,
            ),
            camera=dict(eye=dict(x=-1.55, y=-1.45, z=0.85)),
        ),
        xaxis=dict(
            title="Maturity (Days to Expiration)",
            range=[0.0, global_dte_max * 1.05],
        ),
        yaxis=dict(
            title="Implied Volatility (%)",
            range=z_range,
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.01,
                y=-0.06,
                xanchor="left",
                yanchor="top",
                direction="left",
                pad=dict(r=10, t=10),
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=250, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=False),
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(
                    font=dict(size=12, color="#00e5ff"),
                    prefix="Selected Minute: ",
                    visible=True,
                    xanchor="right",
                ),
                transition=dict(duration=0),
                pad=dict(b=10, t=25),
                len=0.86,
                x=0.12,
                y=-0.04,
                steps=slider_steps,
            )
        ],
    )

    if save_html:
        fig.write_html(save_html, auto_play=False)
        print(f"[INFO] Saved dynamic time-slider animation to '{save_html}'.")

    if show_fig:
        fig.show()

    return fig