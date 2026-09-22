import numpy as np
import plotly.graph_objects as go
import polars as pl


def plot_multi_event_momentum_sample(
        df: pl.DataFrame,
        min_events: int = 2,
        max_span_bars: int = 150,
        pad_bars: int = 20,
        require_positive_only: bool = False,
        chart_height: int = 460,
):
    # Vectorized lookup arrays
    labels_arr = (
        df["target_label"].to_numpy()
        if "target_label" in df.columns
        else np.zeros(len(df))
    )
    ends_arr = (
        df["target_end_idx"].to_numpy()
        if "target_end_idx" in df.columns
        else np.full(len(df), -1)
    )
    mfe_arr = (
        df["target_mfe_bps"].to_numpy()
        if "target_mfe_bps" in df.columns
        else np.zeros(len(df))
    )
    z_arr = (
        df["z_impulse"].to_numpy()
        if "z_impulse" in df.columns
        else np.zeros(len(df))
    )

    # Filter candidate impulse indices
    if require_positive_only:
        event_indices = np.where(labels_arr == 1)[0]
    elif (ends_arr != -1).any():
        event_indices = np.where(ends_arr != -1)[0]
    elif (labels_arr == 1).any():
        event_indices = np.where(labels_arr == 1)[0]
    elif (z_arr >= 1.5).any():
        event_indices = np.where(z_arr >= 1.5)[0]
    else:
        print("Error: No valid event triggers found in dataset.")
        return

    if len(event_indices) < min_events:
        print(
            f"Not enough events found (needed >= {min_events}, found"
            f" {len(event_indices)})."
        )
        return

    # Locate multi-event clusters
    valid_clusters = []
    for i in range(len(event_indices) - min_events + 1):
        first_start = int(event_indices[i])
        last_event_start = int(event_indices[i + min_events - 1])
        last_end = int(max(ends_arr[last_event_start], last_event_start + 1))
        if (last_end - first_start) <= max_span_bars:
            valid_clusters.append((first_start, last_end))

    if not valid_clusters:
        first_start = int(event_indices[0])
        last_end = int(
            max(
                ends_arr[event_indices[min_events - 1]],
                event_indices[min_events - 1] + 1,
            )
        )
        valid_clusters.append((first_start, last_end))

    cluster_idx = int(np.random.choice(len(valid_clusters)))
    first_start, last_end = valid_clusters[cluster_idx]

    idx_low = max(0, first_start - pad_bars)
    idx_high = min(len(df), last_end + pad_bars)

    # Slice sub-dataframe & Convert to AEST (Australia/Sydney)
    sub_pl = df.slice(idx_low, idx_high - idx_low)
    try:
        if sub_pl["ts_event"].dtype.time_zone is None:
            sub_pl = sub_pl.with_columns(
                pl.col("ts_event")
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone("Australia/Sydney")
            )
        else:
            sub_pl = sub_pl.with_columns(
                pl.col("ts_event").dt.convert_time_zone("Australia/Sydney")
            )
    except Exception:
        sub_pl = sub_pl.with_columns(
            (pl.col("ts_event") + pl.duration(hours=10)).alias("ts_event")
        )

    sub_df = sub_pl.to_pandas().reset_index(drop=True)
    sample_price = sub_df["close"].dropna().iloc[0]
    if sample_price < 0.01:
        scale_factor = 1e9
        for col in ["open", "high", "low", "close", "vwap"]:
            if col in sub_df.columns:
                sub_df[col] = sub_df[col] * scale_factor
    n_bars = len(sub_df)
    x_indices = list(range(n_bars))

    # Generate spaced-out, strictly horizontal X-Axis ticks
    step = 5 if n_bars <= 60 else (10 if n_bars <= 120 else 15)
    tick_vals = list(range(0, n_bars, step))
    if (n_bars - 1) not in tick_vals and (n_bars - 1 - tick_vals[-1]) >= (
            step // 2
    ):
        tick_vals.append(n_bars - 1)

    tick_texts = []
    first_date = sub_df.iloc[0]["ts_event"].strftime("%Y-%m-%d")

    for val in tick_vals:
        dt = sub_df.iloc[val]["ts_event"]
        if val == 0:
            # Only show the date once at the very first bar
            tick_texts.append(f"<b>{first_date}</b><br>{dt.strftime('%H:%M')}")
        else:
            # Pure HH:MM thereafter
            tick_texts.append(dt.strftime("%H:%M"))

    # Render Candlestick Chart
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=x_indices,
            open=sub_df["open"],
            high=sub_df["high"],
            low=sub_df["low"],
            close=sub_df["close"],
            name="1m OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            whiskerwidth=0.4,
        )
    )

    if "vwap" in sub_df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_indices,
                y=sub_df["vwap"],
                mode="lines",
                name="VWAP",
                line=dict(color="#fdd835", width=1.1, dash="dot"),
            )
        )

    # Annotate Events and Momentum Zones
    window_events = [
        int(idx) for idx in event_indices if idx_low <= idx < (idx_high - 1)
    ]
    stagger_y_tiers = [0.96, 0.88, 0.80]

    for k, ev_start in enumerate(window_events):
        ev_end = int(
            max(ends_arr[ev_start], ev_start + 1)
            if ends_arr[ev_start] != -1
            else ev_start + 5
        )
        rel_start = ev_start - idx_low
        rel_end = min(ev_end - idx_low, n_bars - 1)

        is_winner = bool(labels_arr[ev_start] == 1)
        # Convert basis points to percentage (1 bp = 0.01%)
        mfe_pct = float(mfe_arr[ev_start]) / 100.0
        z_score = float(z_arr[ev_start])

        color = "#00e676" if is_winner else "#ff5252"
        fill_color = (
            "rgba(0, 230, 118, 0.12)" if is_winner else "rgba(255, 82, 82, 0.12)"
        )

        # Shaded momentum zone
        fig.add_vrect(
            x0=rel_start - 0.4,
            x1=rel_end + 0.4,
            fillcolor=fill_color,
            layer="below",
            line_width=1,
            line_dash="dot",
            line_color=color,
        )

        # Impulse Trigger marker below the candle
        fig.add_trace(
            go.Scatter(
                x=[rel_start],
                y=[sub_df.iloc[rel_start]["low"] * 0.9997],
                mode="markers+text",
                marker=dict(symbol="triangle-up", size=8, color=color),
                text=[f"#{k + 1}<br>Z={z_score:.1f}"],
                textposition="bottom center",
                textfont=dict(size=8, color="#cfd8dc"),
                showlegend=False,
            )
        )

        # Top-staggered pill badge with percentage return
        y_pos = stagger_y_tiers[k % len(stagger_y_tiers)]
        status_label = "WIN" if is_winner else "FAIL"

        fig.add_annotation(
            x=rel_start,
            y=y_pos,
            yref="paper",
            text=f"#{k + 1} {status_label} +{mfe_pct:.2f}%",
            showarrow=False,
            xanchor="left",
            font=dict(size=8, color=color),
            bgcolor="rgba(18, 18, 18, 0.85)",
            bordercolor=color,
            borderwidth=1,
            borderpad=2,
        )

    # Axis and Layout Scaling for 1440p Ultrawide
    price_range = sub_df["high"].max() - sub_df["low"].min()
    major_dtick = 0.50 if price_range > 1.20 else 0.20

    fig.update_layout(
        title=dict(
            text=(
                "Multi-Momentum Window Audit (AEST) — "
                f"{len(window_events)} Events Detected"
            ),
            font=dict(size=12, color="#eceff1"),
            x=0.01,
            y=0.98,
        ),
        font=dict(size=8.5, color="#90a4ae"),
        template="plotly_dark",
        height=chart_height,
        margin=dict(l=45, r=25, t=35, b=35),
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_texts,
            tickangle=0,  # Completely horizontal
            tickfont=dict(size=8.5, color="#b0bec5"),
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.05)",
        ),
        yaxis=dict(
            tickformat=".2f",  # 2 decimal places (cents)
            dtick=major_dtick,  # Major gridlines at $0.20 or $0.50
            minor=dict(
                dtick=0.10,  # Minor gridline every 10 cents
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.03)",
                gridwidth=0.5,
                ticks="inside",
                ticklen=3,
            ),
            tickfont=dict(size=8.5, color="#b0bec5"),
            side="left",
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.08)",
        ),
        legend=dict(
            font=dict(size=8),
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1.0,
        ),
    )

    fig.show()
