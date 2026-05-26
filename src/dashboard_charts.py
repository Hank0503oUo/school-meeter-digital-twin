from __future__ import annotations

import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go

from src.dashboard_state import CampusState


def build_timeseries_chart(
    state: CampusState,
    values: pd.Series | np.ndarray | list[float],
    title: str = "Timeseries",
    y_title: str = "kW",
) -> go.Figure:
    """Build a simple reusable time-series figure."""
    series = pd.Series(values, copy=False)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(series))),
            y=pd.to_numeric(series, errors="coerce").fillna(0.0),
            mode="lines",
            name=state.campus_name or state.campus_id.upper() or "Campus",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Hour",
        yaxis_title=y_title,
        template="plotly_white",
        margin=dict(l=30, r=10, t=40, b=30),
    )
    fig = apply_custom_theme(fig)
    return fig


def apply_custom_theme(fig: go.Figure) -> go.Figure:
    """Apply unified modern theme to Plotly figures."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif", color="#334155"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.95)",
            font_size=13,
            font_family="Inter, sans-serif",
            bordercolor="rgba(0, 0, 0, 0.1)",
        ),
        xaxis=dict(
            gridcolor="#f1f5f9",
            zerolinecolor="#e2e8f0",
            linecolor="#e2e8f0",
        ),
        yaxis=dict(
            gridcolor="#f1f5f9",
            zerolinecolor="#e2e8f0",
            linecolor="#e2e8f0",
        )
    )
    return fig


def as_plotly_pane(fig: go.Figure) -> pn.pane.Plotly:
    """Wrap figure as a Panel Plotly pane."""
    return pn.pane.Plotly(fig, config={"displayModeBar": False}, sizing_mode="stretch_width")
