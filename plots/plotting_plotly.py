from __future__ import annotations
from typing import Optional, Sequence
from pathlib import Path

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

import numpy as np
import pandas as pd
import math


P_FONTSIZE_5 = 48
pio.templates["publish5"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="sans-serif", size=P_FONTSIZE_5),
        title_font=dict(family="sans-serif", size=P_FONTSIZE_5),
        xaxis=dict(title_font=dict(size=P_FONTSIZE_5), tickfont=dict(size=P_FONTSIZE_5-3)),
        yaxis=dict(title_font=dict(size=P_FONTSIZE_5), tickfont=dict(size=P_FONTSIZE_5-3)),
        legend=dict(font=dict(size=P_FONTSIZE_5), title=dict(font=dict(size=P_FONTSIZE_5))),
        annotationdefaults=dict(font=dict(size=P_FONTSIZE_5))
    )
)



def radar_from_df(
    df: pd.DataFrame,
    group_by,
    metrics=None,
    higher_is_better=None,
    category_labels=None,
    title="Model comparison — normalized (higher = better)",
    outfile_pdf="radar_models_normalized.pdf",
):

    df_norm = df.copy()
    
    for m in metrics:
        col = df[m].astype(float).to_numpy()
        med = np.median(col)
        q1 = np.percentile(col, 25)
        q3 = np.percentile(col, 75)
        iqr = q3 - q1 if not np.isclose(q3, q1) else 1.0

        robust = (col - med) / iqr

        norm = 1 / (1 + np.exp(-robust))

        if m not in higher_is_better:
            norm = 1.0 - norm

        df_norm[m] = norm

    categories = [category_labels.get(m, m) for m in metrics]

    cats_closed = categories + [categories[0]]

    fig = go.Figure()

    df_norm_sorted = df_norm.sort_values(group_by)
    
    dash_cycle = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
    dash_map = {conf: dash_cycle[i % len(dash_cycle)] for i, conf in enumerate(df_norm_sorted[group_by].unique())}

    for _, row in df_norm_sorted.iterrows():
        exp = row[group_by]
        values = [float(row[m]) for m in metrics]
        values_closed = values + [values[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=cats_closed,
                mode="lines+markers",
                name=exp,
                line=dict(width=5, dash=dash_map[exp]),
                marker=dict(size=10),
                fill="toself",
                fillcolor="rgba(0,0,0,0.06)",  
                textfont=dict(size=29),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "%{theta}: %{r:.2f}<extra></extra>"
                ),
            )
        )

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    tick_z = np.array([-1, 0, 1])
    tick_vals = sigmoid(tick_z)
    fig.update_layout(
        template="plotly_white+publish5",
        margin=dict(l=40, r=40, t=50, b=20),
        width=980,
        height=720,
        legend=dict(
            title=dict(
                text="",
            ),
            orientation="h",        
            yanchor="top",
            y=-0.15,    
            xanchor="left",
            x=-0.02,
            bgcolor="rgba(255,255,255,0.85)", 
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=0,
            itemwidth=35,
            traceorder="normal",
        ),
        polar=dict(
            bgcolor="white",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickmode="array",
                tickvals=tick_vals,
                ticktext=["-1 IQR", "median", "1 IQR"],
                tickfont=dict(size=28),
                angle=60,
                ticklen=6,
                tickwidth=1,
                gridwidth=2,
                gridcolor="rgba(0,0,0,0.10)",
                linecolor="rgba(0,0,0,0.0)",
            ),
            angularaxis=dict(
                gridcolor="rgba(0,0,0,0.08)",
                rotation=90,
                direction="clockwise",
            ),
        ),
    )

    fig.write_image(outfile_pdf)

    return fig, df, df_norm



def _drop_index_col(df: pd.DataFrame) -> pd.DataFrame:
    """Drop common index column if present."""
    if "Unnamed: 0" in df.columns:
        return df.drop(columns=["Unnamed: 0"])
    return df


def plot_sucess_heatmap(
    df: pd.DataFrame,
    *,
    grids: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
    configs: Optional[Sequence[str]] = None,
    colorscale: str = "Inferno",
    annotate: bool = True,
    annotation_fmt: str = "{:.2f}",
    title: str = "Success rate by <b>model × representation</b> (faceted by grid)",
    width: int = 1200,
    template: str = "plotly_white",
    height = 600,
    out: Optional[str | Path] = None,
) -> go.Figure:
    
    df = _drop_index_col(df).copy()
    required = {"grid", "model", "config", "success"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    grids = list(grids) if grids is not None else sorted(df["grid"].unique())
    models = list(models) if models is not None else sorted(df["model"].unique())
    configs = list(configs) if configs is not None else sorted(df["config"].unique())

    pivots = {}
    for g in grids:
        pv = (
            df[df["grid"] == g]
            .groupby(["model", "config"])["success"]
            .mean()
            .unstack("config")
            .reindex(index=models, columns=configs)
        )
        pivots[g] = pv

    fig = make_subplots(
        rows=len(grids),
        cols=1,
        subplot_titles=[f"<b>{g}</b>" for g in grids],
        vertical_spacing=0.1,
    )

    for i, g in enumerate(grids, start=1):
        pv = pivots[g]
        z = pv.values

        fig.add_trace(
            go.Heatmap(
                z=z,
                x=pv.columns.tolist(),
                y=pv.index.tolist(),
                coloraxis="coloraxis",
            ),
            row=i,
            col=1,
        )

        if annotate:
            for r_idx, model in enumerate(pv.index.tolist()):
                for c_idx, conf in enumerate(pv.columns.tolist()):
                    val = z[r_idx, c_idx]
                    if np.isnan(val):
                        text = "NA"
                    else:
                        text = annotation_fmt.format(val)
                    fig.add_annotation(
                        x=conf,
                        y=model,
                        text=text,
                        showarrow=False,
                        font=dict(size=26),
                        xref=f"x{i}",
                        yref=f"y{i}",
                    )

    fig.update_layout(
        template=template,
        title=dict(text=title, x=0.01, xanchor="left"),
        height=height,
        width=width,
        coloraxis=dict(
            colorscale=colorscale,
            cmin=0,
            cmax=1,
            colorbar=dict(
                title="",
                tickformat=".0%",
                len=0.95,
                thickness=18,
                outlinewidth=0,
            ),
        ),
        margin=dict(l=120, r=60, t=60, b=200),
    )

    for i in range(1, len(grids) + 1):
        fig.update_xaxes(
            row=i,
            col=1,
            title_text="" if i < len(grids) else "configuration",
            tickangle=-25,
            showgrid=False,
            ticks="" if i < len(grids) else "outside",
            ticklen=0 if i < len(grids) else 6,
            showticklabels=i >= len(grids)
        )
        fig.update_yaxes(
            row=i,
            col=1,
            title_text="model" if i == 1 else "",
            showgrid=False,
            ticks="outside",
            ticklen=6,
        )    
    
    fig.for_each_annotation(lambda a: a.update(font=dict(size=42)))

    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(out)

    return fig

def plot_bias_robustness(
    df: pd.DataFrame,
    *,
    metric: str = "success",
    metric_y="success",
    y_tickformat=".0%",
    grids: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
    configs: Optional[Sequence[str]] = None,
    bias_order: Optional[Sequence[int]] = None,
    title: str = "Robustness across disturbance types (bias): <b>absolute success rates</b> (zoomed)",
    width: int = 1200,
    height: int = 400,
    line_width: int = 2,
    marker_size: int = 7,
    template: str = "simple_white",
    out: Optional[str | Path] = None,
) -> go.Figure:

    df = _drop_index_col(df).copy()
    required = {"grid", "model", "config", "bias", "success"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    grids = list(grids) if grids is not None else sorted(df["grid"].unique())
    models = list(models) if models is not None else sorted(df["model"].unique())
    configs = list(configs) if configs is not None else sorted(df["config"].unique())
    bias_order = list(bias_order) if bias_order is not None else sorted(df["bias"].unique())

    succ_bias = (
        df.groupby(["grid", "model", "config", "bias"], observed=True)[metric]
        .mean()
        .reset_index()
    )

    def sparse_ticks(xs: Sequence[int], max_ticks: int = 6) -> list[int]:
        xs = list(xs)
        n = len(xs)
        if n <= max_ticks:
            return xs
        idx = np.linspace(0, n - 1, num=max_ticks, dtype=int)
        tickvals = [xs[i] for i in idx]
        tickvals[0], tickvals[-1] = xs[0], xs[-1]
        outv, seen = [], set()
        for v in tickvals:
            if v not in seen:
                outv.append(v); seen.add(v)
        return outv

    x_tickvals = sparse_ticks(bias_order, max_ticks=6)

    okabe_ito = [
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#009E73",  # bluish green
        "#CC79A7",  # reddish purple
        "#F0E442",  # yellow
        "#56B4E9",  # sky blue
        "#E69F00",  # orange
        "#000000",  # black
    ]

    dash_cycle = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
    symbol_cycle = [
        "circle", "square", "diamond", "triangle-up", "triangle-down",
        "cross", "x", "pentagon", "hexagon", "star",
    ]

    color_map = {conf: okabe_ito[i % len(okabe_ito)] for i, conf in enumerate(configs)}
    dash_map  = {conf: dash_cycle[i % len(dash_cycle)] for i, conf in enumerate(configs)}
    sym_map   = {conf: symbol_cycle[i % len(symbol_cycle)] for i, conf in enumerate(configs)}

    fig = make_subplots(
        rows=len(grids),
        cols=len(models),
        subplot_titles=[f"<b>{g}</b> · {m}" for g in grids for m in models],
        horizontal_spacing=0.06,
        vertical_spacing=0.11,
    )

    for r, g in enumerate(grids, start=1):
        for c, m in enumerate(models, start=1):
            sub = succ_bias[(succ_bias["grid"] == g) & (succ_bias["model"] == m)]

            for conf in configs:
                s = (
                    sub[sub["config"] == conf]
                    .set_index("bias")
                    .reindex(bias_order)[metric]
                )

                fig.add_trace(
                    go.Scatter(
                        x=bias_order,
                        y=s.values,
                        mode="lines+markers",
                        name=conf,
                        legendgroup=conf,
                        showlegend=(r == 1 and c == 1),

                        line=dict(
                            width=line_width,
                            color=color_map[conf],
                            dash=dash_map[conf],
                            shape="linear",
                        ),

                        marker=dict(
                            size=marker_size,
                            symbol=sym_map[conf],
                            color="white",
                            line=dict(color=color_map[conf], width=1.6),
                        ),
                    ),
                    row=r,
                    col=c,
                )

            fig.update_yaxes(
                row=r,
                col=c,
                title_text="",
                showticklabels=(c == 1),
                tickformat=y_tickformat,
                ticks="",
                showline=True,
                linewidth=1,
                linecolor="rgba(0,0,0,0.30)",
                zeroline=False,
                showgrid=True,
                gridcolor="rgba(0,0,0,0.08)",
                griddash="dot",
            )

            fig.update_xaxes(
                row=r,
                col=c,
                title_text="bias" if r == len(grids) else None,
                showticklabels=(r == len(grids)),
                tickmode="array",
                tickvals=x_tickvals,
                ticks="",
                showline=True,
                linewidth=1,
                linecolor="rgba(0,0,0,0.30)",
                zeroline=False,
                showgrid=True,
                gridcolor="rgba(0,0,0,0.08)",
                griddash="dot",
            )

    fig.update_layout(
        template=template,
        title=dict(text=title, x=0.01, xanchor="left"),
        width=width,
        height=height,
        margin=dict(l=90, r=30, t=90, b=70),

        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=1.15,
            xanchor="left",
            x=-0.08,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.10)",
            borderwidth=0,
            itemwidth=60
        ),
    )

    fig.for_each_annotation(lambda a: a.update(font=dict(size=42)))

    fig.add_annotation(
        text=metric_y,
        xref="paper",
        yref="paper",
        x=-0.2,
        y=0.5,
        showarrow=False,
        textangle=-90,
        font=dict(size=P_FONTSIZE_5),
    )

    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".html":
            fig.write_html(out, include_plotlyjs="cdn")
        else:
            fig.write_image(out)

    return fig


def plot_avg_wrong_adjustment_pct_by_config(
    df: pd.DataFrame,
    *,
    metric: str = "wrong_adj_pct",
    metric_y: str = "avg. wrong adjustments (%)",
    y_tickformat: str = ".1f",
    grids: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
    configs: Optional[Sequence[str]] = None,
    title: str = (
        "Average wrong-adjustment percentage by config (bias-averaged)"
    ),
    width: int = 1200,
    height: int = 400,
    template: str = "simple_white",
    out: Optional[str | Path] = None,
) -> go.Figure:

    df = _drop_index_col(df).copy()

    required = {"grid", "model", "config", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    grids = list(grids) if grids is not None else sorted(df["grid"].unique())
    models = list(models) if models is not None else sorted(df["model"].unique())
    configs = list(configs) if configs is not None else sorted(df["config"].unique())

    agg = (
        df.groupby(["grid", "model", "config"], observed=True)[metric]
        .mean()
        .reset_index()
    )

    okabe_ito = [
        "#0072B2", "#D55E00", "#009E73", "#CC79A7",
        "#F0E442", "#56B4E9", "#E69F00", "#000000",
    ]
    color_map = {c: okabe_ito[i % len(okabe_ito)] for i, c in enumerate(configs)}

    fig = make_subplots(
        rows=len(grids),
        cols=len(models),
        subplot_titles=[f"<b>{g}</b> · {m}" for g in grids for m in models],
        horizontal_spacing=0.06,
        vertical_spacing=0.11,
    )

    for r, g in enumerate(grids, start=1):
        for c, m in enumerate(models, start=1):
            sub = agg[
                (agg["grid"] == g) &
                (agg["model"] == m)
            ]

            if sub.empty:
                continue

            y = (
                sub.set_index("config")
                .reindex(configs)[metric]
            )

            fig.add_trace(
                go.Bar(
                    x=configs,
                    y=y.values,
                    marker=dict(
                        color=[color_map[cfg] for cfg in configs],
                        line=dict(color="rgba(0,0,0,0.4)", width=0.8),
                    ),
                    hovertemplate=(
                        "<b>config</b>: %{x}<br>"
                        f"<b>{metric_y}</b>: %{{y:{y_tickformat}}}%"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=r,
                col=c,
            )

            fig.update_yaxes(
                row=r,
                col=c,
                title_text=metric_y if c == 1 else None,
                showticklabels=(c == 1),
                tickformat=y_tickformat,
                zeroline=False,
                showgrid=True,
                gridcolor="rgba(0,0,0,0.08)",
                griddash="dot",
            )

            fig.update_xaxes(
                row=r,
                col=c,
                title_text="config" if r == len(grids) else None,
                showticklabels=(r == len(grids)),
                type="category",
                zeroline=False,
                showgrid=False,
            )

    fig.update_layout(
        template=template,
        title=dict(text=title, x=0.01, xanchor="left"),
        width=width,
        height=height,
        margin=dict(l=90, r=30, t=90, b=90),
        bargap=0.25,
        hoverlabel=dict(bgcolor="white"),
    )

    fig.for_each_annotation(lambda a: a.update(font=dict(size=24)))

    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".html":
            fig.write_html(out, include_plotlyjs="cdn")
        else:
            fig.write_image(out)

    return fig



def plot_bias_robustness_model_agg(
    df: pd.DataFrame,
    *,
    grid_to_optimal_slack=None,
    show_baseline=True,
    metric: str = "success",
    metric_y="success",
    y_tickformat=".0%",
    grids: Optional[Sequence[str]] = None,
    configs: Optional[Sequence[str]] = None,
    bias_order: Optional[Sequence[int]] = None,
    title: str = "Robustness across disturbance types (bias): <b>absolute success rates</b> (zoomed)",
    width: int = 1200,
    height: int = 400,
    line_width: int = 2,
    marker_size: int = 7,
    template: str = "simple_white",
    out: Optional[str | Path] = None,
) -> go.Figure:

    df = _drop_index_col(df).copy()
    required = {"grid", "model", "config", "bias", "success"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    grids = list(grids) if grids is not None else sorted(df["grid"].unique())
    configs = list(configs) if configs is not None else sorted(df["config"].unique())
    bias_order = list(bias_order) if bias_order is not None else sorted(df["bias"].unique())

    succ_bias = (
        df.groupby(["grid", "config", "bias"], observed=True)[metric]
        .mean()
        .reset_index()
    )

    def sparse_ticks(xs: Sequence[int], max_ticks: int = 6) -> list[int]:
        xs = list(xs)
        n = len(xs)
        if n <= max_ticks:
            return xs
        idx = np.linspace(0, n - 1, num=max_ticks, dtype=int)
        tickvals = [xs[i] for i in idx]
        tickvals[0], tickvals[-1] = xs[0], xs[-1]
        outv, seen = [], set()
        for v in tickvals:
            if v not in seen:
                outv.append(v); seen.add(v)
        return outv

    x_tickvals = sparse_ticks(bias_order, max_ticks=6)

    okabe_ito = [
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#009E73",  # bluish green
        "#CC79A7",  # reddish purple
        "#F0E442",  # yellow
        "#56B4E9",  # sky blue
        "#E69F00",  # orange
        "#000000",  # black
    ]

    dash_cycle = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
    symbol_cycle = [
        "circle", "square", "diamond", "triangle-up", "triangle-down",
        "cross", "x", "pentagon", "hexagon", "star",
    ]

    color_map = {conf: okabe_ito[i % len(okabe_ito)] for i, conf in enumerate(configs)}
    dash_map  = {conf: dash_cycle[i % len(dash_cycle)] for i, conf in enumerate(configs)}
    sym_map   = {conf: symbol_cycle[i % len(symbol_cycle)] for i, conf in enumerate(configs)}

    fig = make_subplots(
        rows=1,
        cols=len(grids),
        subplot_titles=[f"<b>{g}</b> · {title}" for g in grids],
        horizontal_spacing=0.06,
        vertical_spacing=0.11,
    )

    for c, g in enumerate(grids, start=1):
        sub = succ_bias[(succ_bias["grid"] == g)]
        slack_y = grid_to_optimal_slack[g]
        if show_baseline:
            fig.add_trace(
                go.Scatter(
                    x=bias_order,
                    y=slack_y,
                    mode="lines+markers",
                    name="OPF",
                    showlegend=(c == 1),
                    line=dict(
                        width=line_width,
                        color="black",
                    ),
                    marker=dict(
                        size=marker_size,
                    ),
                ),
                row=1,
                col=c
            )

        for conf in configs:
            s = (
                sub[sub["config"] == conf]
                .set_index("bias")
                .reindex(bias_order)[metric]
            )

            fig.add_trace(
                go.Scatter(
                    x=bias_order,
                    y=s.values,
                    mode="lines+markers",
                    name=conf,
                    legendgroup=conf,
                    showlegend=(c == 1),

                    line=dict(
                        width=line_width,
                        color=color_map[conf],
                        dash=dash_map[conf],
                        shape="linear",
                    ),

                    marker=dict(
                        size=marker_size,
                        symbol=sym_map[conf],
                        color="white",
                        line=dict(color=color_map[conf], width=1.6),
                    ),
                ),
                row=1,
                col=c
            )

        fig.update_yaxes(
            range=(-1.6,1.1),
            row=1,
            col=c,
            title_text=metric_y if c == 1 else "",
            showticklabels=(c == 1),
            tickformat=y_tickformat,
            ticks="",
            showline=True,
            linewidth=1,
            linecolor="rgba(0,0,0,0.30)",
            zeroline=False,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            griddash="dot",
        )

        fig.update_xaxes(
            row=1,
            col=c,
            title_text="bias",
            showticklabels=True,
            tickmode="array",
            tickvals=x_tickvals,
            ticks="",
            showline=True,
            linewidth=1,
            linecolor="rgba(0,0,0,0.30)",
            zeroline=False,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            griddash="dot",
        )

    fig.update_layout(
        template=template,
        title=dict(text="", x=0.0, y=0.86, xanchor="left"),
        width=width,
        height=height,
        margin=dict(l=100, r=60, t=90, b=70),
        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=1.15,
            xanchor="left",
            x=-0.15,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.10)",
            borderwidth=0,
            itemwidth=40
        ),
    )

    fig.for_each_annotation(lambda a: a.update(font=dict(size=42)))

    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".html":
            fig.write_html(out, include_plotlyjs="cdn")
        else:
            fig.write_image(out)

    return fig



def violin_with_raw(raw_data, metric_col, config_col="config", height=600, x_title="configuration", y_title="", title=None, disable_zero_line=False, clip_on_percentile=100, template="plotly_white"):
    configs = list(raw_data[config_col].dropna().unique())

    fig = go.Figure()

    for cfg in configs:
        y = raw_data.loc[raw_data[config_col] == cfg, metric_col]

        fig.add_trace(
            go.Violin(
                x=[cfg] * len(y),
                y=y,
                name=str(cfg),
                side="both",
                width=0.75,
                scalemode="width",
                spanmode="hard",
                box_visible=True,
                meanline_visible=False,
                points="outliers",
                jitter=0.28,
                pointpos=0.0,
                marker=dict(
                    size=4,
                    opacity=0.25,
                    line=dict(width=0) 
                ),
                fillcolor="rgba(0,0,0,0.10)",
                line=dict(width=1.6),
            )
        )

    if not disable_zero_line:
        fig.add_hline(
            y=0,
            line_width=1.2,
            line_dash="dash",
            line_color="rgba(80,80,80,0.65)"
        )


    fig.update_layout(
        title=dict(text="", x=0.02, xanchor="left"),
        violinmode="group",
        violingap=0.25,
        violingroupgap=0.10,
        template=template,
        width=1100,
        height=height,
        margin=dict(l=70, r=30, t=70, b=70),
        showlegend=False,
    )

    fig.update_xaxes(
        title_text=x_title,
        tickangle=0,
        showgrid=False,
        zeroline=False
    )

    fig.update_yaxes(
        title_text=y_title,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False
    )

    if clip_on_percentile < 100:
        ymax = np.percentile(raw_data[metric_col], clip_on_percentile)

        fig.update_yaxes(range=[0, ymax])
        fig.add_annotation(
            x=0.5, y=0.98, xref="paper", yref="paper",
            text=f"Violins clipped at {clip_on_percentile}th percentile for readability; extreme outliers retained as points.",
            showarrow=False,
            font=dict(color="gray")
        )

    return fig


def violin_with_raw_double(
    raw_data,
    metric_col_top,
    metric_col_bottom,
    y_top_ticks_format=".0%",
    y_bottom_ticks_format=None,
    config_col="config",
    height=600,
    x_title="configuration",
    y_title_top="",
    y_title_bottom="",
    title=None,
    disable_zero_line=False,
    clip_on_percentile=100,
    template="plotly_white",
):
    # Stable ordering (feel free to replace with your preferred order)
    configs = list(raw_data[config_col].dropna().unique())

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06
    )

    # --- TOP ---
    for cfg in configs:
        y = raw_data.loc[raw_data[config_col] == cfg, metric_col_top]
        fig.add_trace(
            go.Violin(
                x=[cfg] * len(y),
                y=y,
                name=str(cfg),
                side="both",
                width=0.75,
                scalemode="width",
                spanmode="hard",
                box_visible=True,
                meanline_visible=False,
                points="outliers",
                jitter=0.28,
                pointpos=0.0,
                marker=dict(
                    size=4,
                    opacity=0.25,
                    line=dict(width=0)
                ),
                fillcolor="rgba(0,0,0,0.10)",
                line=dict(width=1.6),
            ),
            row=1, col=1
        )

    # --- BOTTOM ---
    for cfg in configs:
        y = raw_data.loc[raw_data[config_col] == cfg, metric_col_bottom]
        fig.add_trace(
            go.Violin(
                x=[cfg] * len(y),
                y=y,
                name=str(cfg),
                side="both",
                width=0.75,
                scalemode="width",
                spanmode="hard",
                box_visible=True,
                meanline_visible=False,
                points="outliers",
                jitter=0.28,
                pointpos=0.0,
                marker=dict(
                    size=4,
                    opacity=0.25,
                    line=dict(width=0)
                ),
                fillcolor="rgba(0,0,0,0.10)",
                line=dict(width=1.6),
            ),
            row=2, col=1
        )

    if not disable_zero_line:
        # add a zero line to EACH subplot
        fig.add_hline(
            y=0, row=1, col=1,
            line_width=1.2, line_dash="dash",
            line_color="rgba(80,80,80,0.65)"
        )
        fig.add_hline(
            y=0, row=2, col=1,
            line_width=1.2, line_dash="dash",
            line_color="rgba(80,80,80,0.65)"
        )

    fig.update_layout(
        title=dict(text=(title or ""), x=0.02, xanchor="left"),
        violinmode="group",
        violingap=0.25,
        violingroupgap=0.10,
        template=template,
        width=1100,
        height=height,
        margin=dict(l=70, r=30, t=30, b=140),
        showlegend=False,
    )
    fig.update_xaxes(
        title_text="",
        tickangle=0,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        row=1, col=1,
    )
    fig.update_xaxes(
        title_text=x_title,
        tickangle=0,
        showgrid=False,
        zeroline=False,
        row=2, col=1
    )

    # y axes
    fig.update_yaxes(
        title_text=y_title_top,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
        row=1, col=1,
        tickformat=y_top_ticks_format
    )
    fig.update_yaxes(
        title_text=y_title_bottom,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
        row=2, col=1,
        tickformat=y_bottom_ticks_format
    )

    if clip_on_percentile < 100:
        ymax_top = np.percentile(raw_data[metric_col_top], clip_on_percentile)
        ymax_bottom = np.percentile(raw_data[metric_col_bottom], clip_on_percentile)

        fig.update_yaxes(range=[0, ymax_top], row=1, col=1)
        fig.update_yaxes(range=[0, ymax_bottom], row=2, col=1)

        fig.add_annotation(
            x=0.5, y=0.99, xref="paper", yref="paper",
            text=(
                f"Violins clipped at {clip_on_percentile}th percentile for readability; "
                f"extreme outliers retained as points."
            ),
            showarrow=False,
            font=dict(color="gray")
        )

    return fig

def plot_configs_by_model_and_grid_bars(
    df: pd.DataFrame,
    metric_col: str,
    *,
    model_col: str = "model",
    grid_col: str = "grid",
    config_col: str = "config",
    run_id_col: str = "run_id",
    bias_col: str = "bias",
    within_run_bias_agg: str = "mean",   # duplicates within (run_id,bias)
    across_run_bias_agg: str = "mean",   # across (run_id,bias) values
    error_bars: str | None = "se",       # "se", "sd", or None
    sort_configs: str | None = "desc",   # "asc", "desc", or None (global order)
    template: str = "plotly_white+publish5",
    height: int = 600,
    width: int = 1200,
    y_label: str | None = None,          # if None -> metric_col
    show_values: bool = False,
    value_format: str = ".3g",
    bargap: float = 0.22,
    y_label_gap=-0.14,
):
    required = {model_col, grid_col, config_col, run_id_col, bias_col, metric_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    g1_cols = [model_col, grid_col, config_col, run_id_col, bias_col]
    df1 = (
        df.groupby(g1_cols, dropna=False)[metric_col]
          .agg(within_run_bias_agg)
          .reset_index()
          .rename(columns={metric_col: "_val"})
    )

    g2_cols = [model_col, grid_col, config_col]
    grp = df1.groupby(g2_cols, dropna=False)["_val"]

    if across_run_bias_agg == "mean":
        center = grp.mean()
    elif across_run_bias_agg == "median":
        center = grp.median()
    elif across_run_bias_agg == "sum":
        center = grp.sum()
    else:
        center = grp.agg(across_run_bias_agg)

    out = center.rename("y").reset_index()

    if error_bars is None:
        out["err"] = 0.0
        use_err = False
    else:
        sd = grp.std(ddof=1).rename("_sd").reset_index()
        n = grp.count().rename("_n").reset_index()
        out = out.merge(sd, on=g2_cols, how="left").merge(n, on=g2_cols, how="left")
        out["_sd"] = out["_sd"].fillna(0.0)
        out["_n"] = out["_n"].fillna(0)

        if error_bars == "sd":
            out["err"] = out["_sd"]
        elif error_bars == "se":
            out["err"] = out.apply(lambda r: (r["_sd"] / math.sqrt(r["_n"])) if r["_n"] else 0.0, axis=1)
        else:
            raise ValueError("error_bars must be one of: 'se', 'sd', None")

        use_err = True

    models = sorted(out[model_col].unique())
    grids = sorted(out[grid_col].unique())

    configs = out[config_col].astype(str)
    out[config_col] = configs

    config_order = sorted(out[config_col].unique())

    subplot_titles = [
        f"<b>{g}</b> · {m}"
        for g in grids
        for m in models
    ]

    fig = make_subplots(
        rows=len(grids),
        cols=len(models),
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
        shared_yaxes=False,  
    )

    for i, g in enumerate(grids, start=1):
        for j, m in enumerate(models, start=1):
            cell = out[(out[model_col] == m) & (out[grid_col] == g)].copy()
            if cell.empty:
                continue

            cell = cell.set_index(config_col).reindex(config_order).reset_index()
            x = cell[config_col].tolist()
            y = cell["y"].tolist()
            err = cell["err"].tolist()

            text = None
            textposition = None
            if show_values:
                text = [format(v, value_format) if pd.notna(v) else "" for v in y]
                textposition = "outside"

            fig.add_trace(
                go.Bar(
                    x=x,
                    y=y,
                    error_y=(dict(type="data", array=err, visible=True) if use_err else None),
                    text=text,
                    textposition=textposition,
                    cliponaxis=False if show_values else True,
                ),
                row=i,
                col=j,
            )

    fig.update_layout(
        template=template,
        title=dict(
            text="",
            x=0.5,
            xanchor="center",
            font=dict(size=20),
        ),
        font=dict(family="Inter, Arial, sans-serif", size=14),
        margin=dict(l=80, r=30, t=60, b=80),
        bargap=bargap,
        height=height,
        width=width,
        showlegend=False,
    )

    fig.update_yaxes(title_text="")

    fig.update_xaxes(tickangle=-45, ticks="outside", ticklen=0, showline=False)
    fig.update_yaxes(ticks="outside", ticklen=0, zeroline=False, showline=False)

    for r in range(1, len(grids) + 1):
        for c in range(1, len(models) + 1):
            fig.update_yaxes(
                showticklabels=(c == 1),
                ticks=("outside" if c == 1 else ""),
                showline=False,
                row=r,
                col=c,
            )
            if c != 1:
                fig.update_yaxes(showgrid=True, row=r, col=c)

    for c in range(1, len(models) + 1):
        fig.update_xaxes(
            ticks="",
            ticklen=0,
            showline=False,
            showticklabels=False,
            row=1,
            col=c,
        )

    if y_label is None:
        y_label = metric_col

    fig.for_each_annotation(lambda a: a.update(font=dict(size=P_FONTSIZE_5-10)))

    fig.add_annotation(
        text=y_label,
        xref="paper",
        yref="paper",
        x=y_label_gap,
        y=0.5,
        showarrow=False,
        textangle=-90,
        font=dict(size=P_FONTSIZE_5),
    )


    return fig
