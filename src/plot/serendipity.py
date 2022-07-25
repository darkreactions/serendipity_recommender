import random
from collections import defaultdict
from typing import Any

import ipywidgets as widgets
import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy
import seaborn as sns
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import binom, bootstrap

from ..utils import AMINES, COLORS, MODEL_CODES, MODEL_SHORT_NAMES

CODE_2_SHORT_NAME = dict(zip(MODEL_CODES, MODEL_SHORT_NAMES))

sns.set(rc={"figure.dpi": 150, "savefig.dpi": 150, "font.family": "Myriad Pro"})
sns.set_context("notebook")
sns.set_style("whitegrid")


def set_fonts(small, med, large):
    matplotlib.rc("font", size=med)
    matplotlib.rc("axes", titlesize=med)
    matplotlib.rc("axes", labelsize=med)
    matplotlib.rc("xtick", labelsize=small)  # fontsize of the tick labels
    matplotlib.rc("ytick", labelsize=med)  # fontsize of the tick labels
    matplotlib.rc("legend", fontsize=large)  # legend fontsize


def parse_mpl_colors(color_string):
    return [
        float(a) / 255 if x < 3 else float(a)
        for x, a in enumerate(color_string[5:-1].split(","))
    ]


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    m = np.mean(a)
    res = bootstrap((a,), np.mean, confidence_level=confidence)
    return res.confidence_interval.low, res.confidence_interval.high - m


def averaged_col(df, col_name, model_names):
    y = []
    y_error = []
    for mn in model_names:
        y.append(df[df["model"].str.startswith(mn)][col_name].mean())
        y_error.append(df[df["model"].str.startswith(mn)][col_name].std())

    return y, y_error


class PlotSerendipity:
    def __init__(self, org_plate_path, seren_plate_path):
        self.original_plate = pd.read_csv(org_plate_path)
        self.seren_plate = pd.read_csv(seren_plate_path)
        self.amines = [
            "ZKRCWINLLKOVCL-UHFFFAOYSA-N",
            "JMXLWMIFDJCGBV-UHFFFAOYSA-N",
            "NJQKYAASWUGXIT-UHFFFAOYSA-N",
            "HJFYRMFYQMIZDG-UHFFFAOYSA-N",
        ]
        axis_options = [
            ("Volume Fraction", "volume_fraction"),
            ("Success Fraction", "success_fraction"),
            ("Serendipity", "seren"),
        ]
        x_dropdown = widgets.Dropdown(
            options=axis_options,
            description="Select X Axis metric:",
            value="volume_fraction",
            style={"description_width": "initial"},
        )
        x_dropdown.observe(self.change_x_callback)

        y_dropdown = widgets.Dropdown(
            options=axis_options,
            description="Select Y Axis metric:",
            value="success_fraction",
            style={"description_width": "initial"},
        )
        y_dropdown.observe(self.change_y_callback)

        self.current_x = "volume_fraction"
        self.current_y = "success_fraction"
        self.fig = None
        self.plot_serendipity(
            self.amines, x_data_name=self.current_x, y_data_name=self.current_y
        )
        widget = widgets.VBox([x_dropdown, y_dropdown, self.fig])
        display(widget)

    def change_x_callback(self, change):
        if change["name"] == "value":
            self.current_x = change["new"]
            self.plot_serendipity(
                self.amines, x_data_name=self.current_x, y_data_name=self.current_y
            )

    def change_y_callback(self, change):
        if change["name"] == "value":
            self.current_y = change["new"]
            self.plot_serendipity(
                self.amines, x_data_name=self.current_x, y_data_name=self.current_y
            )

    def mean_confidence_interval(self, data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m = np.mean(a)
        # h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
        res = bootstrap((a,), np.mean, confidence_level=confidence)
        return m, res.confidence_interval.low, res.confidence_interval.high

    def get_conf_interval_traces(
        self,
        plate_df: pd.DataFrame,
        amines: "list[str]",
        data: "list[Any]",
        arrow_coords: "dict[str, Any]",
        suffix="",
        data_source="bake-off",
        x_data_name="seren",
        y_data_name="success_fraction",
        x_name="Serendipity",
        y_name="Success Fraction",
    ) -> "tuple[Any, Any, Any]":
        org_plate_amine_data = plate_df[plate_df["amine"].isin(amines)]
        models = org_plate_amine_data["model"].unique()

        if data_source == "bake-off":
            shape = "circle-open"
            # shape = "circle"
        else:
            shape = "circle"

        conf_interval = {}
        for model in models:
            if "0" in model:
                model_name = model[:-1]
                model_data = org_plate_amine_data[
                    org_plate_amine_data["model"].str.contains(model_name)
                ]
                # if model_name == "F":
                #    print(model_data[y_data_name])
                x = model_data[x_data_name]
                y = model_data[y_data_name]
                arrow_coords[model].append((x, y))
                x_mean, x_lower, x_upper = self.mean_confidence_interval(x.values)
                y_mean, y_lower, y_upper = self.mean_confidence_interval(y.values)
                conf_interval[f"{model_name} {suffix}"] = (
                    x_lower,
                    y_lower,
                    x_upper,
                    y_upper,
                )
                data.append(
                    go.Scatter(
                        x=[x_mean],
                        y=[y_mean],
                        mode="markers",
                        marker_color=COLORS[model],
                        marker_size=26,
                        marker_symbol=shape,
                        marker_line=dict(width=6),
                        name=f"{dict(zip(MODEL_CODES, MODEL_SHORT_NAMES))[model[:-1]]} {suffix}",
                        text=f"{model[:-1]} {suffix} <br>" + plate_df["amine"],
                        legendgroup=f"{model[:-1]}",
                        hovertemplate="<b>%{text}</b><br><br>"
                        + x_name
                        + " %{x}<br>"
                        + y_name
                        + " %{y}<br>"
                        + "<extra></extra>",
                        error_y=dict(
                            type="data", array=[(y_upper - y_lower) / 2], visible=True
                        ),
                        error_x=dict(
                            type="data", array=[(x_upper - x_lower) / 2], visible=True
                        ),
                    )
                )
        return data, conf_interval, arrow_coords

    def plot_serendipity(
        self,
        amines: "list[str]",
        x_data_name: str = "seren",
        y_data_name: str = "success_fraction",
    ):
        data = []
        x_names = {
            "seren": "Serendipity",
            "volume_fraction": "Volume Fraction",
            "success_fraction": "Success fraction",
        }
        arrow_coords = defaultdict(list)
        data, conf_interval, arrow_coords = self.get_conf_interval_traces(
            self.original_plate,
            amines,
            data,
            arrow_coords=arrow_coords,
            # suffix="exploitation rec.",
            x_data_name=x_data_name,
            y_data_name=y_data_name,
            x_name=x_names[x_data_name],
            y_name=x_names[y_data_name],
        )
        if self.seren_plate is not None:
            data, conf_interval2, arrow_coords = self.get_conf_interval_traces(
                self.seren_plate,
                amines,
                data,
                arrow_coords=arrow_coords,
                # suffix="serendipity rec.",
                data_source="serendipity",
                x_data_name=x_data_name,
                y_data_name=y_data_name,
                x_name=x_names[x_data_name],
                y_name=x_names[y_data_name],
            )
            conf_interval.update(conf_interval2)

        if not self.fig:
            self.fig = go.FigureWidget(
                data=data, layout=go.Layout(plot_bgcolor="rgba(0,0,0,0)")
            )
        else:
            with self.fig.batch_update():
                for i, trace in enumerate(data):
                    self.fig.data[i].x = trace.x
                    self.fig.data[i].y = trace.y
        self.fig.update_layout(
            font_family="Myriad Pro",
            width=1200,
            height=700,
            # title=f"{x_names[x_data_name]} Plot",
            xaxis_title=x_names[x_data_name],
            yaxis_title=x_names[y_data_name],
            font=dict(
                size=36,
            ),
            xaxis=dict(
                mirror=True,
                ticks="outside",
                showline=True,
                linewidth=2,
                linecolor="black",
                gridcolor="Gainsboro",
                zeroline=True,
                zerolinecolor="Gainsboro",
                zerolinewidth=1,
            ),
            yaxis=dict(
                mirror=True,
                ticks="outside",
                showline=True,
                linewidth=2,
                linecolor="black",
                gridcolor="Gainsboro",
                zeroline=True,
                zerolinecolor="Gainsboro",
                zerolinewidth=1,
            ),
            showlegend=True,
            legend=dict(
                bordercolor="Black",
                borderwidth=2,
                # orientation="h",
                # yanchor="top",
                # y=0.95,
                # xanchor="right",
                # x=0.95,
            ),
        )
        self.fig.update_yaxes(range=[0, 1])


class PlotAllScatter:
    def __init__(self, org_plate_path, seren_plate_path):
        self.original_plate = pd.read_csv(org_plate_path)
        self.seren_plate = pd.read_csv(seren_plate_path)
        models = ["CLB", "DT", "KNN", "PLT", "MIT", "F"]
        self.plot_all_scatter(models)

    def get_mean_std_points(self, bakeoff_df, seren_df, col_name, model_name):
        filtered_bo = bakeoff_df[bakeoff_df["model"].str.startswith(model_name)][
            col_name
        ]
        mean_bo, conf_bo = mean_confidence_interval(filtered_bo)
        filtered_seren = seren_df[seren_df["model"].str.startswith(model_name)][
            col_name
        ]
        mean_seren, conf_seren = mean_confidence_interval(filtered_seren)

        return (
            filtered_bo.mean(),
            conf_bo,
            filtered_seren.mean(),
            conf_seren,
        )

    def plot_all_scatter(self, models: "list[str]"):
        """
        Scatter plot that shows volume_fraction, success_fraction and serendipity
        for all models averaged over all amines for bakeoff plate vs Serendipity plate
        """
        bakeoff_df = self.original_plate
        seren_df = self.seren_plate
        FONT_SIZE = 11
        set_fonts(FONT_SIZE, FONT_SIZE, FONT_SIZE)
        fig, axes = plt.subplots(1, 1, figsize=(3.5, 3.5))
        for pos in ["bottom", "top", "right", "left"]:
            axes.spines[pos].set_color("black")
            axes.spines[pos].set_linewidth(0.5)

        axes.tick_params(axis="both", direction="out")
        axes.add_artist(Line2D([-0.01, 1], [-0.01, 1], color="gray", zorder=1))

        marker_styles = {
            "volume_fraction": "o",
            "success_fraction": "^",
            "seren": "s",
        }

        # Add data
        for model in models:
            for data_column, marker in marker_styles.items():
                # Get point data
                x, x_err, y, y_err = self.get_mean_std_points(
                    bakeoff_df, seren_df, data_column, model
                )
                # Scatter points
                axes.scatter(
                    x=x,
                    y=y,
                    color=parse_mpl_colors(COLORS[model]),
                    marker=marker,
                    edgecolor="black",
                    linewidth=1.0,
                    s=70,
                    zorder=3,
                )
                reduced_alpha = parse_mpl_colors(COLORS[model])
                reduced_alpha[-1] = 0.7
                # Error bars
                axes.errorbar(
                    x, y, yerr=y_err, xerr=x_err, color=reduced_alpha, zorder=2
                )

        # Plot properties
        axes.set_ylim([-0.01, 1])
        axes.set_xlim([-0.01, 1])

        # Add legends

        # Model legend
        custom_lines = [
            Patch(
                facecolor=parse_mpl_colors(COLORS[model]),
                edgecolor="black",
                linewidth=1.0,
                alpha=1.0,
                label=CODE_2_SHORT_NAME[model],
            )
            for model in models
        ]

        legend = fig.legend(
            handles=custom_lines,
            loc="center",
            ncol=3,
            bbox_to_anchor=(0.52, -0.07),
            frameon=False,
            fontsize=10,
        )

        # Data type legend
        marker_labels = {
            "Volume Fraction": "o",
            "Success Fraction": "^",
            "Serendipity": "s",
        }

        custom_markers = [
            Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                label=label,
                markerfacecolor="black",
                markersize=9,
            )
            for label, marker in marker_labels.items()
        ]
        legend2 = axes.legend(
            handles=custom_markers,
            loc="lower right",
            title="Axis measures",
            title_fontsize=9,
            fontsize=9,
        )
        legend2.get_frame().set_edgecolor("black")

        axes.add_artist(legend2)
        axes.set_xlabel("Exploitation recommender")
        axes.set_ylabel("Serendipity recommender")


class PlotBar:
    MODEL_SHORT_NAMES = [
        "KNN",
        "DT",
        "BART",
        "PLT",
        "BGP",
        "FAL",
    ]

    MODEL_CODES = (
        "KNN",
        "DT",
        "CLB",
        "PLT",
        "MIT",
        "F",
    )

    def __init__(self, org_plate_path, seren_plate_path):
        self.original_plate = pd.read_csv(org_plate_path)
        self.seren_plate = pd.read_csv(seren_plate_path)

    def plot_bar(self, col_name):
        bakeoff_df = self.original_plate
        seren_df = self.seren_plate
        set_fonts(9, 9, 9)
        y1, y1_error = averaged_col(bakeoff_df, col_name, self.MODEL_CODES)
        y2, y2_error = averaged_col(seren_df, col_name, self.MODEL_CODES)

        fig, axes = plt.subplots(1, 1, figsize=(3.5, 2))
        width = 0.35
        offset = 0.2
        for pos in ["bottom", "top", "right", "left"]:
            axes.spines[pos].set_color("black")
            axes.spines[pos].set_linewidth(0.5)

        if col_name != "volume_fraction":
            axes.set_ylim([-0.1, 1])
            axes.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axes.xaxis.grid(False)
        # axes.set_xlim([-0.01, 1])

        bars = axes.bar(
            x=np.array([i for i in range(len(self.MODEL_CODES))]) - offset,
            label=self.MODEL_CODES,
            height=y1,
            edgecolor="black",
            linewidth=2.0,
            width=width,
            yerr=y1_error,
            error_kw=dict(lw=1, capsize=3, capthick=1),
        )
        for i, bar in enumerate(bars):
            # bar.set_facecolor(parse_mpl_colors(COLORS[MODEL_CODES[i]]))
            bar.set_facecolor("white")
            bar.set_edgecolor(parse_mpl_colors(COLORS[self.MODEL_CODES[i]]))

        bars2 = axes.bar(
            x=np.array([i for i in range(len(self.MODEL_CODES))]) + offset,
            label=self.MODEL_CODES,
            height=y2,
            edgecolor="black",
            linewidth=2.0,
            width=width,
            yerr=y2_error,
            error_kw=dict(lw=1, capsize=3, capthick=1),
        )
        axes.set_xticklabels(["0"] + list(self.MODEL_SHORT_NAMES))
        axes.set_ylabel(
            "Serendipity"
            if col_name == "seren"
            else col_name.replace("_", " ").capitalize()
        )
        if col_name == "seren":
            axes.set_ylim([-0.1, 1.1])

        axes.set_xlabel("Model")
        for i, bar in enumerate(bars2):
            bar.set_facecolor(parse_mpl_colors(COLORS[self.MODEL_CODES[i]]))

        custom_lines = [
            Patch(
                facecolor="white",
                edgecolor="gray",
                linewidth=2.0,
                alpha=1.0,
                label="Exploitative recommender",
            ),
            Patch(
                facecolor="gray",
                edgecolor="black",
                linewidth=2.0,
                alpha=1.0,
                # hatch="//",
                label="Serendipity recommender",
            ),
        ]
