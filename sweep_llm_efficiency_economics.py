#!/usr/bin/env python3
"""SLURM-friendly sweep of economic LLM productivity scenarios."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import label as connected_components

from simulate_llm_efficiency import (
    _hardware_replica_count, _resolve_backend, hardware_purchase_cost_usd)
from simulate_llm_efficiency_economics import (
    SERVICE_TYPES, economic_config, load_economic_configuration,
    run_economic_simulation)


def write_dense_h5(path: Path, **frames: pd.DataFrame) -> None:
    """Write tabular sweep outputs without requiring the optional PyTables package."""
    with h5py.File(path, "w") as handle:
        for name, frame in frames.items():
            group = handle.create_group(name)
            group.attrs["columns"] = json.dumps(frame.columns.tolist())
            for column in frame:
                values = frame[column]
                if values.dtype == object or pd.api.types.is_string_dtype(
                    values
                ):
                    data = values.fillna("").astype(str).to_numpy(dtype=object)
                    group.create_dataset(
                        column,
                        data=data,
                        dtype=h5py.string_dtype("utf-8"),
                        compression="gzip",
                        compression_opts=9,
                    )
                else:
                    group.create_dataset(
                        column,
                        data=values.to_numpy(),
                        compression="gzip",
                        compression_opts=9,
                    )


def read_dense_h5(path: Path, name: str) -> pd.DataFrame:
    with h5py.File(path, "r") as handle:
        group = handle[name]
        columns = json.loads(group.attrs["columns"])
        data = {}
        for column in columns:
            values = group[column][()]
            if values.dtype.kind == "O":
                values = np.asarray(
                    [
                        value.decode() if isinstance(value, bytes) else value
                        for value in values
                    ]
                )
            data[column] = values
    return pd.DataFrame(data, columns=columns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("llm_efficiency_economics_config.yaml"),
    )
    parser.add_argument(
        "--profile", help="Run one named company profile (for SLURM arrays)."
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge profile outputs and generate analysis plots.",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Use the dense_grid YAML section and compressed HDF5 results.",
    )
    parser.add_argument(
        "--replot-cache",
        action="store_true",
        help="Regenerate dense plots from dense_plot_cache.h5 without merging sweep results.",
    )
    parser.add_argument(
        "--merge-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Concurrent company-profile reducers for a dense merge.",
    )
    parser.add_argument(
        "--backend", choices=["numba", "numpy", "auto"], default="numba"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("llm_efficiency_economics_sweep"),
    )
    return parser.parse_args()


def onprem_budget_per_person_month_usd(config, economics: dict) -> float:
    """Normalize funded on-prem CAPEX and applicable IT support to a monthly FTE cap."""
    replicas, budget_scale = _hardware_replica_count(
        config, "onprem_full_capacity"
    )
    capex_usd = (
        hardware_purchase_cost_usd(
            config.hardware_calibration, "onprem_full_capacity"
        )
        * replicas
        * budget_scale
    )
    it_usd_per_month = (
        config.it_support_monthly_cost_eur / float(economics["eur_per_usd"])
        if config.it_support_enabled
        and config.users <= config.it_support_max_users
        else 0.0
    )
    return (
        capex_usd / (config.users * config.years * 12.0)
        + it_usd_per_month / config.users
    )


def run_profile(
    values: dict,
    profile: str,
    output_dir: Path,
    backend: str,
    dense: bool = False,
) -> None:
    sweep = values["dense_grid"] if dense else values["sweep"]
    mix = values["sweep"]["company_profiles"][profile]
    npv_frames, cashflow_frames = [], []
    seed = int(sweep["seed"])
    index = 0

    def append_results(
        config, service_types, budget_usd, budget_source, confidential_fraction
    ):
        nonlocal index
        npv, cashflows = run_economic_simulation(values, config, service_types)
        for frame in (npv, cashflows):
            frame["company_profile"] = profile
            frame["users"] = config.users
            frame["service_budget_per_person_usd"] = (
                config.max_monthly_service_budget_usd / config.users
                if config.max_monthly_service_budget_usd is not None
                else np.nan
            )
            frame["hardware_budget_usd"] = (
                config.max_upfront_hardware_budget_usd
            )
            frame["confidential_document_fraction"] = confidential_fraction
            frame["maximum_budget_per_person_month_usd"] = budget_usd
            frame["budget_source"] = budget_source
        npv_frames.append(npv)
        cashflow_frames.append(cashflows)
        index += 1

    confidential_fractions = sweep.get(
        "confidential_document_fractions", [0.0]
    )
    if dense:
        for users in sweep["headcounts"]:
            for confidential_fraction in confidential_fractions:
                for service_budget in sweep["service_budgets_usd"]:
                    config = economic_config(
                        values,
                        users=int(users),
                        years=float(sweep["years"]),
                        resolution_months=int(sweep["resolution_months"]),
                        runs=int(sweep["runs"]),
                        seed=seed + index * 1009,
                        backend=backend,
                        employee_mix=mix,
                        service_budget_per_person_usd=float(service_budget),
                        confidential_document_fraction=float(
                            confidential_fraction
                        ),
                    )
                    append_results(
                        config,
                        ("cloud", "european_service", "cloud_oss_europe"),
                        float(service_budget),
                        "cloud_service_cap",
                        confidential_fraction,
                    )
                for hardware_budget in sweep["hardware_budgets_usd"]:
                    config = economic_config(
                        values,
                        users=int(users),
                        years=float(sweep["years"]),
                        resolution_months=int(sweep["resolution_months"]),
                        runs=int(sweep["runs"]),
                        seed=seed + index * 1009,
                        backend=backend,
                        employee_mix=mix,
                        hardware_budget_usd=float(hardware_budget),
                        confidential_document_fraction=float(
                            confidential_fraction
                        ),
                    )
                    append_results(
                        config,
                        ("in_prem",),
                        onprem_budget_per_person_month_usd(
                            config, values["economics"]
                        ),
                        "onprem_capex_plus_it",
                        confidential_fraction,
                    )
    else:
        service_types = tuple(sweep.get("service_types", SERVICE_TYPES))
        for users in sweep["headcounts"]:
            for confidential_fraction in confidential_fractions:
                for service_budget in sweep["service_budgets_usd"]:
                    for hardware_budget in sweep["hardware_budgets_usd"]:
                        config = economic_config(
                            values,
                            users=int(users),
                            years=float(sweep["years"]),
                            resolution_months=int(sweep["resolution_months"]),
                            runs=int(sweep["runs"]),
                            seed=seed + index * 1009,
                            backend=backend,
                            employee_mix=mix,
                            service_budget_per_person_usd=float(
                                service_budget
                            ),
                            hardware_budget_usd=float(hardware_budget),
                            confidential_document_fraction=float(
                                confidential_fraction
                            ),
                        )
                        append_results(
                            config,
                            service_types,
                            float(service_budget),
                            "cloud_service_cap",
                            confidential_fraction,
                        )
    output_dir.mkdir(parents=True, exist_ok=True)
    npv = pd.concat(npv_frames, ignore_index=True)
    cashflows = pd.concat(cashflow_frames, ignore_index=True)
    if dense:
        write_dense_h5(
            output_dir / "dense_results.h5",
            npv_by_horizon=npv,
            cashflows_final_horizon=cashflows,
        )
    else:
        npv.to_csv(output_dir / "npv_by_horizon.csv", index=False)
        cashflows.to_csv(
            output_dir / "cashflows_final_horizon.csv", index=False
        )


def _final_npv(npv: pd.DataFrame) -> pd.DataFrame:
    dimensions = [
        "run",
        "service_type",
        "scenario",
        "company_profile",
        "users",
        "maximum_budget_per_person_month_usd",
    ]
    if "confidential_document_fraction" in npv:
        dimensions.append("confidential_document_fraction")
    return npv[
        npv["period"] == npv.groupby(dimensions)["period"].transform("max")
    ].copy()


def plot_histograms(final: pd.DataFrame, output_dir: Path) -> None:
    for fixed, stacked in (
        ("users", "company_profile"),
        ("company_profile", "users"),
    ):
        for value, panel in final.groupby(fixed, sort=True):
            groups = list(panel.groupby(stacked, sort=True))
            fig, axes = plt.subplots(
                len(groups), 1, figsize=(8, 2.6 * len(groups)), sharex=True
            )
            axes = np.atleast_1d(axes)
            for axis, (name, data) in zip(axes, groups, strict=True):
                for (service, scenario), values in data.groupby(
                    ["service_type", "scenario"]
                ):
                    axis.hist(
                        values["npv_eur"],
                        bins=30,
                        alpha=0.45,
                        label=f"{service}: {scenario}",
                    )
                axis.set_title(f"{stacked} = {name}")
                axis.set_ylabel("runs")
                axis.legend(fontsize=7)
            axes[-1].set_xlabel("NPV (EUR)")
            fig.suptitle(f"NPV distributions; {fixed} = {value}")
            fig.tight_layout()
            fig.savefig(
                output_dir / f"npv_hist_fixed_{fixed}_{value}.png", dpi=160
            )
            plt.close(fig)


def plot_cashflows(cashflows: pd.DataFrame, output_dir: Path) -> None:
    for stacked in ("company_profile", "users"):
        groups = list(cashflows.groupby(stacked, sort=True))
        fig, axes = plt.subplots(
            len(groups), 1, figsize=(8, 2.5 * len(groups)), sharex=True
        )
        axes = np.atleast_1d(axes)
        for axis, (name, data) in zip(axes, groups, strict=True):
            averages = data.groupby(
                ["month", "service_type", "scenario"], as_index=False
            )["cashflow_eur"].mean()
            for (service, scenario), series in averages.groupby(
                ["service_type", "scenario"]
            ):
                axis.plot(
                    series["month"],
                    series["cashflow_eur"],
                    marker="o",
                    label=f"{service}: {scenario}",
                )
            axis.set_title(f"{stacked} = {name}")
            axis.set_ylabel("mean cashflow (EUR)")
            axis.legend(fontsize=7)
        axes[-1].set_xlabel("month")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"average_cashflows_stacked_by_{stacked}.png", dpi=160
        )
        plt.close(fig)


def plot_budget_3d(final: pd.DataFrame, output_dir: Path) -> None:
    maxima = final.groupby(
        ["service_type", "hardware_budget_usd"], as_index=False
    ).agg(max_npv_eur=("npv_eur", "max"))
    fig = plt.figure(figsize=(7, 5))
    axis = fig.add_subplot(projection="3d")
    labels = sorted(maxima["service_type"].unique())
    x = (
        maxima["service_type"]
        .map({name: index for index, name in enumerate(labels)})
        .to_numpy()
    )
    budget_depth = max(float(maxima["hardware_budget_usd"].max()) * 0.03, 1.0)
    axis.bar3d(
        x,
        maxima["hardware_budget_usd"],
        np.zeros(len(x)),
        0.55,
        budget_depth,
        maxima["max_npv_eur"],
        shade=True,
    )
    axis.set_xticks(range(len(labels)), labels, rotation=15)
    axis.set_ylabel("hardware budget (USD)")
    axis.set_zlabel("maximum NPV (EUR)")
    fig.tight_layout()
    fig.savefig(
        output_dir / "maximum_npv_by_service_and_budget_3d.png", dpi=160
    )
    plt.close(fig)


_DENSE_DIMENSIONS = [
    "company_profile",
    "scenario",
    "service_type",
    "users",
    "confidential_document_fraction",
    "maximum_budget_per_person_month_usd",
]
_DENSE_PAIRS = (
    ("users", "confidential_document_fraction"),
    ("users", "maximum_budget_per_person_month_usd"),
    ("confidential_document_fraction", "maximum_budget_per_person_month_usd"),
)


def dense_npv_summary(final: pd.DataFrame) -> pd.DataFrame:
    return (
        final.groupby(_DENSE_DIMENSIONS, as_index=False)["npv_eur"]
        .agg(
            min_npv_eur="min",
            mean_npv_eur="mean",
            median_npv_eur="median",
            max_npv_eur="max",
            stddev_npv_eur="std",
            p10_npv_eur=lambda values: values.quantile(0.10),
            p90_npv_eur=lambda values: values.quantile(0.90),
            positive_npv_share=lambda values: (values > 0).mean(),
        )
        .fillna({"stddev_npv_eur": 0.0})
    )


def dense_pair_selections(summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    selections = {}
    for x_name, y_name in _DENSE_PAIRS:
        selections[f"{x_name}_vs_{y_name}"] = summary.loc[
            summary.groupby(["company_profile", "scenario", x_name, y_name])[
                "mean_npv_eur"
            ].idxmax()
        ].reset_index(drop=True)
    return selections


def apply_dense_axis_scaling(axis, x_name: str, y_name: str) -> None:
    if x_name in ("users", "maximum_budget_per_person_month_usd"):
        axis.set_xscale("log")
    if y_name in ("users", "maximum_budget_per_person_month_usd"):
        axis.set_yscale("log")


def dense_axis_label(name: str) -> str:
    if name == "confidential_document_fraction":
        return "confidential-work exposure scale"
    return name.replace("_", " ")


def plot_dense_best_service_3d(
    selections: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    """Plot each dense-grid axis pairing after maximizing mean NPV by service."""
    colors = {
        "cloud": "#1f77b4",
        "european_service": "#9467bd",
        "in_prem": "#ff7f0e",
        "cloud_oss_europe": "#2ca02c",
    }
    for label, best in selections.items():
        x_name, y_name = label.split("_vs_")
        profiles = sorted(best["company_profile"].unique())
        scenarios = sorted(best["scenario"].unique())
        fig = plt.figure(figsize=(6.2 * len(scenarios), 3.6 * len(profiles)))
        for row, profile in enumerate(profiles):
            for column, scenario in enumerate(scenarios):
                axis = fig.add_subplot(
                    len(profiles),
                    len(scenarios),
                    row * len(scenarios) + column + 1,
                    projection="3d",
                )
                panel = best[
                    (best["company_profile"] == profile)
                    & (best["scenario"] == scenario)
                ]
                axis.scatter(
                    panel[x_name],
                    panel[y_name],
                    panel["mean_npv_eur"],
                    c=panel["service_type"].map(colors),
                    s=22,
                )
                axis.set_title(f"{profile}; {scenario}", fontsize=8)
                axis.set_xlabel(dense_axis_label(x_name), fontsize=7)
                axis.set_ylabel(dense_axis_label(y_name), fontsize=7)
                axis.set_zlabel("maximum mean NPV (EUR)", fontsize=7)
                apply_dense_axis_scaling(axis, x_name, y_name)
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=name,
                markerfacecolor=color,
                markersize=7,
            )
            for name, color in colors.items()
        ]
        fig.legend(
            handles=handles,
            loc="upper right",
            title="best service type",
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(
            output_dir / f"dense_maximum_npv_{x_name}_vs_{y_name}_3d.png",
            dpi=170,
        )
        plt.close(fig)


def plot_dense_best_service_2d(
    selections: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    """Plot contour fields of NPV with service-specific fixed-point markers."""
    markers = {
        "cloud": "o",
        "european_service": "D",
        "in_prem": "s",
        "cloud_oss_europe": "^",
    }
    all_selected = pd.concat(selections.values(), ignore_index=True)
    low, high = (
        all_selected["mean_npv_eur"].min(),
        all_selected["mean_npv_eur"].max(),
    )
    norm = plt.Normalize(low, high if high > low else low + 1.0)
    for label, best in selections.items():
        x_name, y_name = label.split("_vs_")
        profiles = sorted(best["company_profile"].unique())
        scenarios = sorted(best["scenario"].unique())
        fig, axes = plt.subplots(
            len(profiles),
            len(scenarios),
            figsize=(5.5 * len(scenarios), 3.3 * len(profiles)),
            squeeze=False,
        )
        for row, profile in enumerate(profiles):
            for column, scenario in enumerate(scenarios):
                axis = axes[row, column]
                panel = best[
                    (best["company_profile"] == profile)
                    & (best["scenario"] == scenario)
                ]
                if (
                    len(panel) >= 3
                    and panel[x_name].nunique() >= 2
                    and panel[y_name].nunique() >= 2
                ):
                    axis.tricontourf(
                        panel[x_name],
                        panel[y_name],
                        panel["mean_npv_eur"],
                        levels=16,
                        cmap="viridis",
                        norm=norm,
                    )
                for service_type, values in panel.groupby("service_type"):
                    axis.scatter(
                        values[x_name],
                        values[y_name],
                        color="black",
                        marker=markers[service_type],
                        s=35,
                        edgecolors="white",
                        linewidths=0.4,
                        zorder=3,
                    )
                axis.set_title(f"{profile}; {scenario}", fontsize=8)
                axis.set_xlabel(dense_axis_label(x_name), fontsize=7)
                axis.set_ylabel(dense_axis_label(y_name), fontsize=7)
                apply_dense_axis_scaling(axis, x_name, y_name)
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="black",
                linestyle="",
                label=name,
                markersize=6,
            )
            for name, marker in markers.items()
        ]
        fig.legend(
            handles=handles,
            loc="upper right",
            title="best service type",
            fontsize=8,
        )
        fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis"),
            ax=axes.ravel().tolist(),
            label="maximum mean NPV (EUR)",
        )
        fig.subplots_adjust(right=0.84, hspace=0.42, wspace=0.30)
        fig.savefig(
            output_dir / f"dense_maximum_npv_{x_name}_vs_{y_name}_2d.png",
            dpi=170,
        )
        plt.close(fig)


def _family_best(
    summary: pd.DataFrame,
    x_name: str,
    y_name: str,
    service_types: tuple[str, ...],
) -> pd.DataFrame:
    candidates = summary[summary["service_type"].isin(service_types)]
    return candidates.loc[
        candidates.groupby(["company_profile", "scenario", x_name, y_name])[
            "mean_npv_eur"
        ].idxmax()
    ]


def plot_service_contours(
    axis, panel: pd.DataFrame, x_name: str, y_name: str, zero_floor: float
) -> None:
    """Shade and outline the on-prem winning region without competing with NPV colors."""
    codes = {
        "cloud": 0,
        "european_service": 1,
        "in_prem": 2,
        "cloud_oss_europe": 3,
    }
    points = panel[(panel[x_name] > 0)].copy()
    points[y_name] = points[y_name].where(points[y_name] > 0, zero_floor)
    if points.empty:
        return
    x_values, y_values = points[x_name].to_numpy(float), points[
        y_name
    ].to_numpy(float)
    x_grid = (
        np.geomspace(x_values.min(), x_values.max(), 60)
        if x_name in ("users", "maximum_budget_per_person_month_usd")
        else np.linspace(x_values.min(), x_values.max(), 60)
    )
    y_grid = (
        np.geomspace(y_values.min(), y_values.max(), 45)
        if y_name in ("users", "maximum_budget_per_person_month_usd")
        else np.linspace(y_values.min(), y_values.max(), 45)
    )
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    scaled_x = (
        np.log10(x_values)
        if x_name in ("users", "maximum_budget_per_person_month_usd")
        else x_values
    )
    scaled_y = (
        np.log10(y_values)
        if y_name in ("users", "maximum_budget_per_person_month_usd")
        else y_values
    )
    query_x = (
        np.log10(grid_x)
        if x_name in ("users", "maximum_budget_per_person_month_usd")
        else grid_x
    )
    query_y = (
        np.log10(grid_y)
        if y_name in ("users", "maximum_budget_per_person_month_usd")
        else grid_y
    )
    nearest = (
        (query_x[..., None] - scaled_x) ** 2
        + (query_y[..., None] - scaled_y) ** 2
    ).argmin(axis=2)
    regions = np.array([codes[name] for name in points["service_type"]])[
        nearest
    ]
    onprem_mask = (regions == codes["in_prem"]).astype(float)
    if onprem_mask.min() != onprem_mask.max():
        shade = axis.contourf(
            grid_x,
            grid_y,
            onprem_mask,
            levels=[0.5, 1.5],
            colors=["#d9d9d9"],
            alpha=0.26,
            zorder=3,
        )
        shade.set_edgecolor("none")
        axis.contour(
            grid_x,
            grid_y,
            onprem_mask,
            levels=[0.5],
            colors="#8c8c8c",
            linestyles="solid",
            linewidths=0.65,
            zorder=4,
        )
        components, component_count = connected_components(onprem_mask)
        minimum_area = max(80, onprem_mask.size * 0.08)
        for component in range(1, component_count + 1):
            rows, columns = np.where(components == component)
            if len(rows) < minimum_area:
                continue
            center_row, center_column = int(np.median(rows)), int(
                np.median(columns)
            )
            axis.text(
                grid_x[center_row, center_column],
                grid_y[center_row, center_column],
                "on-prem",
                color="#666666",
                fontsize=7,
                ha="center",
                va="center",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.55,
                    "edgecolor": "none",
                },
                zorder=5,
            )


def draw_dense_contour_panel(
    axis,
    summary: pd.DataFrame,
    panel: pd.DataFrame,
    scenario: str,
    profile: str,
    x_name: str,
    y_name: str,
    norm,
    title: str | None = None,
    show_uncertainty: bool = True,
) -> None:
    cloud_types = ("cloud", "european_service", "cloud_oss_europe")
    for family in (cloud_types, ("in_prem",)):
        values = _family_best(
            summary[summary["scenario"] == scenario], x_name, y_name, family
        )
        values = values[values["company_profile"] == profile]
        values = values[(values[x_name] > 0) & (values[y_name] > 0)]
        if (
            len(values) >= 3
            and values[x_name].nunique() >= 2
            and values[y_name].nunique() >= 2
        ):
            axis.tricontourf(
                values[x_name],
                values[y_name],
                values["mean_npv_eur"],
                levels=17,
                cmap="RdBu_r",
                norm=norm,
                alpha=0.72,
            )
    positive_y = panel.loc[panel[y_name] > 0, y_name]
    zero_floor = positive_y.min() / 1.8 if not positive_y.empty else 1.0
    plot_service_contours(axis, panel, x_name, y_name, zero_floor)
    if show_uncertainty:
        uncertainty = panel["stddev_npv_eur"] / panel[
            "mean_npv_eur"
        ].abs().clip(lower=1.0)
        uncertain = panel[uncertainty > 0.5]
        if not uncertain.empty:
            axis.scatter(
                uncertain[x_name],
                uncertain[y_name].where(uncertain[y_name] > 0, zero_floor),
                marker="o",
                facecolors="none",
                edgecolors="#f2c14e",
                linewidths=1.1,
                s=62,
                zorder=5,
            )
    shares = panel["service_type"].value_counts(normalize=True)
    share_label = " · ".join(
        f"{name.replace('_', ' ')} {share:.0%}"
        for name, share in shares.items()
    )
    axis.text(
        0.02,
        0.02,
        f"winner shares: {share_label}",
        transform=axis.transAxes,
        fontsize=6.5,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
    )
    if title is not None:
        axis.set_title(title, fontsize=9)
    axis.set_xlabel(dense_axis_label(x_name), fontsize=7)
    axis.set_ylabel(dense_axis_label(y_name), fontsize=7)
    apply_dense_axis_scaling(axis, x_name, y_name)


def plot_dense_contours(
    summary: pd.DataFrame,
    selections: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Publication-oriented 2D views: separate service-family contours and fixed points."""
    profiles = sorted(summary["company_profile"].unique())
    scenarios = sorted(summary["scenario"].unique())
    individual_dir = output_dir / "contours_by_organization"
    individual_dir.mkdir(parents=True, exist_ok=True)
    for label, selected in selections.items():
        x_name, y_name = label.split("_vs_")
        for scenario in scenarios:
            scenario_summary = summary[summary["scenario"] == scenario]
            scenario_selected = selected[selected["scenario"] == scenario]
            limit = np.nanquantile(
                np.abs(scenario_summary["mean_npv_eur"]), 0.98
            )
            norm = matplotlib.colors.TwoSlopeNorm(
                vcenter=0.0, vmin=-max(limit, 1.0), vmax=max(limit, 1.0)
            )
            fig, axes = plt.subplots(3, 2, figsize=(11, 10), squeeze=False)
            for axis, profile in zip(axes.ravel(), profiles, strict=False):
                panel = selected[
                    (selected["company_profile"] == profile)
                    & (selected["scenario"] == scenario)
                ]
                draw_dense_contour_panel(
                    axis,
                    summary,
                    panel,
                    scenario,
                    profile,
                    x_name,
                    y_name,
                    norm,
                    title=profile.replace("_", " "),
                )
            for axis in axes.ravel()[len(profiles) :]:
                axis.set_visible(False)
            handles = [
                matplotlib.patches.Patch(
                    facecolor="#d9d9d9",
                    edgecolor="#8c8c8c",
                    alpha=0.45,
                    label="on-prem winning region",
                )
            ]
            uncertainty = scenario_selected[
                "stddev_npv_eur"
            ] / scenario_selected["mean_npv_eur"].abs().clip(lower=1.0)
            if (uncertainty > 0.5).any():
                handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        markerfacecolor="none",
                        markeredgecolor="#f2c14e",
                        linestyle="",
                        label="high NPV uncertainty",
                        markersize=7,
                    )
                )
            if handles:
                fig.legend(
                    handles=handles,
                    loc="upper center",
                    bbox_to_anchor=(0.45, 0.955),
                    ncol=3,
                    fontsize=7,
                    frameon=False,
                )
            color_axis = fig.add_axes([0.88, 0.16, 0.018, 0.65])
            fig.colorbar(
                matplotlib.cm.ScalarMappable(norm=norm, cmap="RdBu_r"),
                cax=color_axis,
                label="mean NPV (EUR; blue = loss, red = gain)",
            )
            fig.suptitle(
                f"{scenario.replace('_', ' ')} · NPV contours with on-prem winning region",
                y=0.995,
                fontsize=11,
            )
            fig.subplots_adjust(
                top=0.86, right=0.84, bottom=0.08, hspace=0.38, wspace=0.25
            )
            fig.savefig(
                output_dir / f"dense_npv_contour_{label}_{scenario}.png",
                dpi=180,
            )
            plt.close(fig)
            for profile in profiles:
                figure, axis = plt.subplots(figsize=(6.2, 4.8))
                panel = selected[
                    (selected["company_profile"] == profile)
                    & (selected["scenario"] == scenario)
                ]
                draw_dense_contour_panel(
                    axis,
                    summary,
                    panel,
                    scenario,
                    profile,
                    x_name,
                    y_name,
                    norm,
                    title=profile.replace("_", " "),
                )
                figure.colorbar(
                    matplotlib.cm.ScalarMappable(norm=norm, cmap="RdBu_r"),
                    ax=axis,
                    label="mean NPV (EUR; blue = loss, red = gain)",
                )
                figure.tight_layout()
                figure.savefig(
                    individual_dir
                    / f"dense_npv_contour_{profile}_{label}_{scenario}.png",
                    dpi=180,
                )
                plt.close(figure)


def plot_dense_pairwise_contours(
    summary: pd.DataFrame,
    selections: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Write one two-panel contour figure per organisation and economic scenario."""
    left_label = "users_vs_maximum_budget_per_person_month_usd"
    right_label = "users_vs_confidential_document_fraction"
    pair_dir = output_dir / "contours_by_organization" / "paired"
    pair_dir.mkdir(parents=True, exist_ok=True)
    profiles = sorted(summary["company_profile"].unique())
    scenarios = sorted(summary["scenario"].unique())
    for scenario in scenarios:
        for profile in profiles:
            left = selections[left_label][
                (selections[left_label]["company_profile"] == profile)
                & (selections[left_label]["scenario"] == scenario)
            ]
            right = selections[right_label][
                (selections[right_label]["company_profile"] == profile)
                & (selections[right_label]["scenario"] == scenario)
            ]
            values = pd.concat([left["mean_npv_eur"], right["mean_npv_eur"]])
            limit = np.nanquantile(np.abs(values), 0.98)
            norm = matplotlib.colors.TwoSlopeNorm(
                vcenter=0.0, vmin=-max(limit, 1.0), vmax=max(limit, 1.0)
            )
            figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), squeeze=False)
            draw_dense_contour_panel(
                axes[0, 0],
                summary,
                left,
                scenario,
                profile,
                "users",
                "maximum_budget_per_person_month_usd",
                norm,
                title="users vs budget",
            )
            draw_dense_contour_panel(
                axes[0, 1],
                summary,
                right,
                scenario,
                profile,
                "users",
                "confidential_document_fraction",
                norm,
                title="users vs confidential-work exposure",
            )
            figure.subplots_adjust(
                left=0.08, right=0.84, bottom=0.13, top=0.91, wspace=0.32
            )
            color_axis = figure.add_axes([0.89, 0.16, 0.018, 0.70])
            figure.colorbar(
                matplotlib.cm.ScalarMappable(norm=norm, cmap="RdBu_r"),
                cax=color_axis,
                label="mean NPV (EUR; blue = loss, red = gain)",
            )
            figure.savefig(
                pair_dir / f"dense_npv_contour_pair_{profile}_{scenario}.png",
                dpi=180,
            )
            plt.close(figure)


def plot_dense_pairwise_contour_vertical(
    summary: pd.DataFrame,
    selections: dict[str, pd.DataFrame],
    output_dir: Path,
    profile: str = "mixed",
    scenario: str = "labor_cost_savings",
) -> None:
    """Write the requested vertically stacked pair with a horizontal colorbar."""
    left_label = "users_vs_maximum_budget_per_person_month_usd"
    right_label = "users_vs_confidential_document_fraction"
    left = selections[left_label][
        (selections[left_label]["company_profile"] == profile)
        & (selections[left_label]["scenario"] == scenario)
    ]
    right = selections[right_label][
        (selections[right_label]["company_profile"] == profile)
        & (selections[right_label]["scenario"] == scenario)
    ]
    values = pd.concat([left["mean_npv_eur"], right["mean_npv_eur"]])
    limit = np.nanquantile(np.abs(values), 0.98)
    norm = matplotlib.colors.TwoSlopeNorm(
        vcenter=0.0, vmin=-max(limit, 1.0), vmax=max(limit, 1.0)
    )
    figure, axes = plt.subplots(2, 1, figsize=(6.2, 9.0), squeeze=False)
    draw_dense_contour_panel(
        axes[0, 0],
        summary,
        left,
        scenario,
        profile,
        "users",
        "maximum_budget_per_person_month_usd",
        norm,
        title="users vs budget",
        show_uncertainty=False,
    )
    draw_dense_contour_panel(
        axes[1, 0],
        summary,
        right,
        scenario,
        profile,
        "users",
        "confidential_document_fraction",
        norm,
        title="users vs confidential-work exposure",
        show_uncertainty=False,
    )
    figure.subplots_adjust(
        left=0.14, right=0.96, bottom=0.14, top=0.96, hspace=0.38
    )
    color_axis = figure.add_axes([0.18, 0.055, 0.74, 0.022])
    figure.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap="RdBu_r"),
        cax=color_axis,
        orientation="horizontal",
        label="mean NPV (EUR; blue = loss, red = gain)",
    )
    pair_dir = output_dir / "contours_by_organization" / "paired"
    pair_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        pair_dir / f"dense_npv_contour_pair_{profile}_{scenario}_vertical.png",
        dpi=180,
    )
    plt.close(figure)


def fixed_budget_selection(
    summary: pd.DataFrame, budget_usd: float
) -> pd.DataFrame:
    """Choose each route's closest simulated affordability cap, then its best service."""
    route_keys = [
        "company_profile",
        "scenario",
        "users",
        "confidential_document_fraction",
        "service_type",
    ]
    candidates = summary.copy()
    candidates["budget_distance"] = (
        candidates["maximum_budget_per_person_month_usd"] - budget_usd
    ).abs()
    closest = candidates.loc[
        candidates.groupby(route_keys)["budget_distance"].idxmin()
    ]
    point_keys = route_keys[:-1]
    return closest.loc[
        closest.groupby(point_keys)["mean_npv_eur"].idxmax()
    ].reset_index(drop=True)


def plot_dense_fixed_budget_contours(
    summary: pd.DataFrame, dense_grid: dict, output_dir: Path
) -> None:
    """Plot users × exposure at fixed affordability caps, without maximizing over budget."""
    profiles = sorted(summary["company_profile"].unique())
    scenarios = sorted(summary["scenario"].unique())
    for budget_usd in dense_grid.get("fixed_budget_slices_usd", []):
        selected = fixed_budget_selection(summary, float(budget_usd))
        for scenario in scenarios:
            panel_scenario = selected[selected["scenario"] == scenario]
            limit = np.nanquantile(
                np.abs(panel_scenario["mean_npv_eur"]), 0.98
            )
            norm = matplotlib.colors.TwoSlopeNorm(
                vcenter=0.0, vmin=-max(limit, 1.0), vmax=max(limit, 1.0)
            )
            fig, axes = plt.subplots(3, 2, figsize=(11, 10), squeeze=False)
            for axis, profile in zip(axes.ravel(), profiles, strict=False):
                panel = panel_scenario[
                    panel_scenario["company_profile"] == profile
                ]
                if len(panel) >= 3:
                    axis.tricontourf(
                        panel["users"],
                        panel["confidential_document_fraction"],
                        panel["mean_npv_eur"],
                        levels=17,
                        cmap="RdBu_r",
                        norm=norm,
                        alpha=0.72,
                    )
                plot_service_contours(
                    axis,
                    panel,
                    "users",
                    "confidential_document_fraction",
                    0.001,
                )
                axis.set_title(profile.replace("_", " "), fontsize=9)
                axis.set_xlabel("users", fontsize=7)
                axis.set_ylabel(
                    dense_axis_label("confidential_document_fraction"),
                    fontsize=7,
                )
                axis.set_xscale("log")
            for axis in axes.ravel()[len(profiles) :]:
                axis.set_visible(False)
            fig.legend(
                handles=[
                    matplotlib.patches.Patch(
                        facecolor="#d9d9d9",
                        edgecolor="#8c8c8c",
                        alpha=0.45,
                        label="on-prem winning region",
                    )
                ],
                loc="upper center",
                bbox_to_anchor=(0.45, 0.955),
                fontsize=7,
                frameon=False,
            )
            color_axis = fig.add_axes([0.88, 0.16, 0.018, 0.65])
            fig.colorbar(
                matplotlib.cm.ScalarMappable(norm=norm, cmap="RdBu_r"),
                cax=color_axis,
                label="mean NPV (EUR; blue = loss, red = gain)",
            )
            fig.suptitle(
                f"{scenario.replace('_', ' ')} · fixed affordability cap ${budget_usd:g}/person/month",
                y=0.995,
                fontsize=11,
            )
            fig.subplots_adjust(
                top=0.86, right=0.84, bottom=0.08, hspace=0.38, wspace=0.25
            )
            fig.savefig(
                output_dir
                / f"dense_npv_fixed_budget_{budget_usd:g}_{scenario}.png",
                dpi=180,
            )
            plt.close(fig)


def plot_selected_cashflow_fans(
    selections: dict[str, pd.DataFrame],
    cashflows: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Show mean and P10-P90 cashflow paths for the highest-NPV fixed point per panel."""
    for label, selection in selections.items():
        series = cashflows[label]
        profiles = sorted(selection["company_profile"].unique())
        scenarios = sorted(selection["scenario"].unique())
        fig, axes = plt.subplots(3, 2, figsize=(11, 10), squeeze=False)
        for axis, profile in zip(axes.ravel(), profiles, strict=False):
            for scenario in scenarios:
                chosen = selection[
                    (selection["company_profile"] == profile)
                    & (selection["scenario"] == scenario)
                ]
                if chosen.empty:
                    continue
                chosen = chosen.loc[chosen["mean_npv_eur"].idxmax()]
                keys = {name: chosen[name] for name in _DENSE_DIMENSIONS}
                path = series.copy()
                for name, value in keys.items():
                    path = path[path[name] == value]
                path = path.sort_values("month")
                axis.plot(
                    path["month"],
                    path["mean_cashflow_eur"],
                    label=f"{scenario.replace('_', ' ')} · {chosen['service_type'].replace('_', ' ')}",
                )
                axis.fill_between(
                    path["month"],
                    path["p10_cashflow_eur"],
                    path["p90_cashflow_eur"],
                    alpha=0.18,
                )
            axis.axhline(0, color="black", linewidth=0.6)
            axis.set_title(profile.replace("_", " "), fontsize=9)
            axis.set_xlabel("month")
            axis.set_ylabel("cashflow (EUR)")
            axis.legend(fontsize=6)
        for axis in axes.ravel()[len(profiles) :]:
            axis.set_visible(False)
        fig.suptitle(
            f"Selected maximum-NPV cashflow paths · P10–P90 bands · {label.replace('_', ' ')}",
            y=0.98,
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(
            output_dir / f"dense_selected_cashflow_fans_{label}.png", dpi=180
        )
        plt.close(fig)


def selected_cashflow_summary(
    cashflows: pd.DataFrame, selection: pd.DataFrame
) -> pd.DataFrame:
    keys = _DENSE_DIMENSIONS
    selected = selection[keys].drop_duplicates()
    rows = cashflows.merge(selected, on=keys, how="inner")
    return (
        rows.groupby(keys + ["month", "horizon_month"], as_index=False)[
            "cashflow_eur"
        ]
        .agg(
            min_cashflow_eur="min",
            mean_cashflow_eur="mean",
            median_cashflow_eur="median",
            max_cashflow_eur="max",
            stddev_cashflow_eur="std",
            p10_cashflow_eur=lambda values: values.quantile(0.10),
            p90_cashflow_eur=lambda values: values.quantile(0.90),
        )
        .fillna({"stddev_cashflow_eur": 0.0})
    )


def write_dense_plot_cache(
    output_dir: Path, final: pd.DataFrame, cashflows: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    summary = dense_npv_summary(final)
    selections = dense_pair_selections(summary)
    frames: dict[str, pd.DataFrame] = {"npv_summary": summary}
    for label, selection in selections.items():
        frames[f"selected_npv_{label}"] = selection
        frames[f"selected_cashflows_{label}"] = selected_cashflow_summary(
            cashflows, selection
        )
    write_dense_h5(output_dir / "dense_plot_cache.h5", **frames)
    return selections


def profile_dense_plot_cache(path: Path) -> dict[str, pd.DataFrame]:
    """Reduce one profile independently so the parent never receives raw HDF5 tables."""
    final = _final_npv(read_dense_h5(path, "npv_by_horizon"))
    summary = dense_npv_summary(final)
    selections = dense_pair_selections(summary)
    cashflows = read_dense_h5(path, "cashflows_final_horizon")
    frames: dict[str, pd.DataFrame] = {"npv_summary": summary}
    for label, selection in selections.items():
        frames[f"selected_npv_{label}"] = selection
        frames[f"selected_cashflows_{label}"] = selected_cashflow_summary(
            cashflows, selection
        )
    return frames


def write_combined_dense_plot_cache(
    output_dir: Path, profile_frames: list[dict[str, pd.DataFrame]]
) -> dict[str, pd.DataFrame]:
    names = profile_frames[0].keys()
    frames = {
        name: pd.concat(
            [profile[name] for profile in profile_frames], ignore_index=True
        )
        for name in names
    }
    write_dense_h5(output_dir / "dense_plot_cache.h5", **frames)
    return {
        f"{x_name}_vs_{y_name}": frames[f"selected_npv_{x_name}_vs_{y_name}"]
        for x_name, y_name in _DENSE_PAIRS
    }


def replot_dense_cache(output_dir: Path, dense_grid: dict) -> None:
    path = output_dir / "dense_plot_cache.h5"
    if not path.exists():
        raise SystemExit(
            f"Missing dense plot cache: {path}. Run --dense --merge once to create it."
        )
    selections = {
        f"{x_name}_vs_{y_name}": read_dense_h5(
            path, f"selected_npv_{x_name}_vs_{y_name}"
        )
        for x_name, y_name in _DENSE_PAIRS
    }
    cashflows = {
        f"{x_name}_vs_{y_name}": read_dense_h5(
            path, f"selected_cashflows_{x_name}_vs_{y_name}"
        )
        for x_name, y_name in _DENSE_PAIRS
    }
    summary = read_dense_h5(path, "npv_summary")
    plot_dense_best_service_3d(selections, output_dir)
    plot_dense_contours(summary, selections, output_dir)
    plot_dense_pairwise_contours(summary, selections, output_dir)
    plot_dense_pairwise_contour_vertical(summary, selections, output_dir)
    plot_dense_fixed_budget_contours(summary, dense_grid, output_dir)
    plot_selected_cashflow_fans(selections, cashflows, output_dir)


def plot_pca_clusters(final: pd.DataFrame, output_dir: Path) -> None:
    dimensions = [
        "users",
        "service_budget_per_person_usd",
        "hardware_budget_usd",
        "service_type",
        "scenario",
        "company_profile",
    ]
    designs = final.groupby(dimensions, as_index=False)["npv_eur"].mean()
    features = pd.get_dummies(designs[dimensions], dtype=float)
    raw = features.to_numpy(float)
    matrix = (raw - raw.mean(axis=0)) / np.where(
        raw.std(axis=0) == 0.0, 1.0, raw.std(axis=0)
    )
    _, singular_values, components = np.linalg.svd(matrix, full_matrices=False)
    coordinates = matrix @ components[:2].T
    explained = singular_values**2 / max(1e-12, (singular_values**2).sum())
    cluster_count = min(4, len(designs))
    centers = matrix[
        np.linspace(0, len(designs) - 1, cluster_count, dtype=int)
    ].copy()
    for _ in range(50):
        clusters = (
            ((matrix[:, None, :] - centers[None, :, :]) ** 2)
            .sum(axis=2)
            .argmin(axis=1)
        )
        updated = np.array(
            [
                (
                    matrix[clusters == index].mean(axis=0)
                    if np.any(clusters == index)
                    else centers[index]
                )
                for index in range(cluster_count)
            ]
        )
        if np.allclose(updated, centers):
            break
        centers = updated
    result = designs[["npv_eur"]].copy()
    result["pca_1"], result["pca_2"], result["cluster"] = (
        coordinates[:, 0],
        coordinates[:, 1],
        clusters,
    )
    result.to_csv(output_dir / "npv_pca_clusters.csv", index=False)
    fig, axis = plt.subplots(figsize=(7, 5))
    scatter = axis.scatter(
        result["pca_1"],
        result["pca_2"],
        c=result["npv_eur"],
        s=28,
        cmap="viridis",
    )
    axis.set_xlabel(f"PC1 ({explained[0]:.0%})")
    axis.set_ylabel(f"PC2 ({explained[1]:.0%})")
    fig.colorbar(scatter, ax=axis, label="mean NPV (EUR)")
    fig.tight_layout()
    fig.savefig(output_dir / "npv_pca_clusters.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(
        {
            "feature": features.columns,
            "pc1_loading": components[0],
            "pc2_loading": components[1],
        }
    ).to_csv(output_dir / "npv_pca_loadings.csv", index=False)
    designs.groupby("service_type", as_index=False)["npv_eur"].mean().rename(
        columns={"npv_eur": "mean_npv_eur"}
    ).to_csv(output_dir / "npv_driver_service_type.csv", index=False)


def merge_and_analyse(output_dir: Path) -> None:
    npv_paths = sorted(output_dir.glob("*/npv_by_horizon.csv"))
    cashflow_paths = sorted(output_dir.glob("*/cashflows_final_horizon.csv"))
    if not npv_paths or not cashflow_paths:
        raise SystemExit(
            "No profile outputs found; run one or more --profile jobs first."
        )
    npv = pd.concat(
        [pd.read_csv(path) for path in npv_paths], ignore_index=True
    )
    cashflows = pd.concat(
        [pd.read_csv(path) for path in cashflow_paths], ignore_index=True
    )
    npv.to_csv(output_dir / "npv_by_horizon.csv", index=False)
    cashflows.to_csv(output_dir / "cashflows_final_horizon.csv", index=False)
    final = _final_npv(npv)
    final.to_csv(output_dir / "final_horizon_npv.csv", index=False)
    plot_histograms(final, output_dir)
    plot_cashflows(cashflows, output_dir)
    plot_budget_3d(final, output_dir)
    plot_pca_clusters(final, output_dir)


def merge_dense_and_plot(
    output_dir: Path, workers: int, dense_grid: dict
) -> None:
    paths = sorted(output_dir.glob("*/dense_results.h5"))
    if not paths:
        raise SystemExit(
            "No dense HDF5 profile outputs found; run one or more --dense --profile jobs first."
        )
    if workers <= 0:
        raise SystemExit("--merge-workers must be positive.")
    with ProcessPoolExecutor(max_workers=min(workers, len(paths))) as executor:
        profile_frames = list(executor.map(profile_dense_plot_cache, paths))
    selections = write_combined_dense_plot_cache(output_dir, profile_frames)
    plot_dense_best_service_3d(selections, output_dir)
    cache = output_dir / "dense_plot_cache.h5"
    summary = read_dense_h5(cache, "npv_summary")
    cashflows = {
        f"{x_name}_vs_{y_name}": read_dense_h5(
            cache, f"selected_cashflows_{x_name}_vs_{y_name}"
        )
        for x_name, y_name in _DENSE_PAIRS
    }
    plot_dense_contours(summary, selections, output_dir)
    plot_dense_pairwise_contours(summary, selections, output_dir)
    plot_dense_pairwise_contour_vertical(summary, selections, output_dir)
    plot_dense_fixed_budget_contours(summary, dense_grid, output_dir)
    plot_selected_cashflow_fans(selections, cashflows, output_dir)


def main() -> None:
    args = parse_args()
    values = load_economic_configuration(args.config)
    profiles = values["sweep"]["company_profiles"]
    if args.replot_cache:
        if not args.dense:
            raise SystemExit(
                "--replot-cache is currently available for --dense results only."
            )
        replot_dense_cache(args.output_dir, values["dense_grid"])
        return
    if args.merge:
        if args.dense:
            merge_dense_and_plot(
                args.output_dir, args.merge_workers, values["dense_grid"]
            )
        else:
            merge_and_analyse(args.output_dir)
        return
    if args.profile is None:
        raise SystemExit("--profile is required unless --merge is used.")
    if args.profile not in profiles:
        raise SystemExit(f"Unknown profile: {args.profile}")
    run_profile(
        values,
        args.profile,
        args.output_dir / args.profile,
        _resolve_backend(args.backend),
        args.dense,
    )


if __name__ == "__main__":
    main()
