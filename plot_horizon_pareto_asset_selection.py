#!/usr/bin/env python3
"""Plot the Pareto asset selected at each evaluation horizon in a sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from tqdm.auto import tqdm


SELECTIONS = ("low_risk", "optimum", "high_return")
SERVICE_LABELS = {
    "global_service": "Global cloud",
    "european_service": "EU cloud",
    "local_hardware": "On-prem OSS",
    "eu_cloud_oss": "Cloud OSS",
}
SERVICE_STYLES = {
    "Global cloud": ("#2563eb", "o"),
    "EU cloud": ("#059669", "s"),
    "On-prem OSS": ("#dc2626", "^"),
    "Cloud OSS": ("#7c3aed", "D"),
}
SERVICE_ORDER = tuple(SERVICE_STYLES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_folder", type=Path, help="Sweep root containing company-profile output folders.")
    parser.add_argument("--output-dir", type=Path, help="Default: <sweep_folder>/horizon_pareto_plots.")
    parser.add_argument("--profiles", help="Comma-separated profile folders to include (default: all discovered).")
    return parser.parse_args()


def discover_profiles(sweep_folder: Path, requested: str | None) -> list[Path]:
    profiles = sorted(path.parent for path in sweep_folder.rglob("portfolio_fixed_asset_manifest.csv"))
    if requested:
        names = {name.strip() for name in requested.split(",") if name.strip()}
        profiles = [path for path in profiles if path.name in names]
    if not profiles:
        raise SystemExit(f"No fixed-asset sweep outputs found below {sweep_folder}.")
    return profiles


def load_profile(profile_dir: Path) -> pd.DataFrame:
    manifest = pd.read_csv(profile_dir / "portfolio_fixed_asset_manifest.csv")
    rewards = pd.read_csv(profile_dir / "portfolio_fixed_asset_rate_changes.csv")
    risks = pd.read_csv(profile_dir / "portfolio_fixed_asset_rate_change_risk.csv")
    reward_long = rewards.melt(id_vars="month", var_name="asset_column_label", value_name="reward")
    risk_long = risks.melt(id_vars="month", var_name="asset_column_label", value_name="risk")
    values = reward_long.merge(risk_long, on=["month", "asset_column_label"], validate="one_to_one")
    values = values.merge(
        manifest[["asset_column_label", "users", "service_provider", "model", "access_plan", "hardware", "shock_combination"]],
        on="asset_column_label",
        how="left",
        validate="many_to_one",
    )
    values["company_profile"] = profile_dir.name
    values["service"] = values["service_provider"].map(SERVICE_LABELS).fillna("Other")
    return values.replace([np.inf, -np.inf], np.nan).dropna(subset=["reward", "risk"])


def efficient_frontier(points: pd.DataFrame) -> pd.DataFrame:
    ordered = points.sort_values(["risk", "reward"], ascending=[True, False])
    best_reward = -np.inf
    selected = []
    for _, row in ordered.iterrows():
        if row["reward"] > best_reward + 1e-12:
            selected.append(row)
            best_reward = row["reward"]
    return pd.DataFrame(selected)


def choose(frontier: pd.DataFrame, selection: str) -> pd.Series:
    if selection == "low_risk":
        return frontier.sort_values(["risk", "reward"], ascending=[True, False]).iloc[0]
    if selection == "high_return":
        return frontier.sort_values(["reward", "risk"], ascending=[False, True]).iloc[0]
    positive_risk = frontier[frontier["risk"] > 0.0]
    if positive_risk.empty:
        return frontier.sort_values(["reward", "risk"], ascending=[False, True]).iloc[0]
    scores = positive_risk["reward"] / positive_risk["risk"]
    return positive_risk.loc[scores.idxmax()]


def select_by_horizon(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (_, _, month), group in values.groupby(["company_profile", "users", "month"], sort=True):
        frontier = efficient_frontier(group)
        if frontier.empty:
            continue
        for selection in SELECTIONS:
            row = choose(frontier, selection).to_dict()
            row["selection"] = selection
            row["frontier_size"] = len(frontier)
            rows.append(row)
    return pd.DataFrame(rows)


def plot_selection(selected: pd.DataFrame, selection: str, output_dir: Path) -> None:
    profiles = sorted(selected["company_profile"].unique())
    figure, axes = plt.subplots(len(profiles), 1, figsize=(11, max(3.0, 2.6 * len(profiles))), sharex=True, squeeze=False)
    for axis, profile in zip(axes[:, 0], profiles, strict=True):
        values = selected[(selected["company_profile"] == profile) & (selected["selection"] == selection)].sort_values("month")
        for _, bracket_values in values.groupby("users", sort=True):
            axis.plot(bracket_values["month"], bracket_values["reward"], color="#9ca3af", linewidth=1.0, zorder=1)
        for service, service_values in values.groupby("service", sort=True):
            color, marker = SERVICE_STYLES.get(service, ("#374151", "x"))
            axis.scatter(service_values["month"], service_values["reward"], color=color, marker=marker, s=44, zorder=2)
        for _, row in values.iterrows():
            axis.annotate(row["service"], (row["month"], row["reward"]), xytext=(3, 4), textcoords="offset points", fontsize=6)
        axis.set_title(profile.replace("_", " "))
        axis.set_ylabel("Median return-rate change")
        axis.grid(axis="y", alpha=0.25)
    axes[-1, 0].set_xlabel("Evaluation horizon (months)")
    handles = [Line2D([], [], color=color, marker=marker, linestyle="", label=service) for service, (color, marker) in SERVICE_STYLES.items()]
    axes[0, 0].legend(handles=handles, title="Selected service asset", loc="best", fontsize=7, title_fontsize=8)
    figure.suptitle(f"{selection.replace('_', ' ').title()} Pareto asset by evaluation horizon", y=0.995)
    figure.tight_layout()
    figure.savefig(output_dir / f"horizon_{selection}_pareto_asset.png", dpi=180)
    plt.close(figure)


def plot_combined_selection(selected: pd.DataFrame, selection: str, output_dir: Path) -> None:
    values = selected[selected["selection"] == selection].copy()
    profiles = sorted(values["company_profile"].unique())
    colors = dict(zip(profiles, plt.get_cmap("tab10").colors, strict=False))
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for profile, profile_values in values.groupby("company_profile", sort=True):
        for _, bracket_values in profile_values.groupby("users", sort=True):
            bracket_values = bracket_values.sort_values("month")
            axis.plot(bracket_values["month"], bracket_values["reward"], color=colors[profile], linewidth=1.1, alpha=0.65)
        for service, service_values in profile_values.groupby("service", sort=True):
            _, marker = SERVICE_STYLES.get(service, ("#374151", "x"))
            axis.scatter(service_values["month"], service_values["reward"], color=colors[profile], marker=marker, s=44, zorder=2)
    profile_handles = [Line2D([], [], color=colors[profile], linewidth=2, label=profile.replace("_", " ")) for profile in profiles]
    service_handles = [Line2D([], [], color="#374151", marker=marker, linestyle="", label=service) for service, (_, marker) in SERVICE_STYLES.items()]
    profile_legend = axis.legend(handles=profile_handles, title="Company profile", loc="upper left", fontsize=7, title_fontsize=8)
    axis.add_artist(profile_legend)
    axis.legend(handles=service_handles, title="Selected asset", loc="upper right", fontsize=7, title_fontsize=8)
    axis.set_xlabel("Evaluation horizon (months)")
    axis.set_ylabel("Median return-rate change")
    axis.set_title(f"{selection.replace('_', ' ').title()} Pareto asset by horizon and company profile")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / f"horizon_{selection}_pareto_asset_combined.png", dpi=180)
    plt.close(figure)


def sankey_periods(months: pd.Series, resolution: str) -> list[float]:
    months = sorted(months.unique())
    if resolution == "monthly":
        if any(right - left != 1 for left, right in zip(months, months[1:])):
            raise SystemExit(
                "Monthly Sankeys require sweep output at one-month intervals. "
                "Re-run the sweep with portfolio.resolution_months: 1."
            )
        return months
    step = 3 if resolution == "quarterly" else 12
    periods = [month for month in months if month % step == 0]
    # A finite horizon can end between calendar boundaries (for example at
    # month 33 for a three-year quarterly sweep). Keep that terminal point so
    # the annual Sankey reaches the end of the simulated horizon.
    if resolution == "annual" and months and months[-1] not in periods:
        periods.append(months[-1])
    return periods


def plot_service_transition_sankey(selected: pd.DataFrame, output_dir: Path, resolution: str) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return
    periods = sankey_periods(selected["month"], resolution)
    if len(periods) < 2:
        return
    figure = make_subplots(
        rows=len(SELECTIONS),
        cols=1,
        specs=[[{"type": "sankey"}] for _ in SELECTIONS],
        subplot_titles=[selection.replace("_", " ").title() for selection in SELECTIONS],
    )
    service_colors = {service: style[0] for service, style in SERVICE_STYLES.items()}
    population = (
        selected[["company_profile", "users"]]
        .drop_duplicates()
        .assign(combo_key=lambda frame: frame["company_profile"] + "|" + frame["users"].astype(str))
    )
    expected_count = len(population)
    for row_number, selection in enumerate(SELECTIONS, start=1):
        values = selected[(selected["selection"] == selection) & (selected["month"].isin(periods))].copy()
        values["combo_key"] = values["company_profile"] + "|" + values["users"].astype(str)
        services = values.pivot(index="combo_key", columns="month", values="service").reindex(
            index=population["combo_key"], columns=periods
        )
        if services.isna().any().any():
            raise SystemExit(f"Incomplete {selection} selections; cannot form constant-N Sankey flows.")
        labels = [f"Month {month:g}: {service}" for month in periods for service in SERVICE_ORDER]
        indices = {label: index for index, label in enumerate(labels)}
        # Give Plotly a vertical service-rank preference while leaving the
        # horizontal layout and crossing minimisation to its snap algorithm.
        node_y = [
            0.03 + 0.94 * service_index / max(1, len(SERVICE_ORDER) - 1)
            for _ in periods
            for service_index in range(len(SERVICE_ORDER))
        ]
        flows = []
        for source_month, target_month in zip(periods, periods[1:]):
            transition_counts = services.groupby([source_month, target_month]).size()
            if transition_counts.sum() != expected_count:
                raise SystemExit(f"{selection} Sankey flow total is not constant at {expected_count} combinations.")
            for (source_service, target_service), count in transition_counts.items():
                flows.append((
                    indices[f"Month {source_month:g}: {source_service}"],
                    indices[f"Month {target_month:g}: {target_service}"],
                    count,
                    service_colors.get(target_service, "#374151"),
                ))
        figure.add_trace(
            go.Sankey(
                node={
                    "label": labels,
                    "color": [service_colors.get(label.rsplit(": ", 1)[1], "#374151") for label in labels],
                    "y": node_y,
                    "pad": 10,
                    "thickness": 14,
                },
                link={
                    "source": [flow[0] for flow in flows],
                    "target": [flow[1] for flow in flows],
                    "value": [flow[2] for flow in flows],
                    "color": [flow[3] for flow in flows],
                },
                arrangement="snap",
            ),
            row=row_number,
            col=1,
        )
    figure.update_layout(
        title=f"Service selection transitions across evaluation horizons ({resolution})",
        font_size=10,
        width=max(1250, 180 * len(periods)),
        height=1200,
    )
    stem = output_dir / f"horizon_service_transition_sankey_{resolution}"
    figure.write_html(f"{stem}.html", include_plotlyjs="cdn")
    try:
        figure.write_image(f"{stem}.png", scale=2)
    except (ImportError, ValueError):
        pass


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.sweep_folder / "horizon_pareto_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = discover_profiles(args.sweep_folder, args.profiles)
    values = pd.concat(
        [load_profile(path) for path in tqdm(profiles, desc="Loading sweep profiles", unit="profile")],
        ignore_index=True,
    )
    selected = select_by_horizon(values)
    selected.to_csv(output_dir / "horizon_pareto_asset_selections.csv", index=False)
    for selection in tqdm(SELECTIONS, desc="Writing horizon plots", unit="plot"):
        plot_selection(selected, selection, output_dir)
        plot_combined_selection(selected, selection, output_dir)
    for resolution in tqdm(("monthly", "quarterly", "annual"), desc="Writing Sankey plots", unit="plot"):
        plot_service_transition_sankey(selected, output_dir, resolution)
    print(f"Wrote horizon Pareto selections and plots to {output_dir}")


if __name__ == "__main__":
    main()
