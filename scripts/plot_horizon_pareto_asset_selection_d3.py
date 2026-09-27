#!/usr/bin/env python3
"""Create D3 Sankey diagrams with sorted links and source-to-target gradients."""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import statistics
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

SELECTIONS = ("low_risk", "optimum", "high_return")
SERVICE_ORDER = ("Global cloud", "EU cloud", "On-prem OSS", "Cloud OSS")
SERVICE_COLORS = {
    "Global cloud": "#2563eb",
    "EU cloud": "#059669",
    "On-prem OSS": "#dc2626",
    "Cloud OSS": "#7c3aed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_folder", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--png",
        action="store_true",
        help="Render each HTML diagram to PNG using Playwright or headless Firefox.",
    )
    parser.add_argument(
        "--show-bracket-composition",
        action="store_true",
        help="Split each service node into employee-bracket share segments.",
    )
    parser.add_argument(
        "--show-organization-composition",
        action="store_true",
        help="Split each service node into organization-profile share segments.",
    )
    return parser.parse_args()


def periods_for(months: list[float], resolution: str) -> list[float]:
    months = sorted(set(months))
    if resolution == "monthly":
        if any(right - left != 1 for left, right in zip(months, months[1:])):
            raise SystemExit(
                "Monthly D3 Sankeys require sweep output at one-month intervals; rerun with resolution_months: 1."
            )
        return months
    step = 3 if resolution == "quarterly" else 12
    periods = [month for month in months if month % step == 0]
    if resolution == "annual" and months and months[-1] not in periods:
        periods.append(months[-1])
    return periods


def load_selections(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"No selections found in {path}.")
    required = {"selection", "company_profile", "users", "month", "service"}
    missing = required - set(rows[0])
    if missing:
        raise SystemExit(
            f"Selection CSV is missing columns: {', '.join(sorted(missing))}."
        )
    for row in rows:
        row["month"] = float(row["month"])
        row["users"] = int(float(row["users"]))
        row["reward"] = float(row["reward"])
        row["combo_key"] = f"{row['company_profile']}|{row['users']}"
    return rows


def build_panel(
    rows: list[dict[str, object]], selection: str, periods: list[float]
) -> dict[str, object]:
    selected = [
        row
        for row in rows
        if row["selection"] == selection and row["month"] in periods
    ]
    combinations = sorted({str(row["combo_key"]) for row in rows})
    by_combo_month = {
        (str(row["combo_key"]), float(row["month"])): str(row["service"])
        for row in selected
    }
    expected = len(combinations)
    if any(
        (combo, month) not in by_combo_month
        for combo in combinations
        for month in periods
    ):
        raise SystemExit(
            f"Incomplete {selection} selections; cannot form constant-N D3 Sankey flows."
        )

    bracket_counts: dict[tuple[float, str], dict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    organization_counts: dict[tuple[float, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for row in selected:
        bracket_counts[(float(row["month"]), str(row["service"]))][
            int(row["users"])
        ] += 1
        organization_counts[(float(row["month"]), str(row["service"]))][
            str(row["company_profile"])
        ] += 1
    nodes = []
    for month in periods:
        for rank, service in enumerate(SERVICE_ORDER):
            counts = bracket_counts[(month, service)]
            total = sum(counts.values())
            if not total:
                continue
            nodes.append(
                {
                    "id": f"{month:g}|{service}",
                    "month": month,
                    "service": service,
                    "rank": rank,
                    "brackets": [
                        {"users": users, "share": count / total}
                        for users, count in sorted(counts.items())
                    ],
                    "organizations": [
                        {"name": name, "share": count / total}
                        for name, count in sorted(
                            organization_counts[(month, service)].items()
                        )
                    ],
                }
            )
    links = []
    for source_month, target_month in zip(periods, periods[1:]):
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for combo in combinations:
            counts[
                (
                    by_combo_month[(combo, source_month)],
                    by_combo_month[(combo, target_month)],
                )
            ] += 1
        if sum(counts.values()) != expected:
            raise SystemExit(
                f"{selection} D3 Sankey flow total is not constant at {expected} combinations."
            )
        for (source_service, target_service), value in sorted(
            counts.items(),
            key=lambda item: (
                SERVICE_ORDER.index(item[0][0]),
                SERVICE_ORDER.index(item[0][1]),
            ),
        ):
            links.append(
                {
                    "source": f"{source_month:g}|{source_service}",
                    "target": f"{target_month:g}|{target_service}",
                    "value": value,
                    "sourceService": source_service,
                    "targetService": target_service,
                }
            )
    returns = []
    base_resolution = min(
        (
            right - left
            for left, right in zip(
                sorted({float(row["month"]) for row in rows}),
                sorted({float(row["month"]) for row in rows})[1:],
            )
            if right > left
        ),
        default=1.0,
    )
    for month in periods:
        for service in SERVICE_ORDER:
            observations = [
                (1.0 + float(row["reward"])) ** (12.0 / base_resolution) - 1.0
                for row in selected
                if float(row["month"]) == month
                and row["service"] == service
                and float(row["reward"]) > -1.0
            ]
            if observations:
                returns.append(
                    {
                        "month": month,
                        "service": service,
                        "medianReturn": statistics.median(observations),
                        "sigmaReturn": (
                            statistics.stdev(observations)
                            if len(observations) > 1
                            else 0.0
                        ),
                    }
                )
    return {
        "nodes": nodes,
        "links": links,
        "returns": returns,
        "population": expected,
    }


def write_html(
    path: Path,
    resolution: str,
    selection: str,
    panel_data: dict[str, object],
    show_bracket_composition: bool,
    show_organization_composition: bool,
) -> None:
    payload = json.dumps(json.dumps(panel_data, separators=(",", ":")))
    document = f"""<!doctype html>
<meta charset="utf-8">
<title>D3 {html.escape(selection)} service transitions ({html.escape(resolution)})</title>
<style>
  body {{ margin: 0; font: 12px sans-serif; color: #263238; }}
  .panel {{ margin: 14px 20px 28px; }}
  .panel h2 {{ margin: 0 0 4px; font-size: 16px; }}
  .node rect {{ stroke: #374151; stroke-width: .5px; }}
  .node text {{ dominant-baseline: middle; fill: #263238; }}
  .link {{ fill: none; stroke-opacity: .58; }}
  .link:hover {{ stroke-opacity: .9; }}
  .return-axis path, .return-axis line {{ stroke: #94a3b8; }}
  .return-axis text {{ fill: #475569; }}
  .return-error {{ stroke-width: 1.2px; stroke-opacity: .75; }}
</style>
<div id="charts"></div>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/d3-sankey@0.12.3/dist/d3-sankey.min.js"></script>
<script>
const DATA = JSON.parse({payload});
const SERVICES = {json.dumps(list(SERVICE_ORDER))};
const COLORS = {json.dumps(SERVICE_COLORS)};
const MARKERS = [d3.symbolCircle, d3.symbolSquare, d3.symbolTriangle, d3.symbolDiamond];
const SHOW_BRACKET_COMPOSITION = {str(show_bracket_composition).lower()};
const SHOW_ORGANIZATION_COMPOSITION = {str(show_organization_composition).lower()};
const COMPOSITION_MODE = SHOW_ORGANIZATION_COMPOSITION ? 'organization' : (SHOW_BRACKET_COMPOSITION ? 'bracket' : 'none');
const BRACKETS = [...new Set(DATA.nodes.flatMap(n => n.brackets.map(b => b.users)))].sort((a, b) => a - b);
const ORGANIZATIONS = [...new Set(DATA.nodes.flatMap(n => n.organizations.map(o => o.name)))].sort();
const COMPOSITION_VALUES = COMPOSITION_MODE === 'organization' ? ORGANIZATIONS : BRACKETS;
const COMPOSITION_OPACITY = d3.scaleLinear().domain([0, Math.max(1, COMPOSITION_VALUES.length - 1)]).range([.35, .9]);
const WIDTH = Math.max(1100, 160 * DATA.nodes.filter(n => n.rank === 0).length);
const HEIGHT = 270;
const NODE_WIDTH = 15;
const margin = {{top: 24, right: 110, bottom: 22, left: 110}};

const selection = {json.dumps(selection)};
const data = DATA;
const panel = d3.select('#charts').append('section').attr('class', 'panel');
panel.append('h2').text(selection.replace('_', ' ').replace(/\\b\\w/g, c => c.toUpperCase()) + ' — ' + {json.dumps(resolution)});
  const svg = panel.append('svg').attr('viewBox', `0 0 ${{WIDTH}} ${{HEIGHT}}`).attr('width', WIDTH).attr('height', HEIGHT);
  const graph = d3.sankey()
    .nodeId(d => d.id)
    .nodeAlign(d3.sankeyJustify)
    .nodeSort((a, b) => d3.ascending(a.rank, b.rank) || d3.ascending(a.id, b.id))
    .linkSort((a, b) => d3.ascending(a.source.rank, b.source.rank) || d3.ascending(a.target.rank, b.target.rank))
    .nodeWidth(NODE_WIDTH).nodePadding(12)
    .extent([[margin.left, margin.top], [WIDTH - margin.right, HEIGHT - margin.bottom]])
    .iterations(48);
  const layout = graph({{nodes: data.nodes.map(d => ({{...d}})), links: data.links.map(d => ({{...d}}))}});
  const defs = svg.append('defs');
  layout.links.forEach((link, i) => {{
    const gradient = defs.append('linearGradient').attr('id', `gradient-${{selection}}-${{i}}`)
      .attr('gradientUnits', 'userSpaceOnUse')
      .attr('x1', link.source.x1).attr('x2', link.target.x0)
      .attr('y1', (link.source.y0 + link.source.y1) / 2)
      .attr('y2', (link.target.y0 + link.target.y1) / 2);
    gradient.append('stop').attr('offset', '0%').attr('stop-color', COLORS[link.sourceService]);
    gradient.append('stop').attr('offset', '100%').attr('stop-color', COLORS[link.targetService]);
  }});
  svg.append('g').selectAll('path').data(layout.links).join('path')
    .attr('class', 'link').attr('d', d3.sankeyLinkHorizontal())
    .attr('stroke', (d, i) => `url(#gradient-${{selection}}-${{i}})`)
    .attr('stroke-width', d => Math.max(1, d.width))
    .append('title').text(d => `${{d.sourceService}} → ${{d.targetService}}: ${{d.value}} combinations`);
  const node = svg.append('g').selectAll('g').data(layout.nodes).join('g').attr('class', 'node');
  node.append('rect').attr('x', d => d.x0).attr('y', d => d.y0).attr('height', d => d.y1 - d.y0).attr('width', d => d.x1 - d.x0).attr('fill', d => COLORS[d.service]).attr('fill-opacity', d => COMPOSITION_MODE !== 'none' ? .18 : 1);
  if (COMPOSITION_MODE !== 'none') {{
    node.each(function(d) {{
      const composition = COMPOSITION_MODE === 'organization' ? d.organizations : d.brackets;
      let y = d.y0;
      d3.select(this).selectAll('rect.composition').data(composition).join('rect')
        .attr('class', 'composition').attr('x', d.x0).attr('y', b => {{ const height = (d.y1 - d.y0) * b.share; const top = y; y += height; return top; }})
        .attr('height', b => (d.y1 - d.y0) * b.share).attr('width', d.x1 - d.x0)
        .attr('fill', COLORS[d.service]).attr('fill-opacity', b => COMPOSITION_OPACITY(COMPOSITION_MODE === 'organization' ? ORGANIZATIONS.indexOf(b.name) : BRACKETS.indexOf(b.users)))
        .append('title').text(b => `${{COMPOSITION_MODE === 'organization' ? b.name : b.users + ' employees'}}: ${{d3.format('.1%')(b.share)}} of ${{d.service}} contributors`);
    }});
  }}
  node.append('text').attr('x', d => d.x0 < WIDTH / 2 ? d.x1 + 5 : d.x0 - 5).attr('y', d => (d.y0 + d.y1) / 2).attr('text-anchor', d => d.x0 < WIDTH / 2 ? 'start' : 'end').text(d => d.service);
  if (COMPOSITION_MODE !== 'none') {{
    const compositionLegend = svg.append('g').attr('transform', `translate(8,${{margin.top}})`);
    compositionLegend.append('rect').attr('x', -5).attr('y', -16).attr('width', 96).attr('height', COMPOSITION_VALUES.length * 18 + 24).attr('fill', 'white').attr('fill-opacity', .86);
    compositionLegend.append('text').attr('y', 0).attr('fill', '#334155').text(COMPOSITION_MODE === 'organization' ? 'Organization opacity' : 'Bracket opacity');
    COMPOSITION_VALUES.forEach((value, i) => compositionLegend.append('text').attr('y', 18 + i * 18).attr('fill', '#334155').attr('fill-opacity', COMPOSITION_OPACITY(i)).text(COMPOSITION_MODE === 'organization' ? value : `${{value}} employees`));
  }}

const returnHeight = 220;
const returnSvg = panel.append('svg').attr('viewBox', `0 0 ${{WIDTH}} ${{returnHeight}}`).attr('width', WIDTH).attr('height', returnHeight);
const returnMargin = {{top: 20, right: 110, bottom: 34, left: 110}};
const returnX = d3.scalePoint().domain([...new Set(data.returns.map(d => d.month))]).range([returnMargin.left + NODE_WIDTH / 2, WIDTH - returnMargin.right - NODE_WIDTH / 2]);
const returnValues = data.returns.flatMap(d => [d.medianReturn - d.sigmaReturn, d.medianReturn + d.sigmaReturn]);
const returnY = d3.scaleLinear().domain(d3.extent(returnValues)).nice().range([returnHeight - returnMargin.bottom, returnMargin.top]);
returnSvg.append('g').attr('class', 'return-axis').attr('transform', `translate(0,${{returnHeight - returnMargin.bottom}})`).call(d3.axisBottom(returnX).tickFormat(d => `Month ${{d3.format('.0f')(d)}}`));
returnSvg.append('g').attr('class', 'return-axis').attr('transform', `translate(${{returnMargin.left}},0)`).call(d3.axisLeft(returnY).ticks(5, '.1%'));
returnSvg.append('text').attr('transform', `translate(16,${{returnHeight / 2}}) rotate(-90)`).attr('text-anchor', 'middle').text('Median annualized return');
SERVICES.forEach((service, serviceIndex) => {{
  const values = data.returns.filter(d => d.service === service);
  const pointX = d => returnX(d.month) + (serviceIndex - (SERVICES.length - 1) / 2) * 8;
  const error = returnSvg.append('g').attr('class', 'return-error').attr('stroke', COLORS[service]);
  error.selectAll('line').data(values).join('line')
    .attr('x1', pointX).attr('x2', pointX)
    .attr('y1', d => returnY(d.medianReturn - d.sigmaReturn)).attr('y2', d => returnY(d.medianReturn + d.sigmaReturn));
  error.selectAll('path').data(values).join('path')
    .attr('d', d => `M${{pointX(d) - 4}},${{returnY(d.medianReturn - d.sigmaReturn)}}H${{pointX(d) + 4}}M${{pointX(d) - 4}},${{returnY(d.medianReturn + d.sigmaReturn)}}H${{pointX(d) + 4}}`);
  returnSvg.append('g').selectAll('path').data(values).join('path').attr('transform', d => `translate(${{pointX(d)}},${{returnY(d.medianReturn)}})`).attr('d', d3.symbol().type(MARKERS[serviceIndex]).size(64)).attr('fill', COLORS[service]).append('title').text(d => `${{d.service}} at month ${{d.month}}: ${{d3.format('.2%')(d.medianReturn)}}`);
}});
const legend = returnSvg.append('g').attr('transform', `translate(${{returnMargin.left + 12}},${{returnMargin.top + 8}})`);
legend.append('rect').attr('x', -8).attr('y', -14).attr('width', 125).attr('height', SERVICES.length * 18 + 10).attr('fill', 'white').attr('fill-opacity', .82);
SERVICES.forEach((service, i) => {{
  legend.append('path').attr('transform', `translate(2,${{i * 18 - 4}})`).attr('d', d3.symbol().type(MARKERS[i]).size(64)).attr('fill', COLORS[service]);
  legend.append('text').attr('x', 12).attr('y', i * 18).attr('fill', COLORS[service]).text(service);
}});
</script>
"""
    path.write_text(document)


def render_png(html_path: Path, png_path: Path) -> bool:
    """Render a generated D3 HTML file when a supported headless browser exists."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(
                viewport={"width": 2400, "height": 1100}, device_scale_factor=1
            )
            page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
            page.screenshot(path=str(png_path), full_page=True)
            browser.close()
        return True
    except Exception:
        pass

    firefox = shutil.which("firefox")
    if firefox is None:
        return False
    with tempfile.TemporaryDirectory(
        prefix="d3-sankey-firefox-"
    ) as profile_dir:
        try:
            result = subprocess.run(
                [
                    firefox,
                    "--headless",
                    "--no-remote",
                    "-profile",
                    profile_dir,
                    "--screenshot",
                    str(png_path),
                    "--window-size",
                    "2400,1100",
                    html_path.resolve().as_uri(),
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
    return result.returncode == 0 and png_path.exists()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.sweep_folder / "horizon_pareto_plots"
    selection_path = output_dir / "horizon_pareto_asset_selections.csv"
    rows = load_selections(selection_path)
    months = [float(row["month"]) for row in rows]
    for resolution in ("monthly", "quarterly", "annual"):
        periods = periods_for(months, resolution)
        if len(periods) < 2:
            continue
        for selection in SELECTIONS:
            panel_data = build_panel(rows, selection, periods)
            html_path = (
                output_dir
                / f"horizon_service_transition_sankey_d3_{resolution}_{selection}.html"
            )
            write_html(
                html_path,
                resolution,
                selection,
                panel_data,
                args.show_bracket_composition,
                args.show_organization_composition,
            )
            if args.png and not render_png(
                html_path, html_path.with_suffix(".png")
            ):
                print(
                    f"Warning: could not render {html_path.name} to PNG; install Playwright or Firefox."
                )
    print(f"Wrote D3 Sankey diagrams to {output_dir}")


if __name__ == "__main__":
    main()
