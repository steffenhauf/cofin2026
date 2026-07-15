#!/usr/bin/env python3
"""Render a concise PNG explaining one Monte Carlo period and persona effects."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def box(
    ax,
    x: float,
    y: float,
    text: str,
    color: str,
    width: float = 0.72,
    height: float = 0.11,
    fontsize: float = 9,
) -> None:
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=color,
        edgecolor="#334155",
        linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color="#0f172a", wrap=True)


def arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate("", end, start, arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 1.25})


TEXT = {
    "en": {
        "period_title": "One Monte Carlo period",
        "persona_title": "What the persona changes in that period",
        "run": "Run + time period",
        "sample": "Sample employees\nand requested work",
        "access": "Choose access path\ncloud service or local hardware",
        "capacity": "Apply budget and capacity\nservice cap · local replicas · concurrent slots",
        "gain": "Draw adoption and AI outcome\nthen calculate gain and AI cost",
        "aggregate": "Aggregate employees\nreturn rate and risk inputs",
        "employee": "Sample one employee",
        "persona": "Persona\ninnovator → laggard",
        "work": "Work type\nsoftware · admin · manual",
        "inputs": "Employee-specific inputs\nadoption probability · usage intensity\ntask fit · wait tolerance",
        "effects": "Access and gain effects\nwho requests work · who accepts waiting\nhow much delivered use improves output",
        "contribution": "Contribution to period totals",
        "markov_title": "Employee-conditioned sampling in one period",
        "persona_example": "One sampled employee\nadoption propensity = p₀ · usage = u\nwait tolerance = w · work-type fit = f",
        "use_draw": "Use AI?\nP = clip(p₀ × [1 + βₐ(capability − 1)])",
        "no_use": "No AI use\nbaseline output · no AI cost",
        "request": "AI requested\napply service or local access",
        "benefit_draw": "Productive outcome?\nP = clip(pᵦ + βᵢ ×\ncapability × work-type fit f)",
        "no_benefit": "No improvement\ncost may apply; gain = 0",
        "benefit": "AI outcome\nsample + scale gain\n(may be negative)",
    },
    "de": {
        "period_title": "Eine Monte-Carlo-Periode",
        "persona_title": "Einfluss der Persona in dieser Periode",
        "run": "Durchlauf + Zeitpunkt",
        "sample": "Mitarbeitende und\nangefragte Arbeit ziehen",
        "access": "Zugangsweg wählen\nCloud-Dienst oder lokale Hardware",
        "capacity": "Budget und Kapazität anwenden\nDienstlimit · lokale Replikate · parallele Slots",
        "gain": "Adoption und KI-Ergebnis ziehen\nGewinn und KI-Kosten berechnen",
        "aggregate": "Mitarbeitende aggregieren\nRendite- und Risiko-Eingaben",
        "employee": "Eine Person ziehen",
        "persona": "Persona\nInnovator → Nachzügler",
        "work": "Arbeitstyp\nSoftware · Verwaltung · manuell",
        "inputs": "Personenspezifische Eingaben\nAdoptionswahrscheinlichkeit · Nutzungsintensität\nAufgabenpassung · Wartebereitschaft",
        "effects": "Zugangs- und Gewinneffekte\nwer Arbeit anfragt · wer wartet\nwie Nutzung den Output verbessert",
        "contribution": "Beitrag zu den Periodensummen",
        "markov_title": "Personen- und arbeitstypbedingte Ziehung in einer Periode",
        "persona_example": "Eine gezogene Person\nAdoptionsneigung = p₀ · Nutzung = u\nWartebereitschaft = w · Arbeitstyp-Passung = f",
        "use_draw": "KI nutzen?\nP = clip(p₀ × [1 + βₐ(Fähigkeit − 1)])",
        "no_use": "Keine KI-Nutzung\nBasisoutput · keine KI-Kosten",
        "request": "KI angefragt\nDienst- oder lokalen\nZugang anwenden",
        "benefit_draw": "Produktives Ergebnis?\nP = clip(pᵦ + βᵢ ×\nFähigkeit × Arbeitstyp-Passung f)",
        "no_benefit": "Keine Verbesserung\nKosten möglich;\nGewinn = 0",
        "benefit": "KI-Ergebnis\nGewinn ziehen + skalieren\n(kann negativ sein)",
    },
}


def render(output: Path, language: str, persona_mode: str) -> None:
    text = TEXT[language]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={"wspace": 0.16})
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    period, persona = axes
    period.set_title(text["period_title"], fontsize=15, fontweight="bold", pad=14)
    box(period, 0.50, 0.90, text["run"], "#dbeafe")
    box(period, 0.50, 0.75, text["sample"], "#e0f2fe")
    box(period, 0.50, 0.57, text["access"], "#dcfce7")
    box(period, 0.50, 0.40, text["capacity"], "#fef3c7")
    box(period, 0.50, 0.23, text["gain"], "#fce7f3")
    box(period, 0.50, 0.075, text["aggregate"], "#ede9fe")
    for y1, y2 in ((0.845, 0.805), (0.695, 0.63), (0.505, 0.445), (0.335, 0.275), (0.165, 0.12)):
        arrow(period, (0.50, y1), (0.50, y2))

    if persona_mode == "inputs":
        persona.set_title(text["persona_title"], fontsize=15, fontweight="bold", pad=14)
        box(persona, 0.50, 0.90, text["employee"], "#dbeafe")
        box(persona, 0.28, 0.73, text["persona"], "#e0f2fe", width=0.40)
        box(persona, 0.72, 0.73, text["work"], "#e0f2fe", width=0.40)
        box(persona, 0.50, 0.52, text["inputs"], "#dcfce7", width=0.76, height=0.15)
        box(persona, 0.50, 0.29, text["effects"], "#fef3c7", width=0.76, height=0.15)
        box(persona, 0.50, 0.08, text["contribution"], "#ede9fe")
        arrow(persona, (0.50, 0.845), (0.28, 0.795))
        arrow(persona, (0.50, 0.845), (0.72, 0.795))
        arrow(persona, (0.28, 0.665), (0.44, 0.605))
        arrow(persona, (0.72, 0.665), (0.56, 0.605))
        arrow(persona, (0.50, 0.435), (0.50, 0.375))
        arrow(persona, (0.50, 0.205), (0.50, 0.135))
    else:
        persona.set_title(text["markov_title"], fontsize=15, fontweight="bold", pad=14)
        box(persona, 0.50, 0.88, text["persona_example"], "#dbeafe", width=0.76, height=0.14)
        box(persona, 0.50, 0.68, text["use_draw"], "#dcfce7", width=0.76, height=0.13)
        box(persona, 0.24, 0.47, text["no_use"], "#fce7f3", width=0.40, height=0.12)
        box(persona, 0.76, 0.47, text["request"], "#fef3c7", width=0.40, height=0.12)
        box(persona, 0.76, 0.28, text["benefit_draw"], "#dcfce7", width=0.40, height=0.12, fontsize=7.5)
        box(persona, 0.46, 0.11, text["no_benefit"], "#fce7f3", width=0.28, height=0.12, fontsize=7.5)
        box(persona, 0.84, 0.11, text["benefit"], "#ede9fe", width=0.28, height=0.12, fontsize=7.5)
        arrow(persona, (0.50, 0.81), (0.50, 0.745))
        arrow(persona, (0.42, 0.615), (0.24, 0.53))
        arrow(persona, (0.58, 0.615), (0.76, 0.53))
        arrow(persona, (0.76, 0.41), (0.76, 0.345))
        arrow(persona, (0.70, 0.22), (0.46, 0.17))
        arrow(persona, (0.82, 0.22), (0.84, 0.17))

    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("simulation_step_and_persona.png"))
    parser.add_argument("--language", choices=sorted(TEXT), default="en")
    parser.add_argument("--persona-mode", choices=["inputs", "markov"], default="inputs")
    parser.add_argument("--all-variants", action="store_true", help="Render English and German input and Markov variants into --output-dir.")
    parser.add_argument("--output-dir", type=Path, default=Path("simulation_diagrams"))
    args = parser.parse_args()
    if args.all_variants:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for language in TEXT:
            for persona_mode in ("inputs", "markov"):
                output = args.output_dir / f"simulation_step_and_persona_{language}_{persona_mode}.png"
                render(output, language, persona_mode)
                print(f"Wrote {output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render(args.output, args.language, args.persona_mode)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
