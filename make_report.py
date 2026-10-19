#!/usr/bin/env python3
"""Build the Phase 1 PDF."""

import argparse
import json
import math
from datetime import datetime

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402

EPILOG = (
    "example:\n  python make_report.py --results results/results.json --team team.json --out "
    "FunnelVision_Phase1.pdf"
)

TEAM = {
    "name": "Funnel Vision",
    "members": [
        [
            "Soufiane Amribt",
            "UNB Fredericton, Computer Science (cybersecurity)",
            "Data pipeline, statistics, recommender experiment",
        ],
        ["[Teammate 2]", "[Program, school]", "[Role]"],
        ["[Teammate 3]", "[Program, school]", "[Role]"],
    ],
    "code_url": "",
    "collaboration": "",
}

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
PAGE_WIDTH, PAGE_HEIGHT = 11.0, 8.5
MARGIN = 0.55
INK, INK_SOFT, MUTED = "#111827", "#3e4a5a", "#667182"
GRID, AXIS, PANEL, RULE = "#e4e8ee", "#c3cad4", "#f3f5f9", "#d9dfe7"
WOMEN, MEN = "#a34aa8", "#a3abb6"
MEN_LIGHT = "#e3e7ed"
ACCENT_TEXT = "#7a2f80"
STAGE_COLORS = {"published": "#9fb8dc", "shown": "#1d3f7a", "clicked": "#1d3f7a"}
GAIN_COLOR, LOSS_COLOR = "#1d3f7a", "#c0392b"
NULL_COLOR = "#8a93a0"
FONT_FAMILY = "DejaVu Sans"

# Display thresholds
EPSILON = 1e-9
LOG_FLOOR = 1e-3
PARITY = 0.5
MIN_BAND_LABEL = 0.07
MIN_TOPIC_APPEARANCES = 80
MIN_SECTION_STORIES = 30
MIN_COMMUNITY_READERS = 1000
MIN_SCORECARD_SHARE = 0.002
SMALL_GROUP_SHARE = 0.01
CLEAR_LOSS_RATIO = 0.9

SUBSECTION_NAMES = {
    "entertainment/entertainment-celebrity": "Celebrity news",
    "entertainment/celebrity": "Celebrity",
    "lifestyle/lifestyleroyals": "Royals",
    "tv/tv-celebrity": "TV celebrity",
    "video/news": "News video",
    "music/music-celebrity": "Music celebrity",
    "movies/movienews": "Movie news",
    "music/cma-awards": "CMA Awards",
    "lifestyle/lifestylecelebstyle": "Celebrity style",
    "news/newsus": "US news",
    "tv/tvnews": "TV news",
    "lifestyle/lifestylebuzz": "Lifestyle buzz",
    "sports/football_nfl": "NFL",
}


def setup_fonts():
    """Pick Inter and set styles."""
    global FONT_FAMILY
    names = {font.name for font in font_manager.fontManager.ttflist}
    FONT_FAMILY = "Inter" if "Inter" in names else "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "pdf.fonttype": 3,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.labelcolor": INK_SOFT,
            "axes.labelsize": 7.5,
            "text.color": INK,
        }
    )


# ---------------------------------------------------------------------------
# Layout helpers, inches, top-left origin
# ---------------------------------------------------------------------------


def draw_text(fig, x, y, text, size=9, weight="normal", color=INK, ha="left", va="top", **kwargs):
    """Draw text in page inches."""
    return fig.text(
        x / PAGE_WIDTH,
        1 - y / PAGE_HEIGHT,
        text,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        **kwargs,
    )


def wrap_lines(text, width_in, size, weight="normal"):
    """Wrap text to a width."""
    font = FontProperties(family=FONT_FAMILY, size=size, weight=weight)
    limit = width_in * 72
    lines = []
    for block in text.split("\n"):
        current = ""
        for word in block.split():
            trial = (current + " " + word).strip()
            if not current or TextPath((0, 0), trial, prop=font).get_extents().width <= limit:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_paragraph(fig, x, y, text, width, size=8.5, weight="normal", color=INK, line_height=1.38):
    """Draw wrapped text, return bottom."""
    lines = wrap_lines(text, width, size, weight)
    step = size * line_height / 72
    for i, line in enumerate(lines):
        draw_text(fig, x, y + i * step, line, size=size, weight=weight, color=color)
    return y + len(lines) * step


def add_axes(fig, x, y, width, height):
    """Add axes in page inches."""
    return fig.add_axes(
        [x / PAGE_WIDTH, 1 - (y + height) / PAGE_HEIGHT, width / PAGE_WIDTH, height / PAGE_HEIGHT]
    )


def draw_rect(fig, x, y, width, height, color, zorder=0, radius=0.0):
    """Draw a filled rectangle."""
    if radius:
        patch = FancyBboxPatch(
            (x / PAGE_WIDTH, 1 - (y + height) / PAGE_HEIGHT),
            width / PAGE_WIDTH,
            height / PAGE_HEIGHT,
            boxstyle=f"round,pad=0,rounding_size={radius / PAGE_WIDTH}",
            transform=fig.transFigure,
            facecolor=color,
            edgecolor="none",
            zorder=zorder,
        )
    else:
        patch = Rectangle(
            (x / PAGE_WIDTH, 1 - (y + height) / PAGE_HEIGHT),
            width / PAGE_WIDTH,
            height / PAGE_HEIGHT,
            transform=fig.transFigure,
            facecolor=color,
            edgecolor="none",
            zorder=zorder,
        )
    fig.add_artist(patch)


def draw_hline(fig, x0, x1, y, color=RULE, lw=0.8):
    """Draw a horizontal rule."""
    fig.add_artist(
        plt.Line2D(
            [x0 / PAGE_WIDTH, x1 / PAGE_WIDTH],
            [1 - y / PAGE_HEIGHT, 1 - y / PAGE_HEIGHT],
            transform=fig.transFigure,
            color=color,
            lw=lw,
        )
    )


def draw_vline(fig, x, y0, y1, color=RULE, lw=0.8):
    """Draw a vertical rule."""
    fig.add_artist(
        plt.Line2D(
            [x / PAGE_WIDTH, x / PAGE_WIDTH],
            [1 - y0 / PAGE_HEIGHT, 1 - y1 / PAGE_HEIGHT],
            transform=fig.transFigure,
            color=color,
            lw=lw,
        )
    )


def draw_dot(fig, x, y, radius, color):
    """Draw a filled circle."""
    fig.add_artist(
        Circle(
            (x / PAGE_WIDTH, 1 - y / PAGE_HEIGHT),
            radius / PAGE_WIDTH,
            transform=fig.transFigure,
            facecolor=color,
            edgecolor="none",
        )
    )


def draw_arrow(fig, x0, x1, y, color=INK_SOFT, lw=1.0, scale=9):
    """Draw a horizontal arrow."""
    fig.add_artist(
        FancyArrowPatch(
            (x0 / PAGE_WIDTH, 1 - y / PAGE_HEIGHT),
            (x1 / PAGE_WIDTH, 1 - y / PAGE_HEIGHT),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=scale,
            color=color,
            lw=lw,
        )
    )


def style_axes(ax, grid="x"):
    """Apply the minimal axes style."""
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0, pad=3)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    ax.set_facecolor("none")


PERCENT_FORMAT = matplotlib.ticker.PercentFormatter(1.0, decimals=0)


def percent(value, digits=0):
    """Format a share as percent."""
    return (
        "n/a"
        if value is None or (isinstance(value, float) and math.isnan(value))
        else f"{value * 100:.{digits}f}%"
    )


def thousands(value):
    """Format with thousands separators."""
    return f"{value:,.0f}"


def dataset_name(results):
    """Dataset label from results."""
    return results.get("meta", {}).get("dataset") or "MIND"


def date_span(data):
    """Readable date range of logs."""
    try:
        first = datetime.fromisoformat(data["first_log"][:19])
        last = datetime.fromisoformat(data["last_log"][:19])
        if first.month == last.month:
            return f"{first.strftime('%B')} {first.day} to {last.day}, {last.year}"
        return (
            f"{first.strftime('%B')} {first.day} to {last.strftime('%B')} {last.day}, {last.year}"
        )
    except (KeyError, ValueError, TypeError):
        return "November 2019"


def chart_title(fig, x, y, title, subtitle=None, width=3.5):
    """Draw title and subtitle."""
    draw_text(fig, x, y, title, size=9.5, weight="bold")
    if subtitle:
        return (
            draw_paragraph(
                fig, x, y + 0.2, subtitle, width, size=7.5, color=INK_SOFT, line_height=1.3
            )
            + 0.05
        )
    return y + 0.25


def page_frame(fig, page_number, results, preview):
    """Draw header, footer and background."""
    draw_rect(fig, 0, 0, PAGE_WIDTH, PAGE_HEIGHT, "#ffffff", zorder=-10)
    draw_text(
        fig,
        MARGIN,
        0.32,
        "FUNNEL VISION   ·   BEHIND THE FEED DATA CHALLENGE   ·   PHASE 1",
        size=6.8,
        weight="bold",
        color=MUTED,
    )
    draw_text(
        fig,
        PAGE_WIDTH - MARGIN,
        0.32,
        f"{page_number} / 5",
        size=6.8,
        weight="bold",
        color=MUTED,
        ha="right",
    )
    draw_hline(fig, MARGIN, PAGE_WIDTH - MARGIN, 8.06)
    code = TEAM.get("code_url", "")
    code = code.replace("https://", "").replace("http://", "").rstrip("/")
    tail = f" Code and results: {code}" if code and "[" not in code else ""
    where = " Sources, method and limitations: page 5." if page_number < 5 else ""
    base = (
        f"Data: Microsoft News Dataset ({dataset_name(results)}, MSN News logs, "
        f"{date_span(results['data'])}) and Wikidata. Charts and computed figures by the team, "
        f"outside statistics cited where used.{where}"
    )
    lines = (
        [base + tail]
        if len(wrap_lines(base + tail, PAGE_WIDTH - 2 * MARGIN, 6.3)) == 1
        else [base, tail.strip()]
    )
    lines = [line for line in lines if line]
    top = 8.13 if len(lines) == 1 else 8.10
    for i, line in enumerate(lines):
        draw_text(fig, MARGIN, top + i * 0.105, line, size=6.3, color=MUTED)
    if preview:
        fig.text(
            0.5,
            0.5,
            "LAYOUT PREVIEW  ·  FAKE TEST NUMBERS  ·  DO NOT SUBMIT",
            fontsize=30,
            color="#d03b3b",
            alpha=0.18,
            rotation=28,
            ha="center",
            va="center",
            weight="bold",
        )


def section_head(fig, kicker, title, takeaway):
    """Draw kicker, title and takeaway."""
    draw_text(fig, MARGIN, 0.62, kicker, size=7.5, weight="bold", color=ACCENT_TEXT)
    draw_text(fig, MARGIN, 0.82, title, size=20, weight="bold")
    return draw_paragraph(
        fig, MARGIN, 1.25, takeaway, 9.9, size=10, color=INK_SOFT, line_height=1.32
    )


def share_without_sports(results, stage):
    """Women's share excluding Sports."""
    women = gendered = 0.0
    for section, values in results["section"].items():
        if section == "Sports" or values[stage]["women_share"] is None:
            continue
        women += values[stage]["women_share"] * values[stage]["gendered_appearances"]
        gendered += values[stage]["gendered_appearances"]
    return women / gendered if gendered else None


CATEGORY_NAMES = {
    "foodanddrink": "Food and drink",
    "tv": "TV",
    "finance": "Finance",
    "news": "News",
    "sports": "Sports",
    "entertainment": "Entertainment",
    "lifestyle": "Lifestyle",
    "music": "Music",
    "movies": "Movies",
    "health": "Health",
    "travel": "Travel",
    "video": "Video",
    "weather": "Weather",
    "autos": "Autos",
}


def hyphenate(name):
    """Hyphenate community names."""
    return name.replace(" first", "-first")


def scorecard_row(results, name):
    """Find a scorecard row."""
    for row in results["scorecard"]:
        if row["group"] == name:
            return row
    return None


def parity_point(sweep, target=PARITY):
    """First sweep point reaching target."""
    reached = [
        point
        for point in sweep
        if point["top5_women_share"] is not None and point["top5_women_share"] >= target
    ]
    return reached[0] if reached else None


# ---------------------------------------------------------------------------
# Page 1: hook and headline
# ---------------------------------------------------------------------------


def page1(pdf, results, preview):
    """Page 1: hook and headline."""
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
    page_frame(fig, 1, results, preview)
    funnel = results["funnel_gender"]
    women_share, ci = funnel["women_share"], funnel["ci95"]
    ratio_platform, ratio_reader = (
        funnel["ratio_shown_vs_published"],
        funnel["ratio_clicked_vs_shown"],
    )
    data = results["data"]
    decomposition = results["decomposition"]["by_subcategory"]["shown_vs_published"]
    roles = results["roles"]

    draw_text(
        fig,
        MARGIN,
        0.62,
        "WHO GETS SEEN IN YOUR NEWS FEED?",
        size=7.5,
        weight="bold",
        color=ACCENT_TEXT,
    )
    draw_text(fig, MARGIN, 0.80, "Funnel Vision", size=34, weight="bold")
    draw_paragraph(
        fig,
        MARGIN,
        1.42,
        (
            "An audit engine that follows every public figure named in the news from the story "
            f"pool to the screen to the click. We benchmarked it on {dataset_name(results)}, the "
            f"field's standard dataset: {thousands(results['concentration']['people_shown'])} "
            f"public figures in {thousands(data['articles_in_circulation'])} stories, "
            f"{thousands(data['displays'])} recorded displays to {thousands(data['users'])} "
            "readers."
        ),
        6.3,
        size=10.5,
        color=INK_SOFT,
        line_height=1.35,
    )

    draw_text(
        fig,
        MARGIN,
        2.55,
        "Out of every 100 people named in the stories, how many are women?",
        size=10.5,
        weight="bold",
    )
    stage_x = [MARGIN, MARGIN + 2.25, MARGIN + 4.5]
    top, cell = 3.25, 0.15
    for stage, x in zip(("published", "shown", "clicked"), stage_x):
        n_women = int(round(women_share[stage] * 100))
        draw_text(fig, x, 2.82, f"{n_women}", size=26, weight="bold")
        draw_text(fig, x + 0.6, 2.93, "women in 100", size=8, color=INK_SOFT)
        for i in range(100):
            row, col = divmod(i, 10)
            draw_dot(
                fig,
                x + col * cell + cell / 2,
                top + row * cell + cell / 2,
                0.052,
                WOMEN if i < n_women else MEN_LIGHT,
            )
        draw_text(
            fig,
            x,
            top + 1.58,
            {"published": "In the pool", "shown": "Shown", "clicked": "Clicked"}[stage],
            size=10,
            weight="bold",
        )
    definitions = {
        "published": "Every story in circulation, counted once. What publishers supplied.",
        "shown": "Every time the platform displayed a story to a reader. What the platform chose.",
        "clicked": "Every time a reader clicked a story. What people chose.",
    }
    for stage, x in zip(("published", "shown", "clicked"), stage_x):
        draw_paragraph(
            fig, x, top + 1.8, definitions[stage], 1.55, size=7.3, color=INK_SOFT, line_height=1.3
        )
    for x0, ratio, label in (
        (stage_x[0], ratio_platform, "platform step"),
        (stage_x[1], ratio_reader, "reader step"),
    ):
        arrow_y = top + 0.72
        draw_arrow(fig, x0 + 1.58, x0 + 2.12, arrow_y)
        draw_text(
            fig,
            x0 + 1.85,
            arrow_y - 0.12,
            f"×{ratio:.2f}",
            size=8.5,
            weight="bold",
            ha="center",
            va="bottom",
        )
        draw_text(fig, x0 + 1.85, arrow_y + 0.1, label, size=6.5, color=MUTED, ha="center")

    headline_y = 5.66
    bottom = draw_paragraph(
        fig,
        MARGIN + 0.2,
        headline_y,
        "Women became more visible, and more typecast.",
        6.3,
        size=12.5,
        weight="bold",
        line_height=1.22,
    )
    half_width = (ci["shown"][1] - ci["shown"][0]) / 2 * 100
    ratio_ci = ci["ratio_shown_vs_published"]
    lead = (
        "We expected the platform to shrink women's share. It did the opposite: women "
        if ratio_ci[0] > 1
        else "Women "
    )
    body = (
        f"{lead}were {percent(women_share['published'], 1)} of the people named in the story pool "
        f"and {percent(women_share['shown'], 1)} of the people readers were shown (95% interval "
        f"±{half_width:.2f} points). But "
        f"{decomposition['mix'] / decomposition['total'] * 100:.1f}% of that rise came from the "
        "topic mix, led by celebrity and royal stories. "
        f"{percent(roles['shown']['Entertainment & arts']['share_of_women'])} of women's "
        "appearances on screen were as entertainers or artists, up from "
        f"{percent(roles['published']['Entertainment & arts']['share_of_women'])} in the pool. "
        "Our typecast index (how far apart women's and men's roles are) rose from "
        f"{results['role_dissimilarity']['published']:.2f} to "
        f"{results['role_dissimilarity']['shown']:.2f}."
    )
    bottom = draw_paragraph(fig, MARGIN + 0.2, bottom + 0.05, body, 6.3, size=8.8, color=INK_SOFT)
    draw_rect(fig, MARGIN, headline_y, 0.06, bottom - headline_y, WOMEN)

    panel_x, panel_width = 7.35, PAGE_WIDTH - MARGIN - 7.35
    draw_rect(fig, panel_x - 0.2, 0.6, panel_width + 0.2, 7.3, PANEL, radius=0.08)
    draw_text(
        fig, panel_x, 0.82, "WHY THIS MATTERS IN CANADA", size=7.5, weight="bold", color=MUTED
    )
    tiles = [
        (
            "53%",
            "of Canadians used social media for news in the past week, overtaking TV for the "
            "first time, even with news blocked on Facebook and Instagram.",
            "Digital News Report 2026, Canada (Centre d'études sur les médias)",
        ),
        (
            "< 1/3",
            "of the voices quoted in major Canadian news outlets belong to women and "
            "gender-diverse people.",
            "Gender Gap Tracker, Informed Opinions and Simon Fraser University",
        ),
        (
            "99%",
            "of the content on The Globe and Mail's digital pages is placed by its own AI system, "
            "Sophi, as reported by the vendor.",
            "Mather Economics case study, The Globe and Mail and Sophi",
        ),
    ]
    y = 1.12
    for stat, caption, source in tiles:
        draw_text(fig, panel_x, y, stat, size=24, weight="bold")
        y = draw_paragraph(
            fig, panel_x, y + 0.48, caption, panel_width - 0.1, size=8.2, color=INK_SOFT
        )
        y = (
            draw_paragraph(
                fig,
                panel_x,
                y + 0.04,
                source,
                panel_width - 0.1,
                size=6.4,
                color=MUTED,
                line_height=1.25,
            )
            + 0.2
        )
    draw_hline(fig, panel_x, panel_x + panel_width - 0.05, y - 0.05)
    draw_text(fig, panel_x, y + 0.08, "The step rarely measured", size=9.5, weight="bold")
    y = draw_paragraph(
        fig,
        panel_x,
        y + 0.32,
        (
            "Newsrooms get counted. Feeds rarely do. The EU's Digital Services Act already makes "
            "platforms disclose their recommenders' main parameters (Art. 27) and the largest "
            "offer a feed not based on profiling (Art. 38). We found no public dataset of what "
            "Canadians are shown, so we benchmark the engine on the field's standard logs first, "
            "which are US-centred."
        ),
        panel_width - 0.1,
        size=8.2,
        color=INK_SOFT,
    )
    draw_text(fig, panel_x, y + 0.16, "Where this goes", size=9.5, weight="bold")
    draw_paragraph(
        fig,
        panel_x,
        y + 0.4,
        (
            "A funnel scorecard for every algorithmic news feed in Canada. Each newsroom runs the "
            "engine on its own logs and publishes only the scores, so readers can see what the "
            "algorithm adds. Phase 2: the scorecard prototype, a within-section fix, and a pilot "
            "plan for a Canadian newsroom that already curates with an algorithm."
        ),
        panel_width - 0.1,
        size=8.2,
        color=INK_SOFT,
    )

    audit_y = 6.95
    draw_text(
        fig, MARGIN, audit_y, "WHAT IS NEW IN THIS AUDIT", size=7.5, weight="bold", color=MUTED
    )
    items = [
        (
            "People, not topics",
            "Feed audits usually measure topic diversity. We follow the "
            f"{thousands(results['concentration']['people_shown'])} public figures inside the "
            "stories.",
        ),
        (
            "The funnel ratio",
            "Share at one stage ÷ share at the stage before. It splits each gap into a platform "
            "step and a reader step.",
        ),
        (
            "Mix or treatment",
            "A shift-share split separates a change in topic mix from different treatment within "
            "a topic.",
        ),
        (
            "Nothing inferred",
            "No gender guessed from names. Only Wikidata records about public figures. Reader "
            "groups from clicks.",
        ),
    ]
    for i, (item_title, item_text) in enumerate(items):
        x = MARGIN + i * 1.62
        draw_text(fig, x, audit_y + 0.22, f"{i + 1}", size=15, weight="bold", color=ACCENT_TEXT)
        draw_text(fig, x + 0.25, audit_y + 0.25, item_title, size=8, weight="bold")
        draw_paragraph(
            fig,
            x + 0.25,
            audit_y + 0.45,
            item_text,
            1.32,
            size=6.8,
            color=INK_SOFT,
            line_height=1.25,
        )
    pdf.savefig(fig)
    plt.close(fig)


ROLE_GROUPS = [
    ("Sports", ["Sports"], "#3d6bb0"),
    ("Experts", ["Science, health & academia", "Media & writing"], "#4a5a70"),
    ("Politics", ["Politics & government"], "#9fb8dc"),
    ("Others", None, "#d5dbe3"),
    ("Entertainment", ["Entertainment & arts"], "#1f9e8f"),
]


# Ratio ranges named in words
MULTIPLIER_WORDS = (
    (2.75, 3.5, "triples"),
    (1.75, 2.25, "doubles"),
    (0.4, 0.6, "halves"),
    (0.28, 0.4, "cuts to a third"),
)


def role_shares(results):
    """Role group shares per stage."""
    role_rows = {r["group"]: r for r in results["scorecard"] if r["family"].startswith("Role")}
    shares = []
    for stage in ("published", "shown", "clicked"):
        values = []
        for _, members, _ in ROLE_GROUPS:
            if members is None:
                values.append(None)
            else:
                values.append(sum(role_rows[member][stage] for member in members))
        rest = 1 - sum(value for value in values if value is not None)
        shares.append([rest if value is None else value for value in values])
    return shares


def role_bands(fig, results, x, y, width, height):
    """Draw the role flow chart."""
    shares = role_shares(results)
    ax = add_axes(fig, x, y, width, height)
    ax.axis("off")
    bar_width, gap = 0.36, 0.007
    stage_x = [0, 1, 2]
    segments = []
    for values in shares:
        cumulative, column = 0.0, []
        for stage_values in values:
            column.append((cumulative, cumulative + stage_values))
            cumulative += stage_values
        segments.append(column)
    progress = np.linspace(0, 1, 50)
    smooth = 3 * progress**2 - 2 * progress**3
    for k in range(2):
        for group, (_, _, color) in enumerate(ROLE_GROUPS):
            (a0, a1), (b0, b1) = segments[k][group], segments[k + 1][group]
            xa, xb = stage_x[k] + bar_width / 2, stage_x[k + 1] - bar_width / 2
            ax.fill_between(
                xa + (xb - xa) * progress,
                a0 + (b0 - a0) * smooth + gap / 2,
                a1 + (b1 - a1) * smooth - gap / 2,
                color=color,
                alpha=0.25,
                lw=0,
            )
    for k in range(3):
        for group, (_, _, color) in enumerate(ROLE_GROUPS):
            y0, y1 = segments[k][group]
            ax.add_patch(
                Rectangle(
                    (stage_x[k] - bar_width / 2, y0 + gap / 2),
                    bar_width,
                    max(y1 - y0 - gap, 0),
                    facecolor=color,
                    edgecolor="none",
                )
            )
            if y1 - y0 >= MIN_BAND_LABEL:
                rgb = matplotlib.colors.to_rgb(color)
                luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                ax.text(
                    stage_x[k],
                    (y0 + y1) / 2,
                    f"{(y1 - y0) * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7.4,
                    weight="bold",
                    color="white" if luminance < 0.5 else INK,
                )
    for group, (name, _, _) in enumerate(ROLE_GROUPS):
        y0, y1 = segments[2][group]
        stage_values = [shares[k][group] for k in range(3)]
        label = (
            name
            if stage_values[2] >= MIN_BAND_LABEL
            else f"{name}  {stage_values[0] * 100:.1f} → {stage_values[1] * 100:.1f} → "
            f"{stage_values[2] * 100:.1f}%"
        )
        ax.text(
            stage_x[2] + bar_width / 2 + 0.07,
            (y0 + y1) / 2,
            label,
            ha="left",
            va="center",
            fontsize=7,
            color=INK,
        )
    for k, label in enumerate(["In the pool", "Shown", "Clicked"]):
        ax.text(stage_x[k], -0.035, label, ha="center", va="top", fontsize=7.6, color=INK_SOFT)
    ax.set_xlim(-bar_width / 2 - 0.05, 3.55)
    ax.set_ylim(-0.1, 1.0)
    return ax


# ---------------------------------------------------------------------------
# Page 2: where gap opens
# ---------------------------------------------------------------------------


def page2(pdf, results, preview):
    """Page 2: where gap opens."""
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
    page_frame(fig, 2, results, preview)
    women_share = results["funnel_gender"]["women_share"]
    decomposition = results["decomposition"]["by_subcategory"]["shown_vs_published"]
    themes = []
    for contributor in decomposition["top_contributors"]:
        if contributor["within"] + contributor["mix"] <= 0:
            continue
        group = contributor["group"].lower()
        theme = (
            "celebrity"
            if "celeb" in group
            else (
                "royal"
                if "royal" in group
                else SUBSECTION_NAMES.get(contributor["group"], group.split("/")[-1]).lower()
            )
        )
        if theme not in themes:
            themes.append(theme)
    themes = (themes + ["other"])[:2]
    section_head(
        fig,
        "WHERE THE GAP OPENS   ·   CHALLENGE QUESTIONS 2 AND 5",
        "More women on screen, mostly through the topic mix",
        (
            f"Women's share of people rose from {percent(women_share['published'], 1)} in the "
            f"pool to {percent(women_share['shown'], 1)} on screen, then held at "
            f"{percent(women_share['clicked'], 1)} after clicks. "
            f"{decomposition['mix'] / decomposition['total'] * 100:.1f}% of the platform's change "
            f"came from which subsections it showed more, led by {themes[0]} and {themes[1]} "
            "stories. Inside the same subsection it moved women's share by only "
            f"{decomposition['within'] * 100:+.1f} points."
        ),
    )

    # A: roles across the funnel
    x0, y0 = MARGIN, 1.95
    bottom = chart_title(
        fig,
        x0,
        y0,
        "The platform reshapes who fills the screen",
        "Share of all person appearances by role, one role per person from Wikidata occupations. "
        "Experts: science, health, academia, media and writing.",
        3.2,
    )
    role_bands(fig, results, x0, bottom + 0.1, 3.25, 2.08)

    # B: over-shown topics
    x1, topic_width = 4.05, 3.3
    categories = results["category"]
    pool_total = sum(value["published"]["gendered_appearances"] for value in categories.values())
    shown_total = sum(value["shown"]["gendered_appearances"] for value in categories.values())
    topic_rows = []
    for k, value in categories.items():
        pool_appearances, shown_appearances = (
            value["published"]["gendered_appearances"],
            value["shown"]["gendered_appearances"],
        )
        if pool_appearances >= MIN_TOPIC_APPEARANCES:
            topic_rows.append(
                (
                    CATEGORY_NAMES.get(k, k.title()),
                    (shown_appearances / shown_total) / (pool_appearances / pool_total),
                    value["published"]["women_share"],
                )
            )
    topic_rows.sort(key=lambda row: -row[1])
    bottom = chart_title(
        fig,
        x1,
        y0,
        "Which topics the platform over-shows",
        "Funnel ratio of each topic's share of person appearances (shown ÷ pool). Purple: women's "
        f"share of that topic's pool. Topic mix explains {decomposition['mix'] * 100:+.1f} of the "
        f"{decomposition['total'] * 100:+.1f} points women gained.",
        topic_width,
    )
    ax = add_axes(fig, x1 + 0.82, bottom + 0.06, topic_width - 1.32, 1.98)
    style_axes(ax, "x")
    topic_ratios = [row[1] for row in topic_rows]
    ax.barh(
        range(len(topic_rows)),
        topic_ratios,
        color=[
            {1: GAIN_COLOR, -1: LOSS_COLOR, 0: MUTED}[direction(value)] for value in topic_ratios
        ],
        height=0.62,
    )
    ax.set_xscale("log")
    ax.axvline(1, color=INK_SOFT, lw=0.9)
    ax.set_xlim(0.03, 40)
    ax.set_ylim(len(topic_rows) - 0.5, -0.5)
    ax.set_yticks(range(len(topic_rows)), [row[0] for row in topic_rows], fontsize=6.9, color=INK)
    for i, (_, value, pool_women) in enumerate(topic_rows):
        ax.text(
            value * 1.15,
            i,
            f"×{value:.2f}" if value < 1 else f"×{value:.1f}",
            va="center",
            fontsize=6.4,
            color=INK_SOFT,
            zorder=4,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
        )
        ax.text(
            1.04,
            i,
            f"{pool_women * 100:.0f}%",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=6.4,
            color=ACCENT_TEXT,
        )
    ax.text(
        1.04,
        -0.9,
        "women",
        transform=ax.get_yaxis_transform(),
        va="center",
        ha="left",
        fontsize=6.0,
        color=ACCENT_TEXT,
    )
    ax.set_xticks([0.1, 1, 10], ["×0.1", "×1", "×10"], fontsize=6.6, color=MUTED)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    # C: reach and clicks
    section_y = 5.02
    reach = results["article_reach_by_section"]
    sections = [
        section
        for section in reach
        if reach[section]["features_a_woman"]["articles"] >= MIN_SECTION_STORIES
        and reach[section]["men_only"]["articles"] >= MIN_SECTION_STORIES
    ]
    news = reach.get("News")
    title = (
        "In the News section, readers click stories featuring women more than the "
        "platform shows them"
    )
    bottom = chart_title(
        fig,
        MARGIN,
        section_y,
        title,
        "Stories featuring a woman vs stories naming only men, by section. Left: average times "
        "each story was displayed. Right: clicks per 100 displays. Sections are defined on page "
        "3.",
        6.9,
    )
    draw_rect(fig, MARGIN, bottom + 0.02, 0.11, 0.11, WOMEN)
    draw_text(fig, MARGIN + 0.16, bottom, "Features a woman", size=7, color=INK_SOFT)
    draw_rect(fig, MARGIN + 1.3, bottom + 0.02, 0.11, 0.11, MEN)
    draw_text(fig, MARGIN + 1.46, bottom, "Names only men", size=7, color=INK_SOFT)
    label_width, bar_panel_width, bar_panel_height = 0.95, 2.75, 1.38
    bar_top = bottom + 0.25
    bar_height = 0.36
    rows_y = np.arange(len(sections))
    for k, (key, formatter, x_offset) in enumerate(
        (
            ("mean_displays", lambda value: f"{value:,.0f}", 0.0),
            ("ctr", lambda value: f"{value * 100:.1f}", bar_panel_width + 0.3),
        )
    ):
        ax = add_axes(
            fig, MARGIN + label_width + x_offset, bar_top, bar_panel_width, bar_panel_height
        )
        style_axes(ax, "x")
        women_values = [reach[section]["features_a_woman"][key] or 0 for section in sections]
        men_values = [reach[section]["men_only"][key] or 0 for section in sections]
        ax.barh(rows_y - bar_height / 2 - 0.02, women_values, height=bar_height, color=WOMEN)
        ax.barh(rows_y + bar_height / 2 + 0.02, men_values, height=bar_height, color=MEN)
        max_value = max(women_values + men_values)
        for i in range(len(sections)):
            ax.text(
                women_values[i] + max_value * 0.02,
                rows_y[i] - bar_height / 2 - 0.02,
                formatter(women_values[i]),
                va="center",
                fontsize=6.4,
                color=INK_SOFT,
            )
            ax.text(
                men_values[i] + max_value * 0.02,
                rows_y[i] + bar_height / 2 + 0.02,
                formatter(men_values[i]),
                va="center",
                fontsize=6.4,
                color=INK_SOFT,
            )
        ax.set_xlim(0, max_value * 1.3)
        ax.set_ylim(len(sections) - 0.5, -0.5)
        ax.set_yticks(
            rows_y, sections if k == 0 else [""] * len(sections), fontsize=7.5, color=INK
        )
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                (lambda value, _: f"{value:,.0f}")
                if k == 0
                else (lambda value, _: f"{value * 100:.0f}")
            )
        )
        ax.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(4) if k == 0 else matplotlib.ticker.MultipleLocator(0.02)
        )
        ax.set_xlabel(
            "average displays per story" if k == 0 else "clicks per 100 displays",
            fontsize=6.8,
            color=MUTED,
        )
    if news:
        woman_reach, men_reach = news["features_a_woman"], news["men_only"]
        note = (
            f"News: about the same reach ({woman_reach['mean_displays']:,.0f} vs "
            f"{men_reach['mean_displays']:,.0f} displays per story, medians "
            f"{woman_reach['median_displays']:g} vs {men_reach['median_displays']:g}) but "
            f"{(woman_reach['ctr'] / men_reach['ctr'] - 1) * 100:.0f}% more clicks per display. "
            "More reach can itself lower a click rate, so the Entertainment and Lifestyle gaps "
            "are weaker evidence."
        )
        draw_paragraph(
            fig,
            MARGIN,
            bar_top + bar_panel_height + 0.39,
            note,
            6.9,
            size=6.8,
            color=INK_SOFT,
            line_height=1.25,
        )

    # D: most-shown people
    right_x = 7.85
    draw_vline(fig, right_x - 0.22, 1.95, 7.85)
    concentration = results["concentration"]
    top_women = results["top_women"]
    n_entertainers = sum(1 for person in top_women if person["role"] == "Entertainment & arts")
    other_women = [
        person["name"] for person in top_women if person["role"] != "Entertainment & arts"
    ]
    draw_text(fig, right_x, y0, "The 20 most-shown people", size=9.5, weight="bold")
    subtitle = (
        f"{concentration['top20_women_count']} of 20 are women. {n_entertainers} of the "
        f"{len(top_women)} most-shown women are entertainers or royals"
    )
    subtitle += (
        f". The other, {other_women[0]}, is in politics."
        if len(other_women) == 1
        and top_women
        and [person for person in top_women if person["name"] == other_women[0]][0]["role"]
        == "Politics & government"
        else "."
    )
    list_y = (
        draw_paragraph(
            fig,
            right_x,
            y0 + 0.2,
            subtitle,
            PAGE_WIDTH - MARGIN - right_x,
            size=7.4,
            color=INK_SOFT,
            line_height=1.3,
        )
        + 0.08
    )
    draw_text(
        fig,
        PAGE_WIDTH - MARGIN,
        list_y - 0.02,
        "share of all person appearances shown",
        size=5.9,
        color=MUTED,
        ha="right",
    )
    list_y += 0.15
    for i, person in enumerate(results["top_people"][:20]):
        row_y = list_y + i * 0.16
        draw_dot(
            fig, right_x + 0.05, row_y + 0.06, 0.035, WOMEN if person["gender"] == "F" else MEN
        )
        draw_text(
            fig,
            right_x + 0.16,
            row_y,
            person["name"].split(" (")[0],
            size=7.2,
            weight="bold" if person["gender"] == "F" else "normal",
        )
        draw_text(
            fig,
            PAGE_WIDTH - MARGIN,
            row_y,
            percent(person["share_of_all_appearances"], 1),
            size=7,
            color=INK_SOFT,
            ha="right",
        )
    text_y = list_y + 20 * 0.16 + 0.18
    draw_hline(fig, right_x, PAGE_WIDTH - MARGIN, text_y - 0.1)
    top_person = results["top_people"][0]
    draw_text(
        fig,
        right_x,
        text_y,
        percent(top_person["share_of_all_appearances"], 1),
        size=12,
        weight="bold",
    )
    text_y = (
        draw_paragraph(
            fig,
            right_x,
            text_y + 0.25,
            f"of all person appearances shown went to one man, {top_person['name']}.",
            PAGE_WIDTH - MARGIN - right_x,
            size=6.9,
            color=INK_SOFT,
            line_height=1.25,
        )
        + 0.14
    )

    # Questions this raises
    def mult_word(ratio, up, down):
        """Name a ratio in words."""
        for low, high, word in MULTIPLIER_WORDS:
            if low <= ratio < high:
                return word
        return f"{up if ratio > 1 else down} ×{ratio:.1f}"

    role_share_rows = role_shares(results)
    group_names = [group[0] for group in ROLE_GROUPS]
    entertainment_ratio = (
        role_share_rows[1][group_names.index("Entertainment")]
        / role_share_rows[0][group_names.index("Entertainment")]
    )
    expert_ratio = (
        role_share_rows[1][group_names.index("Experts")]
        / role_share_rows[0][group_names.index("Experts")]
    )
    questions = []
    if news:
        woman_reach, men_reach = news["features_a_woman"], news["men_only"]
        questions.append(
            "Why do News stories featuring women earn "
            f"{(woman_reach['ctr'] / men_reach['ctr'] - 1) * 100:.0f}% more clicks per display, "
            "yet get no extra reach?"
        )
    questions.append(
        f"If a feed {mult_word(entertainment_ratio, 'scales', 'scales')} its entertainers and "
        f"{mult_word(expert_ratio, 'scales', 'scales')} its experts, whose expertise do readers "
        "learn to trust?"
    )
    questions.append(
        "Would a Canadian newsroom's feed show the same topic-mix effect, or a different one?"
    )
    draw_text(fig, right_x, text_y, "Questions this raises", size=9.5, weight="bold")
    text_y += 0.24
    for k, question in enumerate(questions, 1):
        draw_text(fig, right_x, text_y, f"{k}", size=7.4, weight="bold", color=ACCENT_TEXT)
        text_y = (
            draw_paragraph(
                fig,
                right_x + 0.15,
                text_y,
                question,
                PAGE_WIDTH - MARGIN - right_x - 0.15,
                size=7.0,
                color=INK,
                line_height=1.25,
            )
            + 0.06
        )
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 3: audit scorecard
# ---------------------------------------------------------------------------

MATERIAL_CHANGE = 0.05  # Within 5% means unchanged


def excludes_one(row, key):
    """True if interval excludes one."""
    ci = row.get(key + "_ci95")
    return bool(ci) and ci[0] is not None and (ci[0] > 1 or ci[1] < 1)


def direction(value):
    """Sign of a material change."""
    if (
        value is None
        or not np.isfinite(value)
        or abs(round(value, 2) - 1) < MATERIAL_CHANGE - EPSILON
    ):
        return 0
    return 1 if value > 1 else -1


def page3(pdf, results, preview):
    """Page 3: audit scorecard."""
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
    page_frame(fig, 3, results, preview)
    scorecard = [
        row
        for row in results["scorecard"]
        if row["group"] != "Other" and row["published"] and row["published"] >= MIN_SCORECARD_SHARE
    ]
    science, media, entertainment = (
        scorecard_row(results, "Science, health & academia"),
        scorecard_row(results, "Media & writing"),
        scorecard_row(results, "Entertainment & arts"),
    )
    short_region = {"Latin America & Caribbean": "Latin America"}
    losers = [
        short_region.get(row["group"], row["group"])
        for row in scorecard
        if row["family"].startswith("Citizenship")
        and row["ratio_platform"] <= CLEAR_LOSS_RATIO
        and row["ratio_reader"] <= CLEAR_LOSS_RATIO
        and excludes_one(row, "ratio_platform")
        and excludes_one(row, "ratio_reader")
    ]
    canada = scorecard_row(results, "Canada")
    takeaway = (
        "Entertainers gained the most at the platform step "
        f"(×{entertainment['ratio_platform']:.2f}). What readers miss: experts lost at both steps "
        f"(science, health and academia ×{science['ratio_platform']:.2f} then "
        f"×{science['ratio_reader']:.2f}, media and writing ×{media['ratio_platform']:.2f} then "
        f"×{media['ratio_reader']:.2f})"
    )
    if losers:
        loser_names = ", ".join(losers[:-1]) + (
            " and " + losers[-1] if len(losers) > 1 else losers[0]
        )
        takeaway += f", and so did people from {loser_names}."
    else:
        takeaway += "."
    if canada and "Canada" not in losers and canada["ratio_reader"] <= CLEAR_LOSS_RATIO:
        takeaway += (
            " Canadian public figures lost mostly at the reader step "
            f"(×{canada['ratio_reader']:.2f})."
        )
    section_head(
        fig,
        "THE AUDIT SCORECARD   ·   CHALLENGE QUESTIONS 2 AND 4",
        "Who gains and who loses at each step",
        takeaway,
    )

    family_titles = {
        "Gender": "WOMEN, SHARE OF PEOPLE WITH A RECORDED GENDER",
        "Role (share of all people)": "ROLES, SHARE OF ALL PEOPLE NAMED",
        "Citizenship (share of people with one)": "CITIZENSHIP, SHARE OF PEOPLE WITH ONE RECORDED",
    }
    rows, last_family = [], None
    for row in scorecard:
        if row["family"] != last_family and row["family"] in family_titles:
            rows.append(("header", family_titles[row["family"]]))
        last_family = row["family"]
        rows.append(("row", row))
    n_rows = len(rows)
    row_height = min(0.205, 4.75 / n_rows)
    y_top = 2.22
    x_name, x_pub = MARGIN, MARGIN + 2.5
    axis_width = 1.5
    axis_x = [MARGIN + 2.7, MARGIN + 4.65]
    draw_text(fig, x_name, 1.98, "Group", size=7, weight="bold", color=MUTED)
    draw_text(fig, x_pub, 1.98, "In pool", size=7, weight="bold", color=MUTED, ha="right")
    draw_text(
        fig, axis_x[0], 1.98, "Platform step (shown ÷ pool)", size=7, weight="bold", color=MUTED
    )
    draw_text(
        fig, axis_x[1], 1.98, "Reader step (clicked ÷ shown)", size=7, weight="bold", color=MUTED
    )
    low_limit, high_limit = 0.35, 3.6
    axes = []
    for axis_left in axis_x:
        ax = add_axes(fig, axis_left, y_top - row_height / 2, axis_width, n_rows * row_height)
        ax.set_xscale("log")
        ax.set_xlim(low_limit, high_limit)
        ax.set_ylim(n_rows - 0.5, -0.5)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.set_yticks([])
        ticks = [0.5, 1, 2, 3]
        ax.set_xticks(ticks, ["×0.5", "×1", "×2", "×3"], fontsize=6.5, color=MUTED)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.tick_params(length=0)
        for tick in [0.5, 0.67, 1.5, 2, 3]:
            ax.axvline(tick, color=GRID, lw=0.6, zorder=0)
        ax.axvline(1, color=INK_SOFT, lw=0.9, zorder=1)
        ax.set_facecolor("none")
        axes.append(ax)
    for i, (kind, row) in enumerate(rows):
        row_y = y_top + i * row_height
        if kind == "header":
            draw_text(fig, x_name, row_y - 0.06, row, size=6.4, weight="bold", color=MUTED)
            continue
        if i % 2 == 0:
            draw_rect(
                fig,
                MARGIN - 0.05,
                row_y - row_height / 2,
                axis_x[1] + axis_width + 0.5 - MARGIN,
                row_height,
                "#f7f9fb",
                zorder=-5,
            )
        indent = 0.14 if row["family"] == "Gender by section" else 0.0
        draw_text(
            fig,
            x_name + indent,
            row_y - 0.065,
            row["group"],
            size=7.4,
            weight="bold" if row["family"] == "Gender" else "normal",
        )
        draw_text(
            fig,
            x_pub,
            row_y - 0.065,
            percent(row["published"], 1),
            size=7.2,
            color=INK_SOFT,
            ha="right",
        )
        for ax, key, axis_left in (
            (axes[0], "ratio_platform", axis_x[0]),
            (axes[1], "ratio_reader", axis_x[1]),
        ):
            value = row[key]
            if value is None or not np.isfinite(value):
                continue
            ci = row.get(key + "_ci95") or [None, None]
            sign = direction(value)
            marker_color = {1: GAIN_COLOR, -1: LOSS_COLOR, 0: MUTED}[sign]
            if ci[0] is not None:
                ax.hlines(
                    i,
                    max(ci[0], low_limit),
                    min(ci[1], high_limit),
                    color=marker_color,
                    lw=1.6,
                    alpha=0.45,
                    zorder=2,
                )
            filled = excludes_one(row, key)
            ax.scatter(
                [min(max(value, low_limit * 1.02), high_limit / 1.02)],
                [i],
                s=30,
                zorder=3,
                facecolor=marker_color if filled else "white",
                edgecolor=marker_color,
                linewidth=1.3,
            )
            strong = filled and sign != 0
            draw_text(
                fig,
                axis_left + axis_width + 0.06,
                row_y - 0.065,
                f"×{value:.2f}",
                size=7,
                color=INK if strong else MUTED,
                weight="bold" if strong else "normal",
            )
    bottom = y_top + n_rows * row_height + 0.2
    draw_dot(fig, MARGIN + 0.06, bottom + 0.06, 0.045, GAIN_COLOR)
    draw_text(fig, MARGIN + 0.16, bottom, "gained", size=6.8, color=INK_SOFT)
    draw_dot(fig, MARGIN + 0.72, bottom + 0.06, 0.045, LOSS_COLOR)
    draw_text(fig, MARGIN + 0.82, bottom, "lost", size=6.8, color=INK_SOFT)
    draw_dot(fig, MARGIN + 1.22, bottom + 0.06, 0.045, MUTED)
    draw_text(fig, MARGIN + 1.32, bottom, "within 5% of ×1", size=6.8, color=INK_SOFT)
    all_clear = all(
        excludes_one(row, k)
        for kind, row in rows
        if kind == "row"
        for k in ("ratio_platform", "ratio_reader")
        if row[k] is not None
    )
    explanation = (
        "Funnel ratio: a group's share at a stage ÷ its share at the stage before. ×1 means the "
        "step kept the group as it found it. Every 95% interval excludes ×1, but intervals "
        "resample readers, not stories, so they are too narrow for groups seen in few stories. "
        "Gray marks shifts too small to matter."
        if all_clear
        else "Funnel ratio: a group's share at a stage ÷ its share at the stage before. "
        "×1 means the step kept the group as it found it. Filled: the 95% interval excludes ×1."
    )
    text_bottom = draw_paragraph(
        fig, MARGIN + 2.4, bottom, explanation, 4.15, size=6.8, color=INK_SOFT, line_height=1.25
    )
    categories = results["category"]
    entertainment_categories = ("entertainment", "tv", "music", "movies", "video", "kids")
    video_pool = categories["video"]["published"]["gendered_appearances"] / sum(
        categories[category]["published"]["gendered_appearances"]
        for category in entertainment_categories
    )
    video_shown = categories["video"]["shown"]["gendered_appearances"] / sum(
        categories[category]["shown"]["gendered_appearances"]
        for category in entertainment_categories
    )
    draw_paragraph(
        fig,
        MARGIN,
        max(text_bottom, bottom + 0.3) + 0.06,
        (
            "Sections group MIND's categories: Money is finance, Entertainment is entertainment, "
            "TV, music, movies, video and kids, Lifestyle is lifestyle, food, travel, health, "
            f"autos and weather. MSN's video channel is {percent(video_pool)} of Entertainment's "
            f"pool appearances but {percent(video_shown)} of what it shows, so the Women in "
            "Entertainment ratio largely reflects that mix."
        ),
        6.8,
        size=6.8,
        color=INK_SOFT,
        line_height=1.25,
    )

    # Right column: roles
    right_x = 7.45
    right_width = PAGE_WIDTH - MARGIN - right_x
    draw_vline(fig, right_x - 0.2, 1.95, 7.85)
    roles = results["roles"]
    women_entertainers = roles["shown"]["Entertainment & arts"]["share_of_women"]
    bottom = chart_title(
        fig,
        right_x,
        1.95,
        "As whom do women appear on screen?",
        "Share of women's and of men's appearances shown, one role per person from Wikidata "
        "occupations. Entertainers: "
        f"{percent(roles['published']['Entertainment & arts']['share_of_women'])} of women's "
        f"appearances in the pool, {percent(women_entertainers)} on screen.",
        right_width,
    )
    role_list = [
        "Politics & government",
        "Business",
        "Science, health & academia",
        "Law & justice",
        "Media & writing",
        "Sports",
        "Entertainment & arts",
    ]
    short_role = {
        "Science, health & academia": "Science & health",
        "Politics & government": "Politics",
        "Entertainment & arts": "Entertainment",
        "Media & writing": "Media & writing",
        "Law & justice": "Law",
    }
    label_width = 1.0
    ax = add_axes(fig, right_x + label_width, bottom + 0.08, right_width - label_width, 1.75)
    women_values = [roles["shown"][row]["share_of_women"] or 0 for row in role_list]
    men_values = [roles["shown"][row]["share_of_men"] or 0 for row in role_list]
    rows_y = np.arange(len(role_list))
    ax.barh(rows_y, [-value for value in women_values], color=WOMEN, height=0.58)
    ax.barh(rows_y, men_values, color=MEN, height=0.58)
    ax.axvline(0, color=INK_SOFT, lw=0.8)
    limit = max(women_values + men_values) * 1.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(len(role_list) - 0.5, -0.5)
    for i in range(len(role_list)):
        ax.text(
            -women_values[i] - limit * 0.03,
            i,
            percent(women_values[i]),
            ha="right",
            va="center",
            fontsize=6.5,
            color=INK_SOFT,
        )
        ax.text(
            men_values[i] + limit * 0.03,
            i,
            percent(men_values[i]),
            ha="left",
            va="center",
            fontsize=6.5,
            color=INK_SOFT,
        )
    ax.axis("off")
    for i, row in enumerate(role_list):
        draw_text(
            fig,
            right_x + label_width - 0.05,
            bottom + 0.08 + (i + 0.5) * 1.75 / len(role_list) - 0.06,
            short_role.get(row, row),
            size=6.9,
            ha="right",
        )
    middle_x = right_x + label_width + (right_width - label_width) / 2
    draw_text(
        fig,
        middle_x - 0.08,
        bottom + 1.9,
        "Women",
        size=7,
        weight="bold",
        color=ACCENT_TEXT,
        ha="right",
    )
    draw_text(fig, middle_x + 0.08, bottom + 1.9, "Men", size=7, weight="bold", color=INK_SOFT)
    dissimilarity = results["role_dissimilarity"]
    draw_paragraph(
        fig,
        right_x,
        bottom + 2.12,
        (
            f"Typecast index (0 = same roles, 1 = no overlap): {dissimilarity['published']:.2f} "
            f"in the pool, {dissimilarity['shown']:.2f} shown, {dissimilarity['clicked']:.2f} "
            "clicked."
        ),
        right_width,
        size=6.9,
        color=INK_SOFT,
        line_height=1.28,
    )

    # Right column: geography
    geography = results["geography"]
    visibility = geography["visibility_index_shown"]
    geography_y = 5.12
    draw_hline(fig, right_x, PAGE_WIDTH - MARGIN, geography_y - 0.13)
    global_south = ["Asia", "Africa", "Latin America & Caribbean"]
    south_population = sum(geography["population_share"][row] for row in global_south)
    south_shown = sum(geography["share"]["shown"][row] for row in global_south)
    bottom = chart_title(
        fig,
        right_x,
        geography_y,
        "Whose world?",
        "Citizenship of the people named, not where stories happen, grouped by Wikidata continent "
        "(dual citizens split evenly). Share of people shown ÷ share of the 2019 world population "
        f"(UN). Asia, Africa and Latin America: {percent(south_population)} of the world, "
        f"{percent(south_shown, 1)} of the people shown.",
        right_width,
    )
    regions = sorted(visibility, key=lambda row: -(visibility[row] or 0))
    ax = add_axes(fig, right_x + 1.0, bottom + 0.05, right_width - 1.3, 1.2)
    geo_bottom = bottom + 0.05 + 1.2
    style_axes(ax, "x")
    bar_values = [max(visibility[row] or LOG_FLOOR, LOG_FLOOR) for row in regions]
    ax.barh(
        range(len(regions)),
        bar_values,
        color=[
            STAGE_COLORS["shown"] if row == "Canada" else STAGE_COLORS["published"]
            for row in regions
        ],
        height=0.58,
    )
    ax.set_xscale("log")
    ax.axvline(1, color=INK_SOFT, lw=0.9)
    ax.set_ylim(len(regions) - 0.5, -0.5)
    ax.set_yticks(
        range(len(regions)),
        [short_region.get(row, row) for row in regions],
        fontsize=6.9,
        color=INK,
    )
    for i, value in enumerate(bar_values):
        ax.text(
            value * 1.15,
            i,
            f"{value:.2f}×" if value < 10 else f"{value:.0f}×",
            va="center",
            fontsize=6.5,
            color=INK_SOFT,
        )
    ax.set_xlim(min(bar_values) / 1.6, max(bar_values) * 5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:g}×"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    draw_paragraph(
        fig,
        right_x,
        geo_bottom + 0.27,
        (
            "Not measured: race, ethnicity and Indigenous identity. Wikidata records ethnicity "
            "for under 1.2% of people (Kronk 2026), and guessing would break the no-inference "
            "rule."
        ),
        right_width,
        size=6.8,
        color=INK_SOFT,
        line_height=1.25,
    )
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 4: personalization and trade-off
# ---------------------------------------------------------------------------


def page4(pdf, results, preview):
    """Page 4: personalization and trade-off."""
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
    page_frame(fig, 4, results, preview)
    personalization, feedback = results["personalization"], results["feedback_loop"]
    recommenders = results.get("recommenders")
    quintiles = feedback["quintiles"]
    communities = results["communities"]
    large_communities = {
        community: value
        for community, value in communities.items()
        if value["users"] >= MIN_COMMUNITY_READERS
        and value["women_share_shown"] is not None
        and community not in ("Mixed interests", "No click history")
    }
    lowest = min(
        large_communities, key=lambda community: large_communities[community]["women_share_shown"]
    )
    highest = max(
        large_communities, key=lambda community: large_communities[community]["women_share_shown"]
    )
    takeaway = (
        "Readers see different worlds. Among readers shown at least "
        f"{results['meta']['min_people_per_user']} named people, one in ten saw "
        f"{percent(personalization['observed_p10'], 1)} women or fewer, while another one in "
        f"ten saw {percent(personalization['observed_p90'], 1)} or more. {hyphenate(lowest)} "
        "readers saw "
        f"{percent(large_communities[lowest]['women_share_shown'], 1)} women, "
        f"{hyphenate(highest).lower()} readers "
        f"{percent(large_communities[highest]['women_share_shown'], 1)}. Past clicks predict it "
        f"(rank correlation {feedback['spearman_observed']:.2f}, against "
        f"{feedback['spearman_null_mean']:.2f} with personalization shuffled away), and twice as "
        f"many readers see under {personalization['low_threshold'] * 100:.0f}% women as when it "
        f"is shuffled away ({percent(personalization['observed_share_below_low'], 2)} against "
        f"{percent(personalization['null_share_below_low'], 2)})."
    )
    section_head(
        fig,
        "YOUR FEED, YOUR GAP   ·   CHALLENGE QUESTIONS 1 AND 3",
        "Your past clicks predict who you see",
        takeaway,
    )

    # A: spread versus null
    y0 = 2.05
    bottom = chart_title(
        fig,
        MARGIN,
        y0,
        "How often each reader saw women in the news",
        (
            "Share of women among people shown to each of "
            f"{thousands(personalization['eligible_users'])} readers (purple) vs the same feeds "
            "shuffled between readers active in the same hour, which removes personalization but "
            f"keeps the news cycle (gray line, {results['meta']['null_reps']} shuffles). Spread: "
            f"×{personalization['excess_spread']:.2f} wider than chance. In "
            f"{percent(personalization['impressions_with_people_but_no_woman'], 1)} of feed "
            "refreshes that named people, none was a woman."
        ),
        4.6,
    )
    ax = add_axes(fig, MARGIN, bottom + 0.1, 4.6, 1.5)
    style_axes(ax, "y")
    bin_edges = np.array(personalization["hist_bins"])
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.bar(
        centers,
        personalization["hist_observed"],
        width=(bin_edges[1] - bin_edges[0]) * 0.86,
        color=WOMEN,
    )
    ax.step(bin_edges[:-1], personalization["hist_null"], where="post", color=INK_SOFT, lw=1.4)
    ax.xaxis.set_major_formatter(PERCENT_FORMAT)
    ax.yaxis.set_major_formatter(PERCENT_FORMAT)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.04))
    visible_centers = [
        center
        for center, observed, null in zip(
            centers, personalization["hist_observed"], personalization["hist_null"]
        )
        if observed > 0.002 or null > 0.002
    ]
    ax.set_xlim(max(0, min(visible_centers) - 0.05), min(1, max(visible_centers) + 0.05))
    ax.set_xlabel("share of women among people shown to the reader", fontsize=7, color=MUTED)

    # B: feedback loop
    x1 = 5.85
    right_bottom = chart_title(
        fig,
        x1,
        y0,
        "Past clicks predict what you are shown",
        (
            f"{thousands(feedback['users'])} readers grouped into fifths by the share of women in "
            "their past clicks. Purple: what they were shown. Gray: the same readers with "
            "personalization shuffled away."
        ),
        4.6,
    )
    ax = add_axes(fig, x1 + 0.1, right_bottom + 0.1, 4.45, 1.65)
    style_axes(ax, "y")
    history_x = [point["history_women_share_mean"] for point in quintiles]
    ax.plot(
        history_x,
        [point["shown_women_share_mean"] for point in quintiles],
        color=WOMEN,
        lw=2,
        marker="o",
        ms=6,
        markeredgecolor="white",
        markeredgewidth=1.5,
    )
    ax.plot(
        history_x,
        [point["null_shown_women_share_mean"] for point in quintiles],
        color=NULL_COLOR,
        lw=1.6,
        marker="o",
        ms=5,
        markeredgecolor="white",
        markeredgewidth=1.2,
    )
    ax.text(
        history_x[-1] + 0.015,
        quintiles[-1]["shown_women_share_mean"],
        f"{percent(quintiles[-1]['shown_women_share_mean'], 1)} shown",
        fontsize=7,
        color=INK_SOFT,
        va="center",
    )
    ax.text(
        history_x[0] + 0.015,
        quintiles[0]["shown_women_share_mean"] - 0.006,
        f"{percent(quintiles[0]['shown_women_share_mean'], 1)} shown",
        fontsize=7,
        color=INK_SOFT,
        va="top",
    )
    ax.text(
        history_x[-1] + 0.015,
        quintiles[-1]["null_shown_women_share_mean"],
        "no personalization",
        fontsize=7,
        color=INK_SOFT,
        va="center",
    )
    ax.set_xlim(0, max(history_x) + 0.2)
    low = min(point["shown_women_share_mean"] for point in quintiles)
    high = max(point["shown_women_share_mean"] for point in quintiles)
    ax.set_ylim(low - 0.025, high + 0.02)
    ax.xaxis.set_major_formatter(PERCENT_FORMAT)
    ax.yaxis.set_major_formatter(PERCENT_FORMAT)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.set_xlabel(
        "share of women among people in the reader's past clicks", fontsize=7, color=MUTED
    )

    # C: communities
    lower_y = 4.95
    shown_communities = {
        community: value
        for community, value in results["communities"].items()
        if value["users"] >= MIN_COMMUNITY_READERS and value["women_share_shown"] is not None
    }
    order = sorted(
        shown_communities, key=lambda community: shown_communities[community]["women_share_shown"]
    )
    bottom = chart_title(
        fig,
        MARGIN,
        lower_y,
        "Reader communities, defined only by what they click",
        (
            "MIND has no reader demographics, so we infer none. A community is the section behind "
            "at least half of a reader's past clicks. Gray: readers with no click history in "
            "MIND, a natural baseline. Line: all readers, "
            f"{percent(results['funnel_gender']['women_share']['shown'], 1)}. Light and heavy "
            f"readers saw {percent(results['activity']['Light']['women_share_shown'], 1)} and "
            f"{percent(results['activity']['Heavy']['women_share_shown'], 1)}, so the gap follows "
            "interests, not activity."
        ),
        4.6,
    )
    ax = add_axes(fig, MARGIN + 1.3, bottom + 0.05, 3.3, 1.6)
    style_axes(ax, "x")
    values = [shown_communities[community]["women_share_shown"] for community in order]
    ax.barh(
        range(len(order)),
        values,
        color=[
            MEN if community == "No click history" else STAGE_COLORS["shown"]
            for community in order
        ],
        height=0.58,
    )
    all_readers = results["funnel_gender"]["women_share"]["shown"]
    ax.axvline(all_readers, color=INK_SOFT, lw=1.0, zorder=2)
    for i, value in enumerate(values):
        ax.text(
            value + 0.006,
            i,
            f"{percent(value, 1)}  ({thousands(shown_communities[order[i]]['users'])} readers)",
            va="center",
            fontsize=6.6,
            color=INK_SOFT,
            zorder=3,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
        )
    ax.set_yticks(
        range(len(order)), [hyphenate(community) for community in order], fontsize=7.4, color=INK
    )
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(0, max(values) * 1.55)
    ax.xaxis.set_major_formatter(PERCENT_FORMAT)
    ax.set_xlabel(
        "share of women among people each community was shown", fontsize=6.8, color=MUTED
    )

    # D: relevance versus representation
    if recommenders:
        sweep = recommenders["tradeoff_hybrid"]
        top5 = {name: model["top5_women_share"] for name, model in recommenders["models"].items()}
        parity = parity_point(sweep)
        right_bottom = chart_title(
            fig,
            x1,
            lower_y,
            "Does relevance have to cost representation?",
            (
                "Baselines trained on the first six days, tested on "
                f"{thousands(recommenders['impressions'])} feed refreshes from day seven. Line: "
                "our best model (hybrid) with a growing boost for stories featuring a woman. "
                f"Top-5 women: personalized {percent(top5['Personalized (reading history)'], 1)}, "
                f"random {percent(top5['Random order'], 1)}, "
                f"most popular {percent(top5['Most popular'], 1)} (lowest)."
            ),
            4.6,
        )
        ax = add_axes(fig, x1 + 0.45, right_bottom + 0.05, 4.1, 1.42)
        style_axes(ax, "both")
        ax.plot(
            [point["top5_women_share"] for point in sweep],
            [point["auc"] for point in sweep],
            color=STAGE_COLORS["shown"],
            lw=1.8,
            marker="o",
            ms=4,
            markeredgecolor="white",
            markeredgewidth=1,
            zorder=3,
        )
        offsets = {
            "Random order": (4, 3),
            "Most popular": (4, 3),
            "Personalized (reading history)": (4, -9),
            "Hybrid (history + popularity)": (-6, 6),
        }
        for name, model in recommenders["models"].items():
            ax.scatter(
                [model["top5_women_share"]],
                [model["auc"]],
                s=34,
                color=INK_SOFT,
                zorder=4,
                edgecolor="white",
                linewidth=1.2,
            )
            ax.annotate(
                name.split(" (")[0],
                (model["top5_women_share"], model["auc"]),
                xytext=offsets.get(name, (4, 3)),
                textcoords="offset points",
                fontsize=6.5,
                color=INK_SOFT,
                ha="right" if offsets.get(name, (4, 3))[0] < 0 else "left",
            )
        ax.axvline(PARITY, color=WOMEN, lw=1.3)
        low_auc = min(model["auc"] for model in recommenders["models"].values())
        high_auc = max(point["auc"] for point in sweep)
        ax.set_ylim(low_auc - 0.012, high_auc + 0.012)
        ax.text(
            0.505, high_auc + 0.009, "half women (average)", color=INK_SOFT, fontsize=6.5, va="top"
        )
        ax.xaxis.set_major_formatter(PERCENT_FORMAT)
        ax.set_xlabel(
            "women's share of people in each reader's top 5 stories", fontsize=6.8, color=MUTED
        )
        ax.set_ylabel("accuracy (AUC)", fontsize=6.8, color=MUTED)
        if parity:
            reach = results["article_reach_by_section"]
            women_stories = sum(value["features_a_woman"]["articles"] for value in reach.values())
            soft_stories = sum(
                reach[community]["features_a_woman"]["articles"]
                for community in ("Entertainment", "Lifestyle")
                if community in reach
            )
            note = (
                "Raising women to half the people in top-5 lists, on average, costs "
                f"{(sweep[0]['auc'] - parity['auc']) * 100:.1f} AUC points ({sweep[0]['auc']:.3f} "
                f"to {parity['auc']:.3f}) and cuts lists that name people but no woman from "
                f"{percent(sweep[0]['top5_with_people_but_no_woman'])} to "
                f"{percent(parity['top5_with_people_but_no_woman'])}. "
                f"{percent(soft_stories / women_stories)} of stories featuring a woman are "
                "Entertainment or Lifestyle, so a blanket boost leans toward them. Phase 2 tests "
                "a boost within each section."
            )
            draw_paragraph(
                fig,
                x1 + 0.1,
                right_bottom + 1.84,
                note,
                4.5,
                size=7,
                color=INK,
                weight="bold",
                line_height=1.3,
            )
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 5: method and limits
# ---------------------------------------------------------------------------


def page5(pdf, results, preview):
    """Page 5: method and limits."""
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
    page_frame(fig, 5, results, preview)
    data, coverage, meta, article_coverage = (
        results["data"],
        results["people_coverage"],
        results["meta"],
        results["article_coverage"],
    )
    section_head(
        fig,
        "DATA SOURCES, METHODOLOGY AND LIMITATIONS   ·   CHALLENGE QUESTION 5",
        "How we measured it, and what the data cannot tell us",
        "Every computed number in this report can be reproduced from public data with our two "
        "Python scripts. Any platform that logs stories, the people they name, and what was shown "
        "and clicked can compute the same six indicators."
        + (
            f" Code: {TEAM['code_url'].replace('https://', '').rstrip('/')}"
            if TEAM.get("code_url") and "[" not in TEAM["code_url"]
            else ""
        ),
    )

    y0 = 1.72
    steps = [
        (
            dataset_name(results),
            f"{thousands(data['users'])} readers, {thousands(data['impression_logs'])} feed "
            f"refreshes, {thousands(data['displays'])} displays",
        ),
        ("Entity links", "Microsoft's Wikidata links in every headline and abstract"),
        (
            "Wikidata",
            f"{thousands(coverage['humans'])} linked people in the catalog, "
            f"{thousands(results['concentration']['people_shown'])} in stories shown",
        ),
        ("Three stages", "Pool, shown, clicked. Funnel ratios with reader bootstrap"),
        ("Tests", "Topic-mix split, shuffle null, feedback loop, recommender trade-off"),
    ]
    box_width = (PAGE_WIDTH - 2 * MARGIN - 4 * 0.18) / 5
    for i, (step_title, step_text) in enumerate(steps):
        x = MARGIN + i * (box_width + 0.18)
        draw_rect(fig, x, y0, box_width, 0.62, PANEL, radius=0.06)
        draw_text(fig, x + 0.1, y0 + 0.08, step_title, size=8.2, weight="bold")
        draw_paragraph(
            fig,
            x + 0.1,
            y0 + 0.27,
            step_text,
            box_width - 0.2,
            size=6.6,
            color=INK_SOFT,
            line_height=1.25,
        )
        if i < 4:
            draw_arrow(
                fig,
                x + box_width + 0.02,
                x + box_width + 0.16,
                y0 + 0.31,
                color=MUTED,
                lw=0.8,
                scale=7,
            )

    left_x, left_width = MARGIN, 5.3
    y = 2.5
    disclosures = [
        (
            "Data sources",
            f"Microsoft News Dataset ({dataset_name(results)}, Wu et al., ACL 2020, research "
            "license), the benchmark behind 57% of news recommender papers from 2022 to 2024: "
            f"anonymized MSN News logs of {thousands(data['users'])} readers, {date_span(data)}, "
            "training and validation splits (the test split has no click labels). Wikidata (CC0, "
            f"retrieved {meta['generated_utc'][:7]}) for each linked person's gender (P21), "
            "occupation (P106), citizenship (P27) and that country's continent (P30). UN World "
            "Population Prospects 2019.",
        ),
        (
            "What it represents, and what it does not",
            "What Microsoft News displayed and what its readers clicked during one week of 2019, "
            "in English. Reader location is not recorded and "
            f"{percent(results['geography']['share']['shown']['United States'])} of the people "
            "shown are US citizens, so this is a US-centred sample, not a Canadian one. It sees "
            "only public figures with a Wikidata page that Microsoft's tagger linked: "
            f"{percent(article_coverage['share_with_person'])} of stories name at least one. A "
            "mention is not a quote. MSN's displays may mix editors' and algorithms' choices. "
            "MIND has no publisher or stance labels, so we measure whose voices appear (roles), "
            "not sources or viewpoints.",
        ),
        (
            "Our proxy for information exposure",
            "A display: one story shown to a reader inside a feed refresh (MIND's impression "
            "log). We assume a displayed story was seen, since MIND has no scroll or position "
            "data. Each person Microsoft's entity links find in the headline or abstract is one "
            "person appearance. The pool counts each story in circulation once. Clicked counts "
            "the displays a reader clicked.",
        ),
        (
            "Limitations and biases",
            "In the data: MIND samples MSN readers with at least 5 clicks in six weeks, so light "
            "readers are under-represented. People named only in a story's body are missed. "
            "Wikidata under-records women (28.7% of humans with a stated gender, Kronk 2026), so "
            "we read changes between stages, not levels, and its attributes are as of 2026 "
            f"({percent(coverage['humans_with_occupation'] / coverage['humans'])} have an "
            f"occupation, {percent(coverage['humans_with_citizenship'] / coverage['humans'])} a "
            "citizenship). Gender is as recorded, never guessed from names, with "
            f"{coverage['humans_other_gender']} people outside the woman and man categories. "
            "Readers have no demographics, so none are inferred. In our method: intervals "
            "resample readers, not stories, so they are too narrow for groups seen in few "
            "stories. One role per person by a fixed rule (Prince Harry lands in Sports). Results "
            "are associations, not causes.",
        ),
        (
            "Why this metric is appropriate",
            "Fairness needs a reference point. Our main one is the pool: the funnel ratio "
            "compares each stage with the one before it, so it isolates which step changes "
            "representation and judges the platform only on what it adds. We also compare with "
            "world population (visibility index) and with parity (half women, the trade-off "
            "target). Six indicators of bias or unequal outcomes: funnel ratio per group (women "
            f"×{results['funnel_gender']['ratio_shown_vs_published']:.2f} at the platform step), "
            "typecast index (Duncan dissimilarity over roles, "
            f"{results['role_dissimilarity']['published']:.2f} to "
            f"{results['role_dissimilarity']['shown']:.2f}), visibility index (Africa "
            f"{results['geography']['visibility_index_shown']['Africa']:.2f}×), top-10 share of "
            f"attention ({percent(results['concentration']['top10_share_of_appearances'], 1)}), "
            "narrow-feed rate "
            f"({percent(results['personalization']['observed_share_below_low'], 2)} of readers "
            f"under {results['personalization']['low_threshold'] * 100:.0f}% women, "
            f"{percent(results['personalization']['null_share_below_low'], 2)} by chance) and "
            "feed refreshes that name people but no woman "
            f"({percent(results['personalization']['impressions_with_people_but_no_woman'], 1)}). "
            f"Shifts within 5% of ×1 count as no change. {meta['bootstrap_reps']} reader "
            "bootstraps (the pool is a full count, so intervals cover the shown and clicked "
            f"stages) and {meta['null_reps']} same-hour shuffles.",
        ),
    ]
    for i, (heading, body) in enumerate(disclosures, 1):
        draw_text(fig, left_x, y, f"{i}", size=13, weight="bold", color=ACCENT_TEXT)
        draw_text(fig, left_x + 0.3, y + 0.02, heading, size=8.6, weight="bold")
        y = (
            draw_paragraph(
                fig,
                left_x + 0.3,
                y + 0.2,
                body,
                left_width - 0.3,
                size=6.85,
                color=INK_SOFT,
                line_height=1.22,
            )
            + 0.035
        )
    draw_hline(fig, left_x, left_x + left_width, y)
    draw_text(fig, left_x, y + 0.06, "How this improves on existing work", size=8.6, weight="bold")
    prior = [
        (
            "Gender Gap Tracker",
            "who Canadian outlets quote at publication, with gender assigned from names.",
        ),
        (
            "RADio",
            "recommendations vs the pool or reader history: topics, viewpoints, minority voices.",
        ),
        ("Vrijenhoek (2023)", "MIND's pipeline stages, for article categories only."),
        (
            "FairRec, EB-NeRD",
            "what men and women readers receive, which needs reader demographics.",
        ),
        (
            "Funnel Vision",
            "recorded attributes of the people in stories, pool to click, no reader demographics.",
        ),
    ]
    prior_y = y + 0.27
    for k, (who, what) in enumerate(prior):
        draw_text(
            fig,
            left_x,
            prior_y,
            who,
            size=7.1,
            weight="bold",
            color=INK if k == len(prior) - 1 else INK_SOFT,
        )
        prior_y = (
            draw_paragraph(
                fig,
                left_x + 1.25,
                prior_y,
                what,
                left_width - 1.25,
                size=7.1,
                color=INK if k == len(prior) - 1 else INK_SOFT,
                line_height=1.22,
            )
            + 0.02
        )

    # Robustness table
    right_x = 6.2
    right_width = PAGE_WIDTH - MARGIN - right_x
    draw_text(
        fig,
        right_x,
        2.5,
        "Robustness: does the result survive other choices?",
        size=8.6,
        weight="bold",
    )
    robust = results["robustness"]
    without_sports = {
        stage: share_without_sports(results, stage) for stage in ("published", "shown", "clicked")
    }
    rows = [
        ("Main analysis", robust["main (titles + abstracts)"]),
        (
            "Excluding Sports",
            {
                "published": without_sports["published"],
                "shown": without_sports["shown"],
                "clicked": without_sports["clicked"],
            },
        ),
        ("Pool = full news catalog", robust.get("published = full news catalog")),
        ("Headlines only", robust.get("titles only")),
        ("Without the 10 most-shown people", robust.get("without the 10 most-shown people")),
        ("Story level, any woman", robust.get("article level (features any woman)")),
        ("First six days only", robust.get("train logs only")),
        ("Last day only", robust.get("dev logs only")),
    ]
    rows = [(name, row) for name, row in rows if row]
    columns = [
        ("Pool", 2.1),
        ("Shown", 2.6),
        ("Clicked", 3.1),
        ("Platform", 3.68),
        ("Reader", 4.22),
    ]
    table_y = 2.76
    for column_name, column_x in columns:
        draw_text(
            fig,
            right_x + column_x,
            table_y,
            column_name,
            size=6.6,
            weight="bold",
            color=MUTED,
            ha="right",
        )
    table_y += 0.2
    draw_hline(fig, right_x, right_x + right_width, table_y - 0.03)
    min_platform = 9e9
    for name, row in rows:
        pool_share, shown_share, clicked_share = row["published"], row["shown"], row["clicked"]
        platform, reader = shown_share / pool_share, clicked_share / shown_share
        min_platform = min(min_platform, platform)
        draw_text(
            fig,
            right_x,
            table_y,
            name,
            size=6.9,
            weight="bold" if name == "Main analysis" else "normal",
        )
        for (_, column_x), value in zip(columns[:3], (pool_share, shown_share, clicked_share)):
            draw_text(
                fig,
                right_x + column_x,
                table_y,
                percent(value, 1),
                size=6.9,
                color=INK_SOFT,
                ha="right",
            )
        draw_text(
            fig,
            right_x + columns[3][1],
            table_y,
            f"×{platform:.2f}",
            size=6.9,
            weight="bold",
            ha="right",
        )
        draw_text(
            fig,
            right_x + columns[4][1],
            table_y,
            f"×{reader:.2f}",
            size=6.9,
            color=INK_SOFT,
            ha="right",
        )
        table_y += 0.175
    draw_hline(fig, right_x, right_x + right_width, table_y)
    table_y = draw_paragraph(
        fig,
        right_x,
        table_y + 0.1,
        (
            "Every check keeps the platform step above "
            f"×{math.floor(min_platform * 10) / 10:.1f}. All entity links have confidence of at "
            "least 0.9, so a confidence filter changes nothing."
        ),
        right_width,
        size=6.8,
        color=INK_SOFT,
        line_height=1.25,
    )
    small_groups = [
        row["group"]
        for row in results["scorecard"]
        if row["group"] != "Other" and row["published"] and row["published"] < SMALL_GROUP_SHARE
    ]
    table_y += 0.1
    draw_text(fig, right_x, table_y, "Is the data sufficient?", size=8.6, weight="bold")
    sufficiency = (
        "For the headline, yes: "
        f"{thousands(results['funnel_gender']['gendered_appearances']['shown'])} person "
        f"appearances shown, and all {len(rows) - 1} alternative checks agree on direction. "
        f"Coverage is the limit: {percent(article_coverage['share_with_person'])} of stories and "
        f"{percent(article_coverage['share_displays_with_gendered_person'])} of displays name a "
        f"linked person ({thousands(results['concentration']['people_shown'])} people in stories "
        f"shown, {thousands(coverage['humans'])} in the catalog, "
        f"{thousands(coverage['humans_women'])} of them women)."
    )
    if small_groups:
        sufficiency += (
            f" For groups under 1% of the pool ({' and '.join(small_groups)}), "
            "read the ratios as indicative."
        )
    table_y = draw_paragraph(
        fig,
        right_x,
        table_y + 0.21,
        sufficiency,
        right_width,
        size=6.8,
        color=INK_SOFT,
        line_height=1.25,
    )

    table_y += 0.12
    draw_text(fig, right_x, table_y, "Team and roles", size=8.6, weight="bold")
    table_y += 0.22
    for member_name, program, role in TEAM["members"]:
        draw_text(fig, right_x, table_y, member_name, size=7.3, weight="bold")
        table_y = (
            draw_paragraph(
                fig,
                right_x + 1.3,
                table_y,
                f"{program}. {role}",
                right_width - 1.3,
                size=6.8,
                color=INK_SOFT,
                line_height=1.22,
            )
            + 0.04
        )
    if TEAM.get("collaboration"):
        table_y = draw_paragraph(
            fig,
            right_x,
            table_y + 0.02,
            TEAM["collaboration"],
            right_width,
            size=6.8,
            color=INK_SOFT,
            line_height=1.25,
        )

    table_y += 0.08
    draw_text(fig, right_x, table_y, "Sources", size=8.6, weight="bold")
    sources_text = (
        "Wu et al. 2020, MIND, ACL. Vrijenhoek et al. 2024, RADio*, ACM TORS. Vrijenhoek 2023, Do "
        "you MIND?, BIAS. FairRec 2021, arXiv:2006.16742. EB-NeRD 2024, arXiv:2410.03432. Asr, "
        "Taboada et al. 2021, Gender Gap Tracker, PLOS ONE. CEM 2026, Digital News Report Canada. "
        "Mather Economics, Globe and Mail and Sophi. Kronk 2026, arXiv:2609.22375. "
        "Diversity-aware news recommender datasets 2025, arXiv:2510.05952. Duncan and Duncan "
        "1955, American Sociological Review. EU Digital Services Act, Regulation (EU) 2022/2065, "
        "Art. 27 and 38. UN World Population Prospects 2019."
    )
    draw_paragraph(
        fig,
        right_x,
        table_y + 0.21,
        sources_text,
        right_width,
        size=6.2,
        color=MUTED,
        line_height=1.25,
    )
    pdf.savefig(fig)
    plt.close(fig)


def main():
    """Parse arguments and build PDF."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results", required=True, help="results.json from funnel_vision.py")
    parser.add_argument(
        "--team", help="team.json with name, members [[name, program, role]], code_url"
    )
    parser.add_argument("--out", default="FunnelVision_Phase1.pdf", help="output PDF path")
    parser.add_argument(
        "--preview", action="store_true", help="stamp every page as a layout preview"
    )
    args = parser.parse_args()
    if args.team:
        with open(args.team, encoding="utf-8") as handle:
            TEAM.update(json.load(handle))
    setup_fonts()
    with open(args.results, encoding="utf-8") as handle:
        results = json.load(handle)
    member_names = ", ".join(member[0] for member in TEAM.get("members", []))
    with PdfPages(
        args.out,
        metadata={
            "Title": "Funnel Vision: who gets seen in your news feed",
            "Author": f"{TEAM['name']} ({member_names})" if member_names else TEAM["name"],
            "Subject": "Behind The Feed Data Challenge, Phase 1: data-driven visual summary",
            "Keywords": "news recommendation, information exposure, fairness, diversity, MIND, "
            "Wikidata",
        },
    ) as pdf:
        for build_page in (page1, page2, page3, page4, page5):
            build_page(pdf, results, args.preview)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
