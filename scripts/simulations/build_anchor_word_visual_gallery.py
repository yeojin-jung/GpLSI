#!/usr/bin/env python3
"""Build compact contact sheets and an HTML index for all pilot figures."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _display_name(path: Path) -> str:
    name = path.stem
    replacements = {
        "tran_mixed_word_decay_graph_adapted_": "Graph-adapted · ",
        "tran_mixed_word_decay_exact_": "Exact Tran · ",
        "paired_current_vs_poisson_A_recovery": "Paired current vs Poisson A",
        "_VH_within_P": " · VH within P0–P3",
        "anchor_word": "anchor word",
        "poisson_full": "Poisson-full",
        "heldout": "held-out",
        "_": " ",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return " ".join(name.split()).strip().replace(" · ", " · ")


def _contact_sheet(
    paths: list[Path],
    target: Path,
    title: str,
    *,
    columns: int = 2,
    tile_width: int = 1280,
    tile_height: int = 930,
) -> None:
    title_height = 100
    label_height = 54
    rows = (len(paths) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * tile_width, title_height + rows * (tile_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(42)
    label_font = _font(26)
    draw.text((36, 26), title, fill="#111111", font=title_font)
    for index, path in enumerate(paths):
        row, column = divmod(index, columns)
        x = column * tile_width
        y = title_height + row * (tile_height + label_height)
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((tile_width - 30, tile_height - 20), Image.Resampling.LANCZOS)
            offset_x = x + (tile_width - image.width) // 2
            offset_y = y + (tile_height - image.height) // 2
            canvas.paste(image, (offset_x, offset_y))
        label = _display_name(path)
        draw.text((x + 24, y + tile_height + 8), label, fill="#222222", font=label_font)
    canvas.save(target, optimize=True)


def _write_html(
    figures: Path,
    output: Path,
    categories: list[tuple[str, list[Path]]],
    overviews: list[Path],
) -> None:
    sections: list[str] = []
    sections.append(
        '<section><h2>Overview sheets</h2><div class="grid">'
        + "".join(
            f'<figure><a href="{html.escape(path.name)}"><img src="{html.escape(path.name)}"></a>'
            f'<figcaption>{html.escape(_display_name(path))}</figcaption></figure>'
            for path in overviews
        )
        + "</div></section>"
    )
    for heading, paths in categories:
        cards = "".join(
            f'<figure><a href="{html.escape(path.name)}"><img loading="lazy" '
            f'src="{html.escape(path.name)}"></a><figcaption>'
            f'{html.escape(_display_name(path))} · '
            f'<a href="{html.escape(path.with_suffix(".pdf").name)}">PDF</a>'
            f'</figcaption></figure>'
            for path in paths
        )
        sections.append(
            f'<section><h2>{html.escape(heading)}</h2><div class="grid">{cards}</div></section>'
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Anchor-word GpLSI pilot — visual inspection gallery</title>
<style>
body {{ margin: 0 auto; max-width: 1600px; padding: 32px; color: #202124;
       background: #f7f8fa; font: 16px Arial, sans-serif; }}
h1 {{ margin-bottom: 8px; }}
p {{ color: #5f6368; }}
section {{ margin-top: 36px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 22px; }}
figure {{ margin: 0; padding: 12px; background: white; border: 1px solid #dadce0;
          border-radius: 10px; box-shadow: 0 1px 3px #00000018; }}
img {{ display: block; width: 100%; height: auto; }}
figcaption {{ padding: 10px 4px 2px; font-weight: 600; }}
a {{ color: #1769aa; }}
</style>
</head>
<body>
<h1>Anchor-word GpLSI pilot — visual inspection gallery</h1>
<p>All figures use the 12-seed pilot. Click any image for the full-resolution PNG;
PDF links preserve vector output.</p>
{''.join(sections)}
</body>
</html>
"""
    output.write_text(document)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("figure_directory", type=Path)
    args = parser.parse_args()
    figures = args.figure_directory.resolve()
    source_pngs = sorted(
        path
        for path in figures.glob("*.png")
        if not path.name.startswith("overview_")
    )
    exact_vh = [path for path in source_pngs if "_exact_" in path.name and "VH_within_P" in path.name]
    graph_vh = [path for path in source_pngs if "_graph_adapted_" in path.name and "VH_within_P" in path.name]
    exact_benchmarks = [
        path for path in source_pngs if "_exact_" in path.name and "VH_within_P" not in path.name
    ]
    graph_benchmarks = [
        path
        for path in source_pngs
        if "_graph_adapted_" in path.name and "VH_within_P" not in path.name
    ]
    paired = [path for path in source_pngs if path.name.startswith("paired_")]
    categories = [
        ("Exact Tran: cross-method and diagnostic figures", exact_benchmarks),
        ("Exact Tran: all VH algorithms within P0–P3", exact_vh),
        ("Graph-adapted: cross-method and diagnostic figures", graph_benchmarks),
        ("Graph-adapted: all VH algorithms within P0–P3", graph_vh),
        ("Paired A recovery", paired),
    ]
    expected = sum(len(paths) for _, paths in categories)
    if expected != len(source_pngs):
        classified = {path for _, paths in categories for path in paths}
        missing = sorted(path.name for path in source_pngs if path not in classified)
        raise RuntimeError(f"gallery classification missed: {missing}")

    overview_specs = [
        ("overview_exact_benchmarks.png", "Exact Tran — benchmarks and diagnostics", exact_benchmarks),
        ("overview_exact_VH_within_P.png", "Exact Tran — all VH algorithms within P0–P3", exact_vh),
        (
            "overview_graph_adapted_benchmarks.png",
            "Graph-adapted — benchmarks and diagnostics",
            graph_benchmarks,
        ),
        (
            "overview_graph_adapted_VH_within_P.png",
            "Graph-adapted — all VH algorithms within P0–P3",
            graph_vh,
        ),
        ("overview_paired_A_recovery.png", "Paired current versus Poisson A recovery", paired),
    ]
    overview_paths: list[Path] = []
    for filename, title, paths in overview_specs:
        target = figures / filename
        _contact_sheet(paths, target, title)
        overview_paths.append(target)
    _write_html(figures, figures / "visual_inspection_gallery.html", categories, overview_paths)
    print(f"gallery: {figures / 'visual_inspection_gallery.html'}")
    print(f"full-resolution plots: {len(source_pngs)}")
    print(f"overview sheets: {len(overview_paths)}")


if __name__ == "__main__":
    main()
