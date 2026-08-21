"""Emit a publication-figure architecture diagram as draw.io XML.

    python ablation/make_drawio.py --config configs/full_legacy.yaml

The 29 layers of the config are grouped into 11 nodes laid out as a feature pyramid:
one row per level (P3, P4, P5), flowing left to right through backbone, top-down path,
bottom-up path and head. Every edge is derived from the config's `from` lists rather
than drawn by hand, and GROUPS is checked against the config at build time, so the
figure cannot claim a connection the network does not have.

A per-layer version of the same diagram is unreadable at IEEE column width: 29 boxes
across a double-column figure leaves each about 6 mm wide.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).parent))
from make_figures import trace  # noqa: E402

# Which config layers each figure node stands for.
GROUPS: dict[str, list[int]] = {
    "stem": [0, 1, 2],
    "bp3": [3, 4],
    "bp4": [5, 6],
    "bp5": [7, 8, 9, 10],
    "xat": [11],
    "cb5": [12],
    "td4": [13, 14, 15],
    "td3": [16, 17, 18, 19],
    "bu4": [20, 21, 22, 23],
    "bu5": [24, 25, 26, 27],
    "head": [28],
}

ROW = {"P3": 110, "P4": 290, "P5": 470}
BOX_W, BOX_H = 162, 112

NODES: dict[str, tuple[int, int, str, str]] = {
    "input": (30, ROW["P3"], "Input\n480 x 480", "input"),
    "stem": (220, ROW["P3"], "Stem\nP2/4, c=128", "conv"),
    "bp3": (410, ROW["P3"], "Backbone P3/8\nc=256, 60x60", "conv"),
    "bp4": (410, ROW["P4"], "Backbone P4/16\nc=256, 30x30", "conv"),
    "bp5": (410, ROW["P5"], "Backbone P5/32\nSPPF, Conv 1x1\nc=512, 15x15", "conv"),
    "xat": (600, ROW["P5"], "Cross-Attention\nQ: P5, K/V: P4\n8 heads, c=512", "attn"),
    "cb5": (790, ROW["P5"], "MS-CBAM\nc=512, 15x15", "cbam"),
    "td4": (790, ROW["P4"], "Top-down P4\nUpsample + Concat\nc=256, 30x30", "neck"),
    "td3": (980, ROW["P3"], "Top-down P3\nConcat, MS-CBAM\nc=128, 60x60", "neck"),
    "bu4": (1170, ROW["P4"], "Bottom-up P4\n3-way Concat\nMS-CBAM, c=256", "neck"),
    "bu5": (1360, ROW["P5"], "Bottom-up P5\n3-way Concat\nMS-CBAM, c=512", "neck"),
    "head": (1550, ROW["P4"], "OBB Head\nP3 / P4 / P5\n8 classes", "head"),
}

STYLE = {
    "input": "fillColor=#FFFFFF;strokeColor=#666666;",
    "conv": "fillColor=#FCE4C8;strokeColor=#B07A3F;",
    "attn": "fillColor=#E8503A;strokeColor=#96301F;fontColor=#FFFFFF;",
    "cbam": "fillColor=#F2A9D2;strokeColor=#B05B8C;",
    "neck": "fillColor=#BDD7EE;strokeColor=#4A7EAA;",
    "head": "fillColor=#4472C4;strokeColor=#2A4A85;fontColor=#FFFFFF;",
}

# Edges whose straight run would cross an intervening box, routed through a clear band.
LANES: dict[tuple[str, str], int] = {("bp4", "bu4"): 248, ("bp5", "bu5"): 432}
EDGE_LABEL = {("bp5", "xat"): "Q", ("bp4", "xat"): "K/V"}


def derive_edges(config: Path) -> list[tuple[str, str, bool]]:
    """Inter-group edges implied by the config, flagged as skip or sequential."""
    layers = {l.index: l for l in trace(config)}
    missing = set(layers) - {i for v in GROUPS.values() for i in v}
    if missing:
        raise SystemExit(f"GROUPS does not cover layers {sorted(missing)}")

    owner = {i: g for g, idx in GROUPS.items() for i in idx}
    order = list(NODES)
    seen: list[tuple[str, str, bool]] = []
    for layer in layers.values():
        for src in layer.sources:
            if src < 0 or owner[src] == owner[layer.index]:
                continue
            a, b = owner[src], owner[layer.index]
            skip = abs(order.index(b) - order.index(a)) > 1
            if (a, b, skip) not in seen:
                seen.append((a, b, skip))
    return [("input", "stem", False)] + seen


def to_xml(edges: list[tuple[str, str, bool]]) -> str:
    cells: list[str] = []
    for name, (x, y, label, kind) in NODES.items():
        style = (
            "rounded=1;whiteSpace=wrap;html=1;fontSize=11;fontFamily=Helvetica;"
            "arcSize=12;verticalAlign=middle;" + STYLE[kind]
        )
        cells.append(
            f'<mxCell id="{name}" value="{escape(label)}" style="{style}" vertex="1" '
            f'parent="1"><mxGeometry x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" '
            f'as="geometry"/></mxCell>'
        )

    for n, (a, b, skip) in enumerate(edges):
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;jettySize=auto;"
            "endArrow=block;endFill=1;endSize=6;fontSize=10;fontFamily=Helvetica;"
            + ("dashed=1;strokeColor=#7F9BB5;strokeWidth=1.4;"
               if skip else "strokeColor=#31506B;strokeWidth=1.8;")
        )
        label = escape(EDGE_LABEL.get((a, b), ""))
        points = ""
        if (a, b) in LANES:
            lane = LANES[(a, b)]
            ax = NODES[a][0] + BOX_W // 2
            bx = NODES[b][0] + BOX_W // 2
            points = (
                f'<Array as="points"><mxPoint x="{ax}" y="{lane}"/>'
                f'<mxPoint x="{bx}" y="{lane}"/></Array>'
            )
        cells.append(
            f'<mxCell id="e{n}" value="{label}" style="{style}" edge="1" parent="1" '
            f'source="{a}" target="{b}"><mxGeometry relative="1" as="geometry">'
            f'{points}</mxGeometry></mxCell>'
        )

    legend = (
        "Solid arrows: sequential dataflow.  Dashed arrows: skip connections feeding "
        "the BiFPN concatenations.  Channel counts are for the s scale (width 0.50); "
        "grids for a 480x480 input."
    )
    cells.append(
        f'<mxCell id="legend" value="{escape(legend)}" '
        f'style="text;html=1;align=left;verticalAlign=top;fontSize=10;'
        f'fontFamily=Helvetica;fontColor=#595959;whiteSpace=wrap;" vertex="1" '
        f'parent="1"><mxGeometry x="30" y="620" width="1680" height="40" '
        f'as="geometry"/></mxCell>'
    )

    body = "".join(cells)
    return (
        '<mxGraphModel dx="1600" dy="800" grid="0" gridSize="10" guides="1" '
        'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
        'pageWidth="1760" pageHeight="700" math="0" shadow="0" adaptiveColors="auto">'
        f"<root><mxCell id=\"0\"/><mxCell id=\"1\" parent=\"0\"/>{body}</root>"
        "</mxGraphModel>"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "configs/full_legacy.yaml")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "architecture.drawio")
    opts = parser.parse_args()

    edges = derive_edges(opts.config)
    opts.out.write_text(to_xml(edges), encoding="utf-8")

    print(f"wrote {opts.out}")
    print(f"  {len(NODES)} nodes, {len(edges)} edges, "
          f"{sum(1 for *_, s in edges if s)} of them skips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
