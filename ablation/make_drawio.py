"""Redraw the published Fig. 5 in its own visual language, with the architecture fixed.

    python ablation/make_drawio.py --config configs/full_legacy.yaml

Keeps what the published figure already does: a dashed backbone panel on the left with
serpentine rows, a solid neck panel on the right pairing a bottom-up column with a
top-down one, the same fill colours, the same `(c=..., ...)` captions and the same `x2`
repeat markers.

What changes is the network depicted. The published figure draws a C2PSA block between
SPPF and the Cross-Attention block, and an MS-CBAM inside the backbone; the trained
config has a plain `Conv [1024,1,1]` where C2PSA appears and no backbone MS-CBAM at
all. It labels the head `Detect` where the model uses `OBB`, and omits the three-input
concatenations that make the neck a BiFPN.

Channel values are the YAML arguments, matching the published figure's convention
rather than the scale-resolved widths.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).parent))
from make_figures import trace  # noqa: E402

BOX_W, BOX_H = 132, 46

# saxutils.escape leaves quotes alone, which would break an attribute value.
QUOTES = {'"': "&quot;", "'": "&apos;"}

# Steel blues carry the stock YOLO operators, deepening with the stage: plain
# convolutions lightest, block stacks mid, the pooling stage darkest. Resampling
# takes a warm neutral and merges a sage, so the two structural operations read
# apart from the convolutions at a glance. The two contributed modules take gold
# and plum, and the head is the single dark terminal. Fills are ordered by
# lightness as well as hue, so the figure holds up printed in greyscale.
FILL = {
    "conv": ("#DCE6F1", "#4A6E96", "#15222F"),
    "c3k2": ("#BFD3E6", "#35608C", "#101C28"),
    "sppf": ("#93B2D2", "#26537F", "#0C1622"),
    "up":   ("#EDE3D4", "#A08A66", "#2B2318"),
    "cat":  ("#CEDCC4", "#68885A", "#1B2716"),
    "cbam": ("#F2D9A0", "#AE8327", "#33260A"),
    "attn": ("#DBC8E4", "#77508F", "#261732"),
    "head": ("#2C4A6B", "#1C334A", "#FFFFFF"),
}

NODES: dict[int, tuple[int, int, str, str, str]] = {
    0:  (55,   165, "Conv\n(c=64,3x3,k=2)", "conv", ""),
    1:  (200,  165, "Conv\n(c=128,3x3,k=2)", "conv", ""),
    2:  (345,  165, "C3k2\n(c=256, scale=0.25)", "c3k2", "x2"),
    3:  (490,  165, "Conv\n(c=256,3x3,k=2)", "conv", ""),
    4:  (490,  268, "C3k2\n(c=512, scale=0.25)", "c3k2", "x2"),
    5:  (345,  268, "Conv\n(c=512,3x3,k=2)", "conv", ""),
    6:  (200,  268, "C3k2\n(c=512, shortcut)", "c3k2", "x2"),
    7:  (55,   268, "Conv\n(c=1024,3x3,k=2)", "conv", ""),
    8:  (55,   371, "C3k2\n(c=1024, shortcut)", "c3k2", "x2"),
    9:  (200,  371, "SPPF\n(c=1024, k=5)", "sppf", ""),
    10: (345,  371, "Conv\n(c=1024, 1x1)", "conv", ""),
    11: (490,  371, "Cross-Attention\n(h=8, c=1024)", "attn", ""),
    12: (635,  371, "MS-CBAM\n(c=1024, r=16)", "cbam", ""),
    13: (635,  474, "UpSample", "up", ""),
    14: (345,  474, "Concat", "cat", ""),
    15: (905,  560, "C3k2\n(c=512, shortcut)", "c3k2", "x2"),
    16: (905,  474, "UpSample", "up", ""),
    17: (905,  388, "Concat", "cat", ""),
    18: (905,  302, "C3k2\n(c=256, scale=0.25)", "c3k2", "x2"),
    19: (905,  216, "MS-CBAM\n(c=256, r=16)", "cbam", ""),
    20: (905,  130, "Conv\n(c=256,3x3,k=2)", "conv", ""),
    21: (1155, 130, "Concat", "cat", ""),
    22: (1155, 202, "C3k2\n(c=512, shortcut)", "c3k2", "x2"),
    23: (1155, 274, "MS-CBAM\n(c=512, r=16)", "cbam", ""),
    24: (1155, 346, "Conv\n(c=512,3x3,k=2)", "conv", ""),
    25: (1155, 418, "Concat", "cat", ""),
    26: (1155, 490, "C3k2\n(c=1024, shortcut)", "c3k2", "x2"),
    27: (1155, 562, "MS-CBAM\n(c=1024, r=16)", "cbam", ""),
    28: (1155, 634, "OBB Detect\n(nc=8)", "head", ""),
}

PANELS = [
    ("bb", 28, 128, 750, 420, "1", "Backbone"),
    ("nk", 868, 95, 432, 632, "0", "Neck and head"),
]

# Long-range links routed through clear lanes, as the published figure does.
# SIDES fixes which edge of a block a link leaves and enters by, so the corner the
# router inserts cannot double back across the block it just left.
SIDES: dict[tuple[int, int], tuple[str, str]] = {
    (5, 11):  ("bottom", "top"),
    (6, 14):  ("top", "left"),
    (6, 21):  ("top", "top"),
    (4, 17):  ("right", "left"),
    (14, 15): ("bottom", "left"),
    (15, 21): ("right", "top"),
    (12, 25): ("top", "left"),
    (10, 25): ("bottom", "left"),
    (19, 28): ("left", "bottom"),
    (23, 28): ("right", "bottom"),
}

WAYPOINTS: dict[tuple[int, int], list[tuple[int, int]]] = {
    (5, 11):  [(411, 344), (556, 344)],
    (6, 14):  [(266, 232), (40, 232), (40, 497)],
    (6, 21):  [(266, 248), (700, 248), (700, 68), (1221, 68)],
    (4, 17):  [(660, 291), (660, 118), (818, 118), (818, 411)],
    (14, 15): [(411, 545), (860, 545), (860, 583)],
    (15, 21): [(1063, 583), (1063, 100), (1221, 100)],
    (12, 25): [(701, 344), (838, 344), (838, 455), (1120, 455)],
    (10, 25): [(411, 462), (1108, 462)],
    (19, 28): [(880, 239), (880, 790), (1221, 790)],
    (23, 28): [(1330, 297), (1330, 760), (1221, 760)],
}

EXIT_X = {"left": 0.0, "right": 1.0, "top": 0.5, "bottom": 0.5}
EXIT_Y = {"left": 0.5, "right": 0.5, "top": 0.0, "bottom": 1.0}


def links(config: Path) -> list[tuple[int, int]]:
    """Every (source, target) pair the config declares."""
    pairs = [
        (src, layer.index)
        for layer in trace(config)
        for src in layer.sources
        if src >= 0
    ]
    missing = {i for pair in pairs for i in pair} - set(NODES)
    if missing:
        raise SystemExit(f"NODES is missing layers {sorted(missing)}")
    return pairs


def to_xml(pairs: list[tuple[int, int]]) -> str:
    cells: list[str] = []

    for name, x, y, w, h, dashed, label in PANELS:
        style = (
            f"rounded=1;arcSize=4;whiteSpace=wrap;html=1;dashed={dashed};"
            "dashPattern=8 5;fillColor=#F7F9FB;strokeColor=#93A5B5;strokeWidth=1.2;"
            "verticalAlign=top;align=left;spacingLeft=12;spacingTop=6;fontSize=11;"
            "fontFamily=Helvetica;fontStyle=1;fontColor=#5F7488;"
        )
        cells.append(
            f'<mxCell id="{name}" value="{escape(label)}" style="{style}" vertex="1" '
            f'parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
            f'as="geometry"/></mxCell>'
        )

    for idx, (x, y, caption, kind, repeat) in NODES.items():
        fill, stroke, font = FILL[kind]
        style = (
            "rounded=1;arcSize=14;whiteSpace=wrap;html=1;fontSize=10;"
            "fontFamily=Helvetica;verticalAlign=middle;"
            f"fillColor={fill};strokeColor={stroke};fontColor={font};strokeWidth=1.1;"
        )
        name, _, params = caption.partition("\n")
        label = f"<b>{escape(name)}</b>"
        if params:
            label += f"<br><font style='font-size:8px'>{escape(params)}</font>"
        cells.append(
            f'<mxCell id="n{idx}" value="{escape(label, QUOTES)}" style="{style}" '
            f'vertex="1" '
            f'parent="1"><mxGeometry x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" '
            f'as="geometry"/></mxCell>'
        )
        if repeat:
            cells.append(
                f'<mxCell id="r{idx}" value="{repeat}" style="text;html=1;align=center;'
                'fontSize=8;fontFamily=Helvetica;fontStyle=2;fontColor=#5F7488;" vertex="1" '
                f'parent="1"><mxGeometry x="{x + BOX_W - 28}" y="{y - 17}" width="28" '
                'height="16" as="geometry"/></mxCell>'
            )

    for n, (a, b) in enumerate(pairs):
        routed = (a, b) in WAYPOINTS
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=1;arcSize=10;html=1;jettySize=auto;"
            "endArrow=blockThin;endFill=1;endSize=6;"
            + ("strokeColor=#6E8399;strokeWidth=1.3;"
               if routed else "strokeColor=#25384B;strokeWidth=1.7;")
        )
        points = ""
        if routed:
            inner = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in WAYPOINTS[(a, b)])
            points = f'<Array as="points">{inner}</Array>'
        if (a, b) in SIDES:
            out_side, in_side = SIDES[(a, b)]
            style += (
                f"exitX={EXIT_X[out_side]};exitY={EXIT_Y[out_side]};exitDx=0;exitDy=0;"
                f"entryX={EXIT_X[in_side]};entryY={EXIT_Y[in_side]};entryDx=0;entryDy=0;"
            )
        cells.append(
            f'<mxCell id="e{n}" style="{style}" edge="1" parent="1" source="n{a}" '
            f'target="n{b}"><mxGeometry relative="1" as="geometry">{points}'
            "</mxGeometry></mxCell>"
        )

    body = "".join(cells)
    return (
        '<mxGraphModel dx="1500" dy="820" grid="0" gridSize="10" guides="1" '
        'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
        'pageWidth="1420" pageHeight="820" math="0" shadow="0" adaptiveColors="0" background="#FFFFFF">'
        f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}</root>'
        "</mxGraphModel>"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "configs/full_legacy.yaml")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "architecture.drawio")
    opts = parser.parse_args()

    pairs = links(opts.config)
    opts.out.write_text(to_xml(pairs), encoding="utf-8")
    print(f"wrote {opts.out}")
    print(f"  {len(NODES)} blocks, {len(pairs)} links, "
          f"{len(WAYPOINTS)} routed through the margins")
    return 0


if __name__ == "__main__":
    sys.exit(main())
