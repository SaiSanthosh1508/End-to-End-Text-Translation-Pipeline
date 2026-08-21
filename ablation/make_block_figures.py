"""Emit the three module figures as draw.io diagrams, drawn from the module code.

    python ablation/make_block_figures.py

Replaces Figs. 4, 5 and 7, each of which depicts something the code does not do:

Fig. 4, channel attention
    Omits the ReLU between the squeeze and expand convolutions, and labels the result
    "Channel Weighted Feature Map (B,C,1,1)" when (B,C,1,1) is the weight vector and
    the weighted map is (B,C,W,H).

Fig. 5, multi-scale spatial attention
    Labels the concatenation of the mean and max maps (B,1,W,H) when concatenating two
    single-channel maps gives (B,2,W,H), and joins the three convolution branches with
    "Concat" when MultiScaleSpatialAttention.forward sums them.

Fig. 7, cross-attention
    Names the query source C2PSA, which the trained config does not contain, and the
    key/value source an MS-CBAM-refined map at (B,256,15,15) when it is a plain P4/16
    convolution output at (B,256,30,30).

Shapes follow the s scale used in every experiment. The style matches
architecture.drawio so the figures read as one set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.sax.saxutils import escape

QUOTES = {'"': "&quot;", "'": "&apos;"}

FILL = {
    "tensor": ("#F4F6F8", "#8FA3B5", "#1A2733"),
    "pool":   ("#E1EDF8", "#3D6B99", "#12202D"),
    "conv":   ("#BFD8EE", "#2C5F8F", "#0E1C27"),
    "act":    ("#D2E4C6", "#5C8749", "#182513"),
    "merge":  ("#F5E6CC", "#B08B4A", "#2E2410"),
    "attn":   ("#E0CBEE", "#6C4A94", "#22142E"),
    "cbam":   ("#F8D79A", "#C0891C", "#332508"),
    "out":    ("#F3CFC8", "#B0503F", "#3A1610"),
}

BOX_H = 62
OPERATOR_D = 46


class Figure:
    def __init__(self, name: str, title: str, width: int, height: int) -> None:
        self.name, self.title = name, title
        self.width, self.height = width, height
        self.nodes: dict[str, tuple[int, int, int, str, str, bool]] = {}
        self.edges: list[tuple[str, str, list[tuple[int, int]], str, str, bool]] = []

    def box(self, ident: str, x: int, y: int, w: int, caption: str, kind: str) -> None:
        self.nodes[ident] = (x, y, w, caption, kind, False)

    def operator(self, ident: str, x: int, y: int, symbol: str, kind: str = "merge") -> None:
        self.nodes[ident] = (x, y, OPERATOR_D, symbol, kind, True)

    def link(self, a: str, b: str, points: list[tuple[int, int]] | None = None,
             label: str = "", sides: str = "", dashed: bool = False) -> None:
        self.edges.append((a, b, points or [], label, sides, dashed))

    def geometry(self, ident: str) -> tuple[int, int, int, int]:
        x, y, w, _, _, round_ = self.nodes[ident]
        h = OPERATOR_D if round_ else BOX_H
        return x, y, w, h


EXIT = {"l": (0.0, 0.5), "r": (1.0, 0.5), "t": (0.5, 0.0), "b": (0.5, 1.0)}


def channel_attention() -> Figure:
    f = Figure("channel-attention", "Channel attention", 1330, 330)
    f.box("in",   25,  110, 148, "Input feature map\n(B, C, W, H)", "tensor")
    f.box("pool", 213, 110, 148, "Adaptive avg pool\n(B, C, 1, 1)", "pool")
    f.box("sq",   401, 110, 158, "Conv 1x1, C to C/r\n(B, C/r, 1, 1)", "conv")
    f.box("relu", 599, 110, 96,  "ReLU", "act")
    f.box("ex",   735, 110, 158, "Conv 1x1, C/r to C\n(B, C, 1, 1)", "conv")
    f.box("sig",  933, 110, 130, "Sigmoid\n(B, C, 1, 1)", "act")
    f.operator("mul", 1103, 118, "x")
    f.box("out",  1189, 110, 116, "Output\n(B, C, W, H)", "out")

    for a, b in (("in", "pool"), ("pool", "sq"), ("sq", "relu"), ("relu", "ex"),
                 ("ex", "sig"), ("sig", "mul"), ("mul", "out")):
        f.link(a, b)
    f.link("in", "mul", [(99, 262), (1126, 262)], "identity", "b,b", dashed=True)
    return f


def spatial_attention() -> Figure:
    f = Figure("ms-spatial-attention", "Multi-scale spatial attention", 1360, 470)
    f.box("in",  25,  200, 148, "Input feature map\n(B, C, W, H)", "tensor")
    f.box("avg", 213, 120, 158, "Mean over channels\n(B, 1, W, H)", "pool")
    f.box("max", 213, 280, 158, "Max over channels\n(B, 1, W, H)", "pool")
    f.box("cat", 409, 200, 148, "Concat\n(B, 2, W, H)", "merge")
    f.box("c3",  595, 40,  158, "Conv 3x3, 2 to 1\n(B, 1, W, H)", "conv")
    f.box("c5",  595, 200, 158, "Conv 5x5, 2 to 1\n(B, 1, W, H)", "conv")
    f.box("c7",  595, 360, 158, "Conv 7x7, 2 to 1\n(B, 1, W, H)", "conv")
    f.operator("sum", 791, 208, "+")
    f.box("sig", 877, 200, 130, "Sigmoid\n(B, 1, W, H)", "act")
    f.operator("mul", 1047, 208, "x")
    f.box("out", 1133, 200, 148, "Output\n(B, C, W, H)", "out")

    f.link("in", "avg"); f.link("in", "max")
    f.link("avg", "cat"); f.link("max", "cat")
    f.link("cat", "c3", [(576, 231), (576, 71)], sides="r,l")
    f.link("cat", "c5", sides="r,l")
    f.link("cat", "c7", [(576, 231), (576, 391)], sides="r,l")
    f.link("c3", "sum", [(772, 71), (772, 231)], sides="r,l")
    f.link("c5", "sum", sides="r,l")
    f.link("c7", "sum", [(772, 391), (772, 231)], sides="r,l")
    f.link("sum", "sig"); f.link("sig", "mul"); f.link("mul", "out")
    f.link("in", "mul", [(99, 440), (1070, 440)], "identity", "b,b", dashed=True)
    return f


def cross_attention() -> Figure:
    f = Figure("cross-attention", "Cross-attention block", 1380, 400)
    # the residual lane runs at y=62, so the title sits clear of it
    f.box("p5",  25,  90,  168, "P5/32 backbone output\n(B, 512, 15, 15)", "tensor")
    f.box("p4",  25,  250, 168, "P4/16 backbone output\n(B, 256, 30, 30)", "tensor")
    f.box("q",   233, 90,  158, "Query\n225 tokens, d=512", "attn")
    f.box("proj", 233, 250, 158, "Conv 1x1, 256 to 512", "conv")
    f.box("kv",  431, 250, 158, "Key and Value\n900 tokens, d=512", "attn")
    f.box("mha", 629, 170, 178, "Scaled dot-product\nattention, h=8", "cbam")
    f.box("op",  845, 170, 148, "Output projection", "conv")
    f.box("ln",  1031, 170, 148, "Add and LayerNorm", "act")
    f.box("out", 1217, 170, 138, "Output\n(B, 512, 15, 15)", "out")

    f.link("p5", "q"); f.link("p4", "proj"); f.link("proj", "kv")
    f.link("q", "mha", [], "Q"); f.link("kv", "mha", [], "K, V")
    f.link("mha", "op"); f.link("op", "ln"); f.link("ln", "out")
    f.link("q", "ln", [(312, 62), (1105, 62)], "residual", "t,t", dashed=True)
    return f


def to_xml(f: Figure) -> str:
    cells: list[str] = []
    cells.append(
        f'<mxCell id="title" value="{escape(f.title)}" style="text;html=1;align=left;'
        'verticalAlign=middle;fontSize=15;fontFamily=Helvetica;fontStyle=1;'
        f'fontColor=#25384B;" vertex="1" parent="1"><mxGeometry x="25" y="22" '
        f'width="600" height="26" as="geometry"/></mxCell>'
    )

    for ident, (x, y, w, caption, kind, round_) in f.nodes.items():
        fill, stroke, font = FILL[kind]
        shape = "ellipse;" if round_ else "rounded=1;arcSize=14;"
        style = (
            f"{shape}whiteSpace=wrap;html=1;fontSize=12;fontFamily=Helvetica;"
            f"fontStyle=1;verticalAlign=middle;fillColor={fill};strokeColor={stroke};"
            f"fontColor={font};strokeWidth=1.1;"
        )
        name, _, params = caption.partition("\n")
        label = f"<b>{escape(name)}</b>"
        if params:
            label += f"<br><b style='font-size:10px'>{escape(params)}</b>"
        _, _, _, h = f.geometry(ident)
        cells.append(
            f'<mxCell id="{ident}" value="{escape(label, QUOTES)}" style="{style}" '
            f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" '
            f'height="{h}" as="geometry"/></mxCell>'
        )

    for n, (a, b, points, label, sides, routed) in enumerate(f.edges):
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=1;arcSize=10;html=1;jettySize=auto;"
            "endArrow=blockThin;endFill=1;endSize=6;fontSize=10;fontFamily=Helvetica;"
            "fontColor=#25384B;labelBackgroundColor=#FFFFFF;"
            + ("strokeColor=#6E8399;strokeWidth=1.3;dashed=1;dashPattern=6 4;"
               if routed else "strokeColor=#25384B;strokeWidth=1.7;")
        )
        if sides:
            out_side, in_side = sides.split(",")
            ox, oy = EXIT[out_side]
            ix, iy = EXIT[in_side]
            style += (f"exitX={ox};exitY={oy};exitDx=0;exitDy=0;"
                      f"entryX={ix};entryY={iy};entryDx=0;entryDy=0;")
        array = ""
        if points:
            inner = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in points)
            array = f'<Array as="points">{inner}</Array>'
        cells.append(
            f'<mxCell id="e{n}" value="{escape(label)}" style="{style}" edge="1" '
            f'parent="1" source="{a}" target="{b}"><mxGeometry relative="1" '
            f'as="geometry">{array}</mxGeometry></mxCell>'
        )

    body = "".join(cells)
    return (
        f'<mxGraphModel dx="1400" dy="800" grid="0" gridSize="10" guides="1" '
        f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
        f'pageWidth="{f.width}" pageHeight="{f.height}" math="0" shadow="0" '
        f'adaptiveColors="0" background="#FFFFFF">'
        f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}</root>'
        "</mxGraphModel>"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "figures")
    opts = parser.parse_args()
    opts.out_dir.mkdir(exist_ok=True)

    for build in (channel_attention, spatial_attention, cross_attention):
        f = build()
        path = opts.out_dir / f"{f.name}.drawio"
        path.write_text(to_xml(f), encoding="utf-8")
        print(f"wrote {path.name}  ({len(f.nodes)} blocks, {len(f.edges)} links)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
