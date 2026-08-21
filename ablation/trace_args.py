"""Replay parse_model's channel and arg arithmetic for the as-deployed MLT config.

No torch required: reimplements only make_divisible and the arg-rewrite branch that
MultiScaleCBAM takes, to show what reduction ratio actually reaches the constructor.
"""

import math

WIDTH = 0.50
MAX_CHANNELS = 1024


def make_divisible(x: float, divisor: int) -> int:
    return math.ceil(x / divisor) * divisor


def scale(c2: int) -> int:
    return make_divisible(min(c2, MAX_CHANNELS) * WIDTH, 8)


# (layer, module, from, yaml_args) for the notebook's custom_yolo11s-obb.yaml
LAYERS = [
    (0, "Conv", -1, [64, 3, 2]),
    (1, "Conv", -1, [128, 3, 2]),
    (2, "C3k2", -1, [256, False, 0.25]),
    (3, "Conv", -1, [256, 3, 2]),
    (4, "C3k2", -1, [512, False, 0.25]),
    (5, "Conv", -1, [512, 3, 2]),
    (6, "C3k2", -1, [512, True]),
    (7, "Conv", -1, [1024, 3, 2]),
    (8, "C3k2", -1, [1024, True]),
    (9, "SPPF", -1, [1024, 5]),
    (10, "Conv", -1, [1024, 1, 1]),
    (11, "CrossAttentionBlock", [5, 10], [1024, 512]),
    (12, "MultiScaleCBAM", -1, [1024, 16]),
    (13, "Upsample", -1, []),
    (14, "Concat", [-1, 6], []),
    (15, "C3k2", -1, [512, False]),
    (16, "Upsample", -1, []),
    (17, "Concat", [-1, 4], []),
    (18, "C3k2", -1, [256, False]),
    (19, "MultiScaleCBAM", -1, [256, 16]),
    (20, "Conv", -1, [256, 3, 2]),
    (21, "Concat", [-1, 15, 6], []),
    (22, "C3k2", -1, [512, False]),
    (23, "MultiScaleCBAM", -1, [512, 16]),
    (24, "Conv", -1, [512, 3, 2]),
    (25, "Concat", [-1, 12, 10], []),
    (26, "C3k2", -1, [1024, True]),
    (27, "MultiScaleCBAM", -1, [1024, 16]),
]

ch: list[int] = [3]
findings: list[str] = []

for idx, module, frm, args in LAYERS:
    prev = ch[-1]

    if module == "Concat":
        c2 = sum(ch[f if f >= 0 else len(ch) + f] for f in frm)
    elif module == "Upsample":
        c2 = prev
    elif module == "CrossAttentionBlock":
        q, kv = scale(args[0]), scale(args[1])
        c2 = q
        print(f"L{idx:<3} {module:<20} Q={q:<5} K/V={kv:<5} (kv projected {kv}->{q})")
        ch.append(c2)
        continue
    elif module == "MultiScaleCBAM":
        c1 = prev
        c2 = scale(args[0])
        # base_modules branch: args = [c1, c2, *args[1:]]
        new_args = [c1, c2, *args[1:]]
        # repeat_modules branch: args.insert(2, n) with n == 1
        new_args.insert(2, 1)
        # MultiScaleCBAM.__init__(self, c1, r=16, *args, **kwargs)
        bound_c1, bound_r = new_args[0], new_args[1]
        bottleneck = bound_c1 // bound_r
        findings.append(
            f"L{idx}: c1={bound_c1} r={bound_r} -> bottleneck={bottleneck} channel(s)"
        )
        print(
            f"L{idx:<3} {module:<20} yaml_args={args} -> ctor{tuple(new_args)}  "
            f"r={bound_r} (NOT {args[1]})  bottleneck={bottleneck}ch"
        )
        ch.append(c2)
        continue
    else:
        c2 = scale(args[0])

    ch.append(c2)

print("\n--- SimpleChannelAttention bottleneck widths as constructed ---")
for f in findings:
    print(" ", f)
print(
    "\nEvery instance collapses the channel descriptor to a single scalar before\n"
    "re-expanding, so the gate vector is rank-1 and cannot weight channels selectively."
)
