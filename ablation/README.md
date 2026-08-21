# Ablation probe: does the channel-attention bottleneck matter?

Reviewer 2 asked for Table 8 to be rerun across at least three seeds. Before spending
that budget, this probe answers a prior question: **the MS-CBAM channel attention in
every run so far was misconstructed, and it is worth knowing whether that is why the
ablation shows nothing.**

## The defect

`MultiScaleCBAM` is registered in both `base_modules` and `repeat_modules` inside
Ultralytics' `parse_model`, so its YAML args are rewritten twice:

```
yaml:            [-1, 1, MultiScaleCBAM, [1024, 16]]
base_modules:    args = [c1, c2, *args[1:]]   ->  [512, 512, 16]
repeat_modules:  args.insert(2, n)            ->  [512, 512, 1, 16]
constructor:     MultiScaleCBAM(c1=512, r=512, *(1, 16))
```

The signature is `(self, c1, r=16, *args, **kwargs)`, so `r` binds to the scaled channel
count and the literal `16` is swallowed by `*args`. `SimpleChannelAttention(512, r=512)`
then builds `Conv2d(512, 1, 1)` — a **one-channel** bottleneck. The same happens at all
four sites (layers 12, 19, 23, 27) of the trained model.

The consequence is that channel attention collapses the whole descriptor to a single
scalar and re-expands it, so the gate vector is rank-1: it can apply a global gain but
cannot weight channels selectively, which is the entire point of the module. The
`ReLU` on that scalar also pins the gate to a constant whenever it goes negative.

The multi-scale *spatial* branch is unaffected — it takes no channel arguments.

`trace_args.py` in the scratchpad reproduces the arithmetic without torch;
`verify_install.py` asserts it against real constructed models.

## The probe

Two arms, three seeds each, six runs, ~28 GPU-hours on two T4s.

| arm | config | channel bottleneck |
| --- | --- | --- |
| `legacy` | `configs/full_legacy.yaml` | 1 channel — reproduces `best.pt` exactly |
| `fixed` | `configs/full_fixed.yaml` | `c // 16` — the module as documented |

The two configs are byte-identical apart from the CBAM class name, so the comparison
isolates the bottleneck width and nothing else.

## Data

Attach `rishiksaisanthosh/dataset-test` to the notebook. Every Google Drive ID the
original notebooks used now returns 404, and that Kaggle dataset is the only surviving
copy of the converted MLT set: 9,000 train / 1,000 val, 9-column oriented labels. It
carries the tree twice under different prefixes; the notebook picks one and prints
which.

Two things worth recording in the manuscript. The split is 90/10, not the 80/20 stated
in Section III-A. And `notebooks/Final_and_correct_mlt_preprocess.ipynb` emits 5-column
axis-aligned boxes, so as published it does not regenerate this dataset — the released
converter and the data actually trained on disagree.

## Running it

```bash
python ablation/install_modules.py       # patch the installed ultralytics
python ablation/verify_install.py        # assert both arms built as intended
python ablation/run_probe.py --data dataset.yaml --project runs/probe
python ablation/aggregate.py runs/probe
```

`install_modules.py` targets a clean Ultralytics install and refuses a hand-modified
copy, including the vendored tree under `Text_Translation_Pipeline/`, which serves the
Hugging Face Space and is not a training environment. It is idempotent; `--check`
reports what would change without writing.

**Do not skip `verify_install.py`.** If the patch fails silently, the fixed arm rebuilds
the one-channel bottleneck and the probe compares a network against itself.

`run_probe.py` skips any run whose `results.csv` already exists, so an interrupted
Kaggle session resumes by re-running the same command.

## Surviving a failed run

Kaggle saves a batch version's output **only when the notebook finishes**. Left
unguarded, a crash in the second run discards the first run as well — 4.6 hours gone.
Three layers guard against that, in order of how much they buy:

1. **Failures are contained.** `run_probe.py` catches a failing run, reports it, and
   carries on. The version still completes, so every finished run is saved. Re-running
   the same command retries only what failed. The notebook deliberately omits
   `check=True` for the same reason.
2. **Snapshots are cheap.** `--snapshot DIR` mirrors `results.csv` and `args.yaml` after
   every run. That is a few kilobytes, against ~20 MB of weights, and it is all
   `aggregate.py` needs.
3. **Optional off-box copy.** `push_snapshot.py` publishes the snapshot to a Kaggle
   Dataset after each run, which is the only layer that survives a session killed by
   timeout or OOM rather than by an exception. Set `PERSIST_TO_DATASET = True` in the
   notebook and add `KAGGLE_USERNAME` / `KAGGLE_KEY` as notebook secrets. It no-ops
   quietly when the credentials are absent.

Working interactively instead of *Save & Run All*? Nothing in `/kaggle/working`
persists unless you press **Save Version** before the session ends.

## Reading the result

`aggregate.py` averages each run over its final 5 epochs rather than reading the last
row. Within the existing MLT run, mAP50 moves 0.0027 and precision 0.022 across epochs
80–100 — the same order as the entire spread of Table 8 — so the last-row convention
was measuring a checkpoint lottery. State the window in the table caption.

The decision rule is whether any metric moves by at least one pooled standard
deviation:

- **It moves** → the fix is real. Run the full six-configuration ablation on the fixed
  modules, and accept the downstream cost: `best.pt` is retrained, ReCTS is rerun, and
  Tables 6, 7 and 9 plus the abstract are updated. The architectural claim may survive.
- **It does not move** → run the six-configuration ablation on the as-deployed modules,
  report the null result the reviewer expects, and disclose the displaced reduction
  ratio as a limitation rather than leaving it for a reader to find.

Either way the defect gets disclosed. Publishing a null result for a module you know
was misconstructed is not a defensible null result.

## Regenerating Figs. 5 and 7

Both figures draw an architecture that was never trained. Fig. 5 shows a C2PSA block
between SPPF and the Cross-Attention block, and an MS-CBAM inside the backbone; the
trained config has a plain `Conv [1024,1,1]` in place of C2PSA and no backbone MS-CBAM.
Fig. 7 labels the query as the C2PSA output at `(B,1024,15,15)` and the key/value as an
MS-CBAM-refined map at `(B,256,15,15)`; they are really the P5/32 output at
`(B,512,15,15)` and a plain P4/16 map at `(B,256,30,30)`.

Two generators, for two different jobs.

`make_drawio.py` emits `architecture.drawio`, the figure meant for the manuscript. The
29 layers are grouped into 11 nodes laid out as a feature pyramid, one row per level,
flowing left to right through backbone, top-down path, bottom-up path and head. Every
edge is derived from the config's `from` lists, and the grouping is checked for full
coverage at build time, so the figure cannot claim a connection the network lacks.
Open it in draw.io and export to SVG or PDF for print.

A per-layer version is unreadable at IEEE width: 29 boxes across a double-column figure
leaves each about 6 mm. `make_figures.py` emits that version as `figures_editable.pptx`
for the repository and for checking the diagram against the YAML.

Point it at a different config to document a different variant:

```bash
python ablation/make_figures.py --config configs/full_rearranged.yaml --out rearranged.pptx
```

## Regenerating Figs. 4, 5 and 7

Each depicts something the modules do not do. Fig. 4 omits the ReLU between the squeeze
and expand convolutions and labels its output `(B,C,1,1)`, which is the weight vector
rather than the weighted map. Fig. 5 labels the concatenation of the mean and max maps
`(B,1,W,H)` when two single-channel maps concatenate to `(B,2,W,H)`, and joins the three
convolution branches with `Concat` when `MultiScaleSpatialAttention.forward` sums them.
Fig. 7 names a C2PSA query source the config does not contain and an MS-CBAM-refined
key/value map at `(B,256,15,15)` that is really a plain P4/16 output at `(B,256,30,30)`.

`make_block_figures.py` emits corrected replacements into `figures/`, styled to match
the architecture diagram. Every connector is tested against every block, so none crosses
one.

Fig. 4 still draws the bottleneck as `(B,C/r,1,1)`, which is the intended design. In the
deployed model it is `(B,1,1,1)` because of the displaced reduction ratio; leave that
until the probe reports, since a fix makes the figure correct as drawn.

## Regenerating Table 5

Table 5 disagrees with Fig. 11(b) on every class by +1.6 to +15.7 mAP50 points, all in
one direction, and its implied overall mAP50 is 73.8 against 66.3 in the figure legend
and 66.1 in `results.csv`. Precision reconciles (80.3 vs 81.0) while recall and mAP do
not, so those two columns did not come from the same evaluation as the rest of the paper.

`revalidate_kaggle.ipynb` re-runs validation from `best.pt` and prints a replacement
table plus ready-to-paste LaTeX. Inference only, about two minutes. If the overall mAP50
returns near 0.663 the checkpoint is the one behind the figure; if not, `best.pt` is a
different run and the table must be rebuilt from whichever checkpoint produced it.

It reads `best.pt` from the attached dataset rather than the clone, because the repo
keeps it in git-lfs and a plain clone yields a pointer file. The script checks for that.

## Files

| file | role |
| --- | --- |
| `custom_modules.py` | attention modules, both CBAM variants |
| `install_modules.py` | registers them in the installed Ultralytics |
| `test_install_modules.py` | checks on the site-packages source surgery |
| `verify_install.py` | asserts the constructed bottleneck widths |
| `configs/full_legacy.yaml` | as-deployed arm |
| `configs/full_fixed.yaml` | corrected arm |
| `run_probe.py` | six training runs, failure-contained |
| `push_snapshot.py` | optional off-box copy to a Kaggle Dataset |
| `aggregate.py` | mean ± SD, decision rule, LaTeX table |
| `build_notebook.py` | regenerates `probe_kaggle.ipynb` |
| `probe_kaggle.ipynb` | upload this to Kaggle |
| `trace_args.py` | reproduces the defect without torch |
| `make_drawio.py` / `architecture.drawio` | Fig. 5 architecture, editable in draw.io |
| `make_block_figures.py` / `figures/` | Figs. 4, 5 and 7 module diagrams |
| `drawio_to_pptx.py` | rebuilds any .drawio as editable PowerPoint shapes |
| `make_figures.py` / `figures_editable.pptx` | per-layer diagram, editable in PowerPoint |
| `revalidate.py` | regenerates the class-wise MLT table from `best.pt` |
| `build_revalidate.py` / `revalidate_kaggle.ipynb` | run that on Kaggle |

`StandardCBAM` in `custom_modules.py` is unused by the probe. It exists for the
six-configuration ablation that follows: it shares `SimpleChannelAttention` with
`MSCBAMFixed` and differs only in the spatial kernel, so the "Std CBAM vs MS-CBAM" row
isolates the multi-scale claim. Ultralytics' own `CBAM` cannot serve that role — it is
absent from `parse_model`'s module sets and its channel branch has no bottleneck.
