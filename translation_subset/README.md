# Supplementary Translation Subset

This folder is for a small held-out evaluation subset used to assess the
end-to-end OCR plus translation pipeline.

## Purpose

`MLT-2019` and `ReCTS` support benchmark evaluation for detection and
recognition, but they do not provide reference translations. This subset is a
manually translated, supplementary evaluation split for reporting end-to-end
pipeline behavior.

## Recommended protocol

1. Use only validation or test images. Do not sample from training data.
2. Generate ranked candidates with `tools/curate_translation_subset.py`.
3. Prefer clean samples with one or two clear text instances and limited clutter.
4. Keep the target language fixed for the study, ideally English.
5. Manually verify the source transcription before writing the reference translation.
6. Use two annotators, or one annotator plus one verifier.
7. Store the final annotations in `annotation_template.csv`.

## Example command

```bash
python3 tools/curate_translation_subset.py \
  --images /path/to/mlt/images/val \
  --labels /path/to/mlt/labels/val \
  --class-names Arabic Latin Chinese Korean Japanese Bangla Hindi Other \
  --per-class 20 \
  --output-dir translation_subset/candidates/mlt_val
```

For `ReCTS`, use the held-out split and pass a single class name:

```bash
python3 tools/curate_translation_subset.py \
  --images /path/to/rects/images/val \
  --labels /path/to/rects/labels/val \
  --class-names text \
  --per-class 20 \
  --output-dir translation_subset/candidates/rects_val
```

## Suggested reporting language

Use wording like this in the paper:

`Because ICDAR MLT-2019 and ReCTS do not provide reference translations, we constructed a small manually translated held-out subset from the benchmark validation/test splits for supplementary end-to-end pipeline evaluation. Candidate images were ranked automatically to favor low-clutter scenes with large, legible text regions, and final samples were manually verified before annotation.`
