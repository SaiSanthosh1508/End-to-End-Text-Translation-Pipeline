# English-Only End-to-End Translation Evaluation

This project now uses a single fixed target language for supplementary
translation evaluation: `English`.

## Why English Only

- It keeps annotation cost manageable.
- It makes manual verification realistic.
- It avoids weak claims about many-to-many translation quality.
- It matches the actual contribution: an end-to-end scene-text understanding
  pipeline, not a new machine translation model.

## Final Evaluation Set

- `ICDAR MLT-2019`
  - Source: held-out subset sampled from the training split with official source
    text ground truth.
  - Count: `140` images
  - Composition: `20` images each for `Arabic`, `Latin`, `Chinese`, `Korean`,
    `Japanese`, `Bangla`, and `Hindi`
- `ReCTS`
  - Source: held-out subset sampled from the `val` split with source text
    recovered from official JSON annotations.
  - Count: `20` images

- Combined total: `160` images

## Files

- Final gold annotation sheet:
  - [english_only_master_annotation_gold.csv](/Users/saisanthosh/Documents/paper-exp/End-to-End-Text-Translation-Pipeline/translation_subset/english_only_master_annotation_gold.csv)
- Saved HF Space evaluation:
  - [hf_space_eval_full.csv](/Users/saisanthosh/Documents/paper-exp/End-to-End-Text-Translation-Pipeline/translation_subset/hf_space_eval_full.csv)
- Saved normalized-text chrF scores:
  - [hf_space_eval_full_chrf_normalized.csv](/Users/saisanthosh/Documents/paper-exp/End-to-End-Text-Translation-Pipeline/translation_subset/hf_space_eval_full_chrf_normalized.csv)

## Current Status

- Image subset selection
- Source-side text references
- English target-language references for all `160` images in the gold sheet
- Saved Hugging Face Space evaluation outputs for the same set

The current saved evaluation run completed `160/160` rows successfully.

## Current HF Space Snapshot

- Overall completed rows: `160/160`
- Exact normalized matches: `31`
- Partial contains matches: `41`
- Average normalized chrF across all `160` rows: `0.3647`
- Average normalized chrF across the `31` exact-match rows: `0.8118`
- Final proportional top-40 subset average normalized chrF: `0.8108`

## Paper-Safe Methodology Text

`Because ICDAR MLT-2019 and ReCTS do not provide benchmark reference translations, we constructed a supplementary English-only evaluation subset for end-to-end translation assessment. The subset comprises 140 MLT-2019 images drawn from a held-out portion of the training split and 20 ReCTS images drawn from the validation split. Candidate images were automatically ranked to favor low-clutter scenes with large, legible text regions, then manually screened before annotation. Source-side text references were recovered from the official benchmark annotations, and English target translations were manually created for evaluation.`

## Paper-Safe Results Framing

`Quantitative translation evaluation is reported only for English as the target language. Multi-target translation remains a qualitative deployment feature of the web application and is not treated as a benchmarked many-to-many translation task.`
