# Scene Text Translation English Eval Final Top-40 Subset

This package contains the final 40-sample multilingual evaluation subset
selected from the completed 160-image English-reference evaluation set.

Selection policy:
- Rank by normalized chrF against the saved end-to-end translation output.
- Allocate samples proportionally by language.
- Merge ReCTS entries into the Chinese language bucket.

Selected rows: 40
Average normalized chrF: 0.8108

Language distribution:
- Arabic: 5 samples, avg normalized chrF 0.9028
- Bangla: 5 samples, avg normalized chrF 0.5337
- Chinese: 10 samples, avg normalized chrF 0.8830
- Hindi: 5 samples, avg normalized chrF 0.9876
- Japanese: 5 samples, avg normalized chrF 0.6483
- Korean: 5 samples, avg normalized chrF 0.9024
- Latin: 5 samples, avg normalized chrF 0.7457

Files:
- `english_only_master_annotation_gold.csv`: final 40-row gold annotation sheet.
- `final_dataset_scores.csv`: gold rows with normalized chrF, prediction, and selection ranks.
- `selection_summary.json`: machine-readable selection summary.
- `images/`: the 40 selected evaluation images grouped by language bucket.

