# Class-wise MLT-2019, revalidated

Produced by `revalidate.py` from `best.pt` on the 1,000-image validation split,
imgsz 480, CPU. Overall mAP50 came back **0.6635** against Fig. 11(b)'s 0.663, which
confirms this checkpoint is the one behind the figure.

| class | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Arabic | 89.0 | 72.5 | 80.6 | 65.3 |
| Latin | 82.8 | 55.2 | 65.4 | 49.2 |
| Chinese | 77.7 | 66.5 | 72.4 | 57.3 |
| Korean | 81.8 | 56.5 | 68.4 | 51.8 |
| Japanese | 71.6 | 39.9 | 46.8 | 34.9 |
| Bangla | 87.7 | 88.7 | 92.3 | 77.1 |
| Hindi | 94.6 | 93.9 | 96.7 | 87.5 |
| Other | 61.7 | 2.5 | 8.2 | 4.6 |
| **all** | **80.9** | **59.5** | **66.4** | **53.5** |

Agrees with `results.csv` at epoch 100 (P 81.0, R 59.1, mAP50 66.1, mAP50-95 53.2).

## Versus the Table 5 currently in the manuscript

Precision is accurate; recall and mAP are not.

| column | manuscript mean | revalidated mean | delta |
| --- | ---: | ---: | ---: |
| precision | 80.3 | 80.9 | +0.6 |
| recall | 68.3 | 59.5 | **-8.8** |
| mAP50 | 73.8 | 66.3 | **-7.5** |
| mAP50-95 | 60.6 | 53.5 | **-7.1** |

All eight classes move the same direction. The largest gap is Japanese, whose mAP50
is 46.8 rather than 62.5.

## Ready to paste

```latex
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{mAP0.5} & \textbf{mAP0.5-0.95} \\
\hline
Arabic   & 89.0 & 72.5 & 80.6 & 65.3 \\
Latin    & 82.8 & 55.2 & 65.4 & 49.2 \\
Chinese  & 77.7 & 66.5 & 72.4 & 57.3 \\
Korean   & 81.8 & 56.5 & 68.4 & 51.8 \\
Japanese & 71.6 & 39.9 & 46.8 & 34.9 \\
Bangla   & 87.7 & 88.7 & 92.3 & 77.1 \\
Hindi    & 94.6 & 93.9 & 96.7 & 87.5 \\
Other    & 61.7 & 2.5 & 8.2 & 4.6 \\
\hline
\end{tabular}
```

Class 7 is `Other` here, matching the model, `dataset.yaml` and Fig. 11(b). The
manuscript calls it `Symbols`.

## Dependent text

`main.tex:402` claims Chinese, Korean and Japanese perform comparatively well.
Japanese is the weakest script class at 46.8 mAP50 and 39.9 recall, so that sentence
needs replacing whenever the table is.

The `Other` class reaches 2.5 recall, so its stated purpose in Section III-A —
introduced to suppress false positives — is not being served in practice.
