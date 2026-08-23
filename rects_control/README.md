# ReCTS detector control

Answers reviewer item 2: *"stock YOLOv11 plus your same fine-tuned OCR, so readers can
see what your detector changes contribute to end-to-end 1-NED on their own."*

Two detectors, one recogniser, everything else identical. Any 1-NED difference is then
attributable to the architecture rather than to recognition.

## Why the recogniser has to be retrained first

The published ReCTS recogniser no longer exists. `paddleocr-training-ReCTS.ipynb`
exported it to `/kaggle/working/PP-OCRv5_server_rec_infer`, but the archiving line was
commented out, so nothing was written to the version output before the session was
cleared. The detector survived — `rishiksaisanthosh/yolo-obb-rects` still holds
`runs/obb/train3/weights/best.pt` — the recogniser did not.

Separately: the released `generate_rrc_task4_e2e_submission` calls **EasyOCR**, not the
fine-tuned PaddleOCR the paper describes. Which back-end produced the published 0.8566
is therefore unresolved. `recognizer.py` provides both so the question can be settled.

## Run order

| | notebook | GPU | produces |
|-|----------|-----|----------|
| 1 | `rects_recognizer_kaggle.ipynb` | ~3 h | fine-tuned PP-OCRv5, exported **and archived** |
| 2 | `rects_control_kaggle.ipynb` | ~5.5 h | RRC Task 3 + Task 4 submissions, both arms |

Notebook 2 attaches notebook 1's output plus `rishiksaisanthosh/yolo-obb-rects`, and
reuses the published detector for the paper arm instead of retraining it. Set
`TRAIN_PAPER_ARM = True` to train that arm here too, which is what a multi-seed
comparison needs — 4.4 h per arm per seed.

## Training recipe

Taken from the published run's `args.yaml`, not chosen: 100 epochs, batch 64,
imgsz 480, cosine LR, `pretrained=True`, `deterministic=True`, seed 42, two GPUs.
`stock` is `ablation/configs/a1_stock.yaml`; `paper` is `full_legacy.yaml`, the
deployed BiFPN + MS-CBAM + cross-attention model. Both are rewritten to `nc=1`.

## Modules

| file | responsibility |
|------|----------------|
| `crops.py` | axis-aligned line crops and PaddleOCR label files from ReCTS json |
| `paddle_config.py` | character dictionary, PP-OCRv5 config patching |
| `detectors.py` | per-arm config and contained training |
| `recognizer.py` | PaddleOCR and EasyOCR back-ends behind one protocol |
| `submission.py` | RRC Task 3 / Task 4 writers |

`pytest rects_control/` covers the parts that fail silently: the clockwise point order
is checked against a verbatim transcription of the released script, because a wrong
order still produces a well-formed file and only shows up as a bad server score days
later.

## Interpreting the result

Report the paired 1-NED whichever way it falls. If the gap is small, the framing the
reviewer already offered — a deployment and systems contribution — is the honest one,
and the seed spread in the MLT ablation is the yardstick for whether a gap means
anything.
