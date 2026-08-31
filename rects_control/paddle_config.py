"""Character dictionary and PP-OCRv5 fine-tuning config for ReCTS.

The dictionary is the union of EasyOCR's simplified-Chinese character list with
digits, ASCII letters and the punctuation ReCTS transcriptions use. It is rebuilt
here rather than shipped so the recogniser's label space is auditable.
"""

from __future__ import annotations

from pathlib import Path

import requests
import yaml

EASYOCR_CHARS = (
    "https://raw.githubusercontent.com/JaidedAI/EasyOCR/master/easyocr/character/"
    "ch_sim_char.txt"
)
DIGITS = "0123456789"
LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYMBOLS = " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~€°"
BLANK, UNK = "[blank]", "[UNK]"


def build_dictionary(path: Path) -> int:
    """Write the PaddleOCR dictionary and return its character count."""
    response = requests.get(EASYOCR_CHARS, timeout=60)
    response.raise_for_status()

    charset = set(response.text) | set(DIGITS) | set(LETTERS) | set(SYMBOLS)
    characters = sorted(c for c in charset if c not in "\n\r\t")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([BLANK, *characters, UNK]) + "\n", encoding="utf-8"
    )
    return len(characters)


def patch_training_config(
    config: Path, *, data_root: Path, dictionary: Path, pretrained: Path,
    epochs: int, batch_size: int, save_model_dir: Path, workers: int = 2,
) -> None:
    """Point the stock PP-OCRv5 mobile recipe at the ReCTS crops.

    ``save_model_dir`` must be absolute. The stock value is ``./output/...``, and a
    relative path there cost a completed 12 h fine-tune: the checkpoints were written
    somewhere the session snapshot did not capture, so the run finished and left
    nothing behind.
    """
    if not save_model_dir.is_absolute():
        raise ValueError(f"save_model_dir must be absolute, got {save_model_dir}")

    spec = yaml.safe_load(config.read_text())

    spec["Global"]["save_model_dir"] = str(save_model_dir)
    spec["Global"]["pretrained_model"] = str(pretrained)
    spec["Global"]["character_dict_path"] = str(dictionary)
    spec["Global"]["epoch_num"] = epochs

    for section, labels in (("Train", "rec_gt_train.txt"), ("Eval", "rec_gt_test.txt")):
        spec[section]["dataset"]["data_dir"] = str(data_root)
        spec[section]["dataset"]["label_file_list"] = [str(data_root / labels)]
        spec[section]["loader"]["num_workers"] = workers
        spec[section]["loader"]["batch_size_per_card"] = batch_size
    spec["Train"]["sampler"]["first_bs"] = batch_size

    config.write_text(yaml.dump(spec, default_flow_style=False, allow_unicode=True))


def retarget_for_export(config: Path, dictionary: Path) -> None:
    """Repoint a recovered training config at paths that exist in this session.

    A config recovered from a finished run still names that run's ``/kaggle/working``
    paths. Export only reads the architecture and the label space, so the dictionary
    is the one path that has to be corrected; leaving it stale produces a model whose
    output indices mean nothing.
    """
    spec = yaml.safe_load(config.read_text())
    spec["Global"]["character_dict_path"] = str(dictionary)
    config.write_text(yaml.dump(spec, default_flow_style=False, allow_unicode=True))
