"""The label space is the one thing export cannot get wrong silently."""

from __future__ import annotations

from pathlib import Path

import yaml

from rects_control.paddle_config import patch_training_config, retarget_for_export

STOCK = {
    "Global": {"character_dict_path": "/gone/dict.txt", "epoch_num": 500,
               "pretrained_model": None},
    "Train": {"dataset": {"data_dir": "/gone", "label_file_list": ["/gone/x.txt"]},
              "loader": {"num_workers": 8, "batch_size_per_card": 128},
              "sampler": {"first_bs": 128}},
    "Eval": {"dataset": {"data_dir": "/gone", "label_file_list": ["/gone/y.txt"]},
             "loader": {"num_workers": 8, "batch_size_per_card": 128}},
}


def write_stock(tmp_path: Path) -> Path:
    config = tmp_path / "rec.yml"
    config.write_text(yaml.dump(STOCK))
    return config


def test_retarget_replaces_only_the_dictionary(tmp_path: Path) -> None:
    config = write_stock(tmp_path)
    retarget_for_export(config, tmp_path / "custom.txt")

    spec = yaml.safe_load(config.read_text())
    assert spec["Global"]["character_dict_path"] == str(tmp_path / "custom.txt")
    assert spec["Global"]["epoch_num"] == 500
    assert spec["Train"]["dataset"]["data_dir"] == "/gone"


def test_training_config_points_every_path_at_the_crops(tmp_path: Path) -> None:
    config = write_stock(tmp_path)
    patch_training_config(
        config, data_root=tmp_path / "rec", dictionary=tmp_path / "d.txt",
        pretrained=tmp_path / "p.pdparams", epochs=35, batch_size=64,
    )

    spec = yaml.safe_load(config.read_text())
    assert spec["Global"]["epoch_num"] == 35
    assert spec["Train"]["sampler"]["first_bs"] == 64
    assert spec["Eval"]["loader"]["batch_size_per_card"] == 64
    assert spec["Train"]["dataset"]["label_file_list"] == [str(tmp_path / "rec/rec_gt_train.txt")]
    assert spec["Eval"]["dataset"]["label_file_list"] == [str(tmp_path / "rec/rec_gt_test.txt")]
