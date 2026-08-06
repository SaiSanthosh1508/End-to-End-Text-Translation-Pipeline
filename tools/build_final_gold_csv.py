#!/usr/bin/env python3
"""Build the final gold English evaluation CSV from agreement, majority vote, and manual resolutions."""

from __future__ import annotations

import csv
from pathlib import Path


MANUAL_RESOLUTIONS = {
    "tr_img_00578": "Traders' Syndicate",
    "tr_img_03149": "Beef Soup and Steamed Dishes",
    "tr_img_03218": "Senior Year, Class 16 — Mom, I'm coming back!",
    "tr_img_03297": "Peking University dream comes true",
    "tr_img_03315": "Zhonghang Science and Technology University",
    "tr_img_03505": "Yun Xiang Xuan — Hunan Cuisine",
    "tr_img_03716": "Welcome the Olympics",
    "tr_img_03806": "Old Tianjin Zhajiang Noodles",
    "tr_img_03893": "Imperial grain and state taxes—failure to pay is a crime.",
    "tr_img_05083": "We puff grains and beans for you",
    "tr_img_06008": "Yōten De-e",
    "tr_img_06501": "Longévité Motosumiyoshi",
    "tr_img_06738": "Turn left when exiting",
    "train_ReCTS_019510": "Night Fire BBQ Bar",
}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    root = Path("/Users/saisanthosh/Documents/paper-exp/End-to-End-Text-Translation-Pipeline")
    master = load_csv(root / "translation_subset/english_only_master_annotation.csv")
    agreed = load_csv(root / "translation_subset/codex_comparison/agreed_rows.csv")
    majority = load_csv(root / "translation_subset/codex_majority_vote/accepted_majority_rows.csv")

    agreed_map = {row["image_id"]: row for row in agreed}
    majority_map = {row["image_id"]: row for row in majority}

    output_path = root / "translation_subset/english_only_master_annotation_gold.csv"
    fieldnames = list(master[0].keys()) + ["finalization_source"]

    written = []
    for row in master:
        image_id = row["image_id"]
        out = dict(row)

        if image_id in agreed_map:
            out["target_translation_reference"] = agreed_map[image_id]["draft_a_translation"]
            out["annotator_1"] = "agreement_auto_accept"
            out["finalization_source"] = "agreed_two_pass"
        elif image_id in majority_map:
            out["target_translation_reference"] = majority_map[image_id]["final_translation"]
            out["annotator_1"] = "majority_vote_accept"
            out["finalization_source"] = "agreed_two_of_three"
        elif image_id in MANUAL_RESOLUTIONS:
            out["target_translation_reference"] = MANUAL_RESOLUTIONS[image_id]
            out["annotator_1"] = "assistant_manual_review"
            out["finalization_source"] = "assistant_manual_resolution"
        else:
            out["finalization_source"] = ""

        written.append(out)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(written)

    print(f"wrote={output_path}")
    print(f"rows={len(written)}")
    print(f"manual_resolved={len(MANUAL_RESOLUTIONS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
