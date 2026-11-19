from pathlib import Path
import csv
import random
from torchvision.datasets import GTSRB

# Run this script from the root of your traffic_sign_recognition project.
# It uses the GTSRB test set already downloaded in ./data; no internet is needed.

ROOT = Path("data")
OUT = Path("test_images")
SEED = 42

# 20 representative classes, grouped for convenient manual testing.
CLASSES = {
    1: "speed_limit_30",
    2: "speed_limit_50",
    3: "speed_limit_60",
    4: "speed_limit_70",
    9: "no_passing",
    10: "no_passing_heavy_vehicles",
    14: "stop",
    15: "no_vehicles",
    17: "no_entry",
    18: "general_caution",
    19: "dangerous_curve_left",
    20: "dangerous_curve_right",
    25: "road_work",
    27: "pedestrians",
    28: "children_crossing",
    12: "priority_road",
    13: "yield",
    33: "turn_right_ahead",
    35: "ahead_only",
    38: "keep_right",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = GTSRB(root=str(ROOT), split="test", download=False)

    wanted = set(CLASSES)
    candidates = {class_id: [] for class_id in wanted}

    print("Dang tim anh trong bo GTSRB test da tai...")
    for index in range(len(dataset)):
        _, label = dataset[index]
        label = int(label)
        if label in wanted:
            candidates[label].append(index)

    missing = [class_id for class_id, indices in candidates.items() if not indices]
    if missing:
        raise RuntimeError(
            f"Khong tim thay anh cho cac lop {missing}. "
            "Hay kiem tra bo du lieu GTSRB trong thu muc data/."
        )

    rng = random.Random(SEED)
    rows = []
    for class_id, name in CLASSES.items():
        index = rng.choice(candidates[class_id])
        image, actual_label = dataset[index]
        filename = f"class_{class_id:02d}_{name}.png"
        image.convert("RGB").save(OUT / filename)
        rows.append([filename, class_id, name, "GTSRB official test split"])
        print(f"Da tao: {filename}")

    with (OUT / "labels.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "true_class_id", "class_name", "source"])
        writer.writerows(rows)

    print("\nHOAN TAT: Da tao 20 anh trong thu muc test_images/")
    print("Bang nhan dung: test_images/labels.csv")
    print("Cac anh duoc lay tu tap test GTSRB da co tren may; khong tai them du lieu.")


if __name__ == "__main__":
    main()
