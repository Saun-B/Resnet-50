import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

def get_data_dir() -> Path:
    return Path(__file__).resolve().parents[1]

def read_two_col_int(path: Path) -> dict[int, int]:
    d = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split()
            d[int(a)] = int(b)
    return d

def read_images_txt(path: Path) -> dict[int, str]:
    d = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_id_str, rel = line.split(maxsplit=1)
            d[int(img_id_str)] = rel.strip()
    return d

def normalize_rel_path(rel_from_images_txt: str) -> str:
    rel = rel_from_images_txt.replace("\\", "/").lstrip("/")
    if not rel.startswith("images/"):
        rel = "images/" + rel
    return rel

def balanced_sample_per_class(
    items_by_class: dict[int, list[int]],
    total: int,
    rng: random.Random,
) -> list[int]:
    
    classes = sorted(items_by_class.keys())
    available_total = sum(len(v) for v in items_by_class.values())
    if total >= available_total:
        out = []
        for c in classes:
            out.extend(items_by_class[c])
        rng.shuffle(out)
        return out

    base_quota = total// len(classes)
    remainder = total % len(classes)

    for c in classes:
        rng.shuffle(items_by_class[c])

    selected = []
    leftovers = []

    for c in classes:
        take = min(base_quota, len(items_by_class[c]))
        selected.extend(items_by_class[c][:take])
        if take < len(items_by_class[c]):
            leftovers.extend(items_by_class[c][take:])

    rng.shuffle(leftovers)
    selected.extend(leftovers[:remainder])

    if len(selected) < total:
        remaining_need = total - len(selected)
        rest = leftovers[remainder:]
        rng.shuffle(rest)
        selected.extend(rest[:remaining_need])

    rng.shuffle(selected)
    return selected[:total]

def main():
    ap = argparse.ArgumentParser("Prepare CUB-200-2011 splits/meta for demo training.")
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--train_images_total", type=int, default=10000)
    ap.add_argument("--num_classes", type=int, default=200)
    ap.add_argument("--pool", type=str, default="all", choices=["all", "official_train", "official_test"])
    ap.add_argument("--max_val_images", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

    images_txt = dataset_root / "images.txt"
    labels_txt = dataset_root / "image_class_labels.txt"
    split_txt  = dataset_root / "train_test_split.txt"

    for p in [images_txt, labels_txt, split_txt]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    id_to_rel = read_images_txt(images_txt)
    id_to_label_1based = read_two_col_int(labels_txt)
    id_to_is_train = read_two_col_int(split_txt)

    records = []
    for img_id, rel0 in id_to_rel.items():
        if img_id not in id_to_label_1based or img_id not in id_to_is_train:
            continue
        rel = normalize_rel_path(rel0)
        label0 = id_to_label_1based[img_id] - 1
        is_train = id_to_is_train[img_id] == 1
        records.append((img_id, rel, label0, is_train))

    if len(records) == 0:
        raise RuntimeError("No records parsed. Check your CUB files.")

    if args.num_classes != 200:
        print("CUB-200-2011 normally has 200 classes.")

    if args.pool == "official_train":
        pool = [r for r in records if r[3] is True]
        if args.train_images_total > len(pool):
            print(f"pool=official_train only has {len(pool)} images, "
                  f"but you requested train_images_total={args.train_images_total}. "
                  f"-> Switching pool to 'all' to satisfy 10k.")
            pool = records

    elif args.pool == "official_test":
        pool = [r for r in records if r[3] is False]
        if args.train_images_total > len(pool):
            print(f"pool=official_test only has {len(pool)} images, "
                  f"but you requested train_images_total={args.train_images_total}. "
                  f"-> Switching pool to 'all'.")
            pool = records

    else:
        pool = records

    by_class = defaultdict(list)
    id_to_rel2 = {}
    id_to_label0 = {}
    label_to_classname = {}

    for img_id, rel, label0, _is_train in pool:
        by_class[label0].append(img_id)
        id_to_rel2[img_id] = rel
        id_to_label0[img_id] = label0
        parts = rel.split("/")
        if len(parts) >= 2:
            cls_folder = parts[1]
            label_to_classname.setdefault(label0, cls_folder)

    missing_labels = [c for c in range(args.num_classes) if c not in by_class]
    if missing_labels:
        print(f"Missing labels in pool (count={len(missing_labels)}). Example: {missing_labels[:10]}")

    selected_train_ids = balanced_sample_per_class(by_class, args.train_images_total, rng)
    train_set = set(selected_train_ids)

    remaining_ids = [img_id for img_id, _, _, _ in pool if img_id not in train_set]
    rng.shuffle(remaining_ids)

    if args.max_val_images is not None and args.max_val_images > 0:
        remaining_ids = remaining_ids[:args.max_val_images]

    val_set = set(remaining_ids)

    classes = []
    class_map = {}
    for label0 in range(args.num_classes):
        name = label_to_classname.get(label0, f"class_{label0:03d}")
        classes.append(name)
        class_map[name] = label0

    data = get_data_dir()
    splits_dir = data / "splits"
    meta_dir = data / "meta"
    splits_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    train_lines = [f"{id_to_rel2[i]}\t{id_to_label0[i]}" for i in selected_train_ids]
    val_lines = [f"{id_to_rel2[i]}\t{id_to_label0[i]}" for i in remaining_ids]

    (splits_dir / "train_list.txt").write_text("\n".join(train_lines), encoding="utf-8")
    (splits_dir / "val_list.txt").write_text("\n".join(val_lines), encoding="utf-8")

    (meta_dir / "class_map.json").write_text(json.dumps(class_map, indent=2), encoding="utf-8")
    (meta_dir / "classes.txt").write_text("\n".join(classes), encoding="utf-8")

    per_class_train = defaultdict(int)
    for i in selected_train_ids:
        per_class_train[id_to_label0[i]] += 1
    per_class_val = defaultdict(int)
    for i in remaining_ids:
        per_class_val[id_to_label0[i]] += 1

    stats = {
        "dataset": "CUB-200-2011",
        "dataset_root": str(dataset_root),
        "pool": args.pool,
        "seed": args.seed,
        "num_classes": args.num_classes,
        "train_images_total": len(selected_train_ids),
        "val_images_total": len(remaining_ids),
        "per_class_train_min_mean_max": [
            int(min(per_class_train.values())) if per_class_train else 0,
            float(sum(per_class_train.values()) / max(1, len(per_class_train))),
            int(max(per_class_train.values())) if per_class_train else 0,
        ],
        "per_class_val_min_mean_max": [
            int(min(per_class_val.values())) if per_class_val else 0,
            float(sum(per_class_val.values()) / max(1, len(per_class_val))),
            int(max(per_class_val.values())) if per_class_val else 0,
        ],
    }
    (meta_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Wrote:")
    print(f"  - {splits_dir / 'train_list.txt'}  ({len(train_lines)} lines)")
    print(f"  - {splits_dir / 'val_list.txt'}    ({len(val_lines)} lines)")
    print(f"  - {meta_dir / 'class_map.json'}")
    print(f"  - {meta_dir / 'classes.txt'}")
    print(f"  - {meta_dir / 'dataset_stats.json'}")

if __name__ == "__main__":
    main()