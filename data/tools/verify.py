import argparse
import json
from pathlib import Path
from collections import defaultdict

def read_list(list_path: Path):
    items = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Bad line at {list_path}:{line_no}: {line}")
            rel_path = parts[0]
            label = int(parts[1])
            items.append((rel_path, label))
    return items

def main():
    ap = argparse.ArgumentParser(description="Verify train_list/val_list against dataset_root.")
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--train_list", type=str, default=None)
    ap.add_argument("--val_list", type=str, default=None)
    ap.add_argument("--class_map", type=str, default=None)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

    tools_dir = Path(__file__).resolve().parent
    two_data_dir = tools_dir.parent
    splits_dir = two_data_dir / "splits"
    meta_dir = two_data_dir / "meta"

    train_list = Path(args.train_list) if args.train_list else (splits_dir / "train_list.txt")
    val_list = Path(args.val_list) if args.val_list else (splits_dir / "val_list.txt")
    if not train_list.exists():
        raise FileNotFoundError(f"train_list not found: {train_list}")
    if not val_list.exists():
        raise FileNotFoundError(f"val_list not found: {val_list}")

    class_map_path = Path(args.class_map) if args.class_map else (meta_dir / "class_map.json")
    num_classes = None
    if class_map_path.exists():
        class_map = json.loads(class_map_path.read_text(encoding="utf-8"))
        num_classes = len(class_map)

    train_items = read_list(train_list)
    val_items = read_list(val_list)

    def check(items):
        missing = 0
        bad_label = 0
        per_class = defaultdict(int)
        for rel, y in items:
            p = dataset_root / rel
            if not p.exists():
                missing += 1
            if num_classes is not None and not (0 <= y < num_classes):
                bad_label += 1
            per_class[y] += 1
        return missing, bad_label, per_class

    train_missing, train_bad, train_counts = check(train_items)
    val_missing, val_bad, val_counts = check(val_items)

    train_set = set([r for r, _ in train_items])
    val_set = set([r for r, _ in val_items])
    overlap = train_set.intersection(val_set)

    print(f"dataset_root: {dataset_root}")
    print(f"train: {len(train_items)} | val: {len(val_items)}")
    if num_classes is not None:
        print(f"num_classes (from class_map): {num_classes}")

    print("\nRESULTS")
    print(f"Missing files: train={train_missing}, val={val_missing}")
    if num_classes is not None:
        print(f"Bad labels:   train={train_bad}, val={val_bad}")
    print(f"Overlap train/val: {len(overlap)}")

    def summarize(counts, name):
        if not counts:
            return
        vals = list(counts.values())
        print(f"\n[{name}] per-class count: min/mean/max = "
              f"{min(vals)} / {sum(vals)/len(vals):.2f} / {max(vals)}")

    summarize(train_counts, "TRAIN")
    summarize(val_counts, "VAL")

    if train_missing == 0 and val_missing == 0 and train_bad == 0 and val_bad == 0 and len(overlap) == 0:
        print("\nok")
    else:
        print("\nIssues found")

if __name__ == "__main__":
    main()