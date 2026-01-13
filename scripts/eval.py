from multiprocessing import freeze_support
from pathlib import Path
import json
import torch

from src.models import build_model
from src.datasets import build_dataset, build_dataloader
from src.losses import build_criterion
from src.utils import setup_logger, set_seed, load_checkpoint
from src.engine.eval.evaluate import evaluate


def main():
    # Chỉnh sửa ở đây
    DATASET_ROOT = r"C:\Dataset\CUB_200_2011"
    EXP_DIR      = r"outputs\exp_001"
    CKPT_NAME    = "best.pth"   # hoặc "last.pth"
    NUM_CLASSES  = 200
    IMG_SIZE     = 224
    BATCH_SIZE   = 64
    NUM_WORKERS  = 0
    SEED         = 42
    LOSS_NAME    = "ce"
    #---------------

    exp_dir = Path(EXP_DIR)
    ckpt_path = exp_dir / CKPT_NAME
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(exp_dir, name="eval", log_filename="eval.log")
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = (device.type == "cuda")
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {ckpt_path}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    val_ds = build_dataset(
        dataset_root=DATASET_ROOT,
        split="val",
        splits_dir="data/splits",
        img_size=IMG_SIZE,
    )
    val_loader = build_dataloader(
        val_ds,
        split="val",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    logger.info(f"Val samples: {len(val_ds)}")

    model = build_model("resnet50", num_classes=NUM_CLASSES).to(device)
    _ = load_checkpoint(str(ckpt_path), model, optimizer=None, scheduler=None, map_location=str(device))

    criterion = build_criterion(LOSS_NAME, label_smoothing_eps=0.0).to(device)

    stats = evaluate(model=model, loader=val_loader, criterion=criterion, device=device)

    logger.info(f"[EVAL] loss={stats['loss']:.4f}, top1={stats['top1']:.2f}, top5={stats['top5']:.2f}")

    out_json = exp_dir / "eval_stats.json"
    out_json.write_text(
        json.dumps(
            {"checkpoint": str(ckpt_path), "dataset_root": DATASET_ROOT, "stats": stats},
            indent=2
        ),
        encoding="utf-8"
    )
    logger.info(f"Saved: {out_json}")


if __name__ == "__main__":
    freeze_support()
    main()