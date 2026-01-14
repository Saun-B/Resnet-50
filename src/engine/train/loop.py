from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import torch

from src.models import build_model
from src.datasets import build_dataset, build_dataloader
from src.losses import build_criterion
from src.optim import build_optimizer, build_scheduler
from src.utils import set_seed, setup_logger, CSVLogger, save_checkpoint, load_checkpoint

from .train_one_epoch import train_one_epoch
from src.engine.eval.evaluate import evaluate

def train(
    dataset_root: str,
    out_dir: str = "outputs/exp_001",
    num_classes: int = 200,
    img_size: int = 224,
    epochs: int = 10,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,

    loss_name: str = "ce",
    label_smoothing_eps: float = 0.1,

    optim_name: str = "sgd",
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,

    sched_name: str = "step",
    step_size: int = 5,
    gamma: float = 0.1,
    t_max: int = 10,
    warmup_steps: int = 0,

    log_interval: int = 50,
    grad_clip_norm: float | None = None,
    amp: bool = True,
    resume_path: str | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir, name="train", log_filename="train.log")
    csv_logger = CSVLogger(out_dir / "metrics.csv")

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    pin_memory = (device.type == "cuda")

    train_ds = build_dataset(dataset_root, split="train", splits_dir="data/splits", img_size=img_size)
    val_ds = build_dataset(dataset_root, split="val", splits_dir="data/splits", img_size=img_size)

    train_loader = build_dataloader(train_ds, split="train", batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = build_dataloader(val_ds, split="val", batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    logger.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    model = build_model("resnet50", num_classes=num_classes)
    model.to(device)

    criterion = build_criterion(loss_name, label_smoothing_eps=label_smoothing_eps)
    criterion.to(device)

    optimizer = build_optimizer(
        model,
        name=optim_name,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
    )

    scheduler = None
    if sched_name.lower() in ("warmup_cosine", "warmupcosine"):
        total_steps = epochs * len(train_loader)
        scheduler = build_scheduler(
            optimizer,
            name="warmup_cosine",
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )
        scheduler._step_per_iteration = True
    else:
        scheduler = build_scheduler(
            optimizer,
            name=sched_name,
            step_size=step_size,
            gamma=gamma,
            t_max=t_max,
        )
        if scheduler is not None:
            scheduler._step_per_iteration = False

    use_amp = amp and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    logger.info(f"AMP enabled: {use_amp}")

    start_epoch = 1
    best_top1 = -1.0
    if resume_path is not None:
        ckpt = load_checkpoint(resume_path, model, optimizer=optimizer, scheduler=scheduler, map_location=str(device))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_top1 = float(ckpt.get("best_top1", -1.0))
        logger.info(f"Resumed from {resume_path} | start_epoch={start_epoch} | best_top1={best_top1:.2f}")

    for epoch in range(start_epoch, epochs + 1):
        logger.info(f"=== Epoch {epoch}/{epochs} ===")

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            logger=logger,
            scheduler=scheduler,
            scaler=scaler,
            log_interval=log_interval,
            grad_clip_norm=grad_clip_norm,
        )

        if scheduler is not None and not getattr(scheduler, "_step_per_iteration", False):
            scheduler.step()

        val_stats = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"[Epoch {epoch}] "
            f"TRAIN: loss={train_stats['loss']:.4f}, top1={train_stats['top1']:.2f}, top5={train_stats['top5']:.2f} | "
            f"VAL: loss={val_stats['loss']:.4f}, top1={val_stats['top1']:.2f}, top5={val_stats['top5']:.2f} | "
            f"lr={lr_now:.6f}"
        )

        csv_logger.log({
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_stats["loss"],
            "train_top1": train_stats["top1"],
            "train_top5": train_stats["top5"],
            "val_loss": val_stats["loss"],
            "val_top1": val_stats["top1"],
            "val_top5": val_stats["top5"],
        })

        last_path = out_dir / "last.pth"
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_top1": best_top1,
            "num_classes": num_classes,
        }
        save_checkpoint(state, last_path)

        if val_stats["top1"] > best_top1:
            best_top1 = val_stats["top1"]
            best_path = out_dir / "best.pth"
            state["best_top1"] = best_top1
            save_checkpoint(state, best_path)
            logger.info(f"New best top1={best_top1:.2f} -> saved {best_path}")

    csv_logger.close()
    logger.info(f"Training done. Best top1={best_top1:.2f}")
    logger.info(f"Outputs saved in: {out_dir}")