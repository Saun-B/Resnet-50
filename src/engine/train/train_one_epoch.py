from __future__ import annotations

from typing import Dict, Optional
import time

import torch
import torch.nn as nn

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger,
    scheduler=None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    log_interval: int = 50,
    grad_clip_norm: float | None = None,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_samples = 0
    correct1 = 0
    correct5 = 0

    start = time.time()

    use_amp = (scaler is not None) and (device.type == "cuda")

    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = torch.as_tensor(targets, dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()

            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()

        if scheduler is not None and getattr(scheduler, "_step_per_iteration", False):
            scheduler.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        maxk = min(5, logits.size(1))
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        correct = pred.eq(targets.view(-1, 1))

        correct1 += correct[:, :1].sum().item()
        correct5 += correct[:, :maxk].sum().item()

        if (step % log_interval) == 0:
            lr = optimizer.param_groups[0]["lr"]
            avg_loss_so_far = total_loss / max(1, total_samples)
            top1_so_far = 100.0 * correct1 / max(1, total_samples)
            top5_so_far = 100.0 * correct5 / max(1, total_samples)
            elapsed = time.time() - start
            logger.info(
                f"Epoch {epoch} | Step {step}/{len(loader)} | "
                f"lr={lr:.6f} | loss={avg_loss_so_far:.4f} | "
                f"top1={top1_so_far:.2f} | top5={top5_so_far:.2f} | "
                f"time={elapsed:.1f}s"
            )

    avg_loss = total_loss / max(1, total_samples)
    top1 = 100.0 * correct1 / max(1, total_samples)
    top5 = 100.0 * correct5 / max(1, total_samples)

    return {"loss": avg_loss, "top1": top1, "top5": top5}