#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from salf_cub200 import CUB200, SALFCBM, eval_transform, load_backbone, load_cbl, device_from_arg

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--cbl", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-json", default="cub_accuracy.json")
    args = p.parse_args()

    device = device_from_arg(args.device)
    model_dir = Path(args.model_dir)
    backbone = load_backbone(Path(args.backbone), device)
    cbl = load_cbl(Path(args.cbl), device)
    mean = torch.load(model_dir/"proj_mean.pt", map_location="cpu", weights_only=True)
    std = torch.load(model_dir/"proj_std.pt", map_location="cpu", weights_only=True)
    W = torch.load(model_dir/"W_g.pt", map_location="cpu", weights_only=True)
    b = torch.load(model_dir/"b_g.pt", map_location="cpu", weights_only=True)

    head = nn.Linear(W.shape[1], W.shape[0])
    with torch.no_grad():
        head.weight.copy_(W)
        head.bias.copy_(b)

    model = SALFCBM(backbone, cbl, head, mean, std).to(device).eval()
    ds = CUB200(args.data, "test", eval_transform(224))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    total = top1 = top5 = 0
    per_correct = torch.zeros(200, dtype=torch.long)
    per_total = torch.zeros(200, dtype=torch.long)

    with torch.inference_mode():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            logits, _, _ = model(images)
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            pred1 = logits.argmax(1)
            pred5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
            top1 += (pred1 == labels).sum().item()
            top5 += pred5.eq(labels[:, None]).any(1).sum().item()
            total += labels.numel()
            for cls in labels.unique():
                c = int(cls)
                mask = labels == cls
                per_total[c] += int(mask.sum())
                per_correct[c] += int((pred1[mask] == labels[mask]).sum())

    result = {
        "num_test_images": total,
        "top1_accuracy": top1 / total,
        "top5_accuracy": top5 / total,
        "mean_per_class_accuracy": (per_correct.float()/per_total.clamp_min(1)).mean().item(),
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
