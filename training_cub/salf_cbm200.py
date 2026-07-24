#!/usr/bin/env python3
"""
Paper-guided SALF-CBM reproduction for CUB-200-2011.

Stages:
  1) train-backbone       ResNet-18 on CUB species labels
  2) make-concepts        Build a concept vocabulary without per-image concept labels
  3) clip-targets         Precompute 7x7 red-circle CLIP pseudo-label maps
  4) cache-features       Cache frozen ResNet spatial features
  5) train-cbl            Fit a 1x1 spatial concept projection with cubic-cosine loss
  6) train-classifier     Softmax-pool, normalize, and train elastic-net linear head
  7) evaluate             Evaluate complete SALF-CBM and export compatible weights

This is a faithful, practical reimplementation of the method described in:
Benou & Riklin-Raviv, CVPR 2025, "Show and Tell".

The authors' public repository currently focuses on demos/inference. Consequently,
some engineering choices here (notably minibatch estimation of the dataset-level
cubic-cosine objective and proximal Adam for the sparse head) are reproduction
choices rather than copied official training code.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

try:
    import open_clip
except ImportError:
    open_clip = None


# ----------------------------- reproducibility ----------------------------- #

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_from_arg(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# -------------------------------- dataset ---------------------------------- #

class CUB200(Dataset):
    """Reads the official CUB metadata and preserves its standard train/test split."""

    def __init__(self, root: str | Path, split: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        if split not in {"train", "test", "all"}:
            raise ValueError("split must be train, test, or all")

        images = self._read_map(self.root / "images.txt", value_type=str)
        labels = self._read_map(self.root / "image_class_labels.txt", value_type=int)
        train_flags = self._read_map(self.root / "train_test_split.txt", value_type=int)

        records = []
        for image_id in sorted(images):
            is_train = bool(train_flags[image_id])
            if split == "train" and not is_train:
                continue
            if split == "test" and is_train:
                continue
            records.append(
                {
                    "image_id": image_id,
                    "path": self.root / "images" / images[image_id],
                    "label": labels[image_id] - 1,
                }
            )
        self.records = records

    @staticmethod
    def _read_map(path: Path, value_type):
        result = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                key, value = line.rstrip("\n").split(maxsplit=1)
                result[int(key)] = value_type(value)
        return result

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record["path"]).convert("RGB")
        x = self.transform(image) if self.transform else image
        return x, record["label"], record["image_id"]


class TensorShardDataset(Dataset):
    """Loads one tensor per image from cache_dir/{image_id}.pt."""

    def __init__(self, cub: CUB200, cache_dir: str | Path):
        self.cub = cub
        self.cache_dir = Path(cache_dir)

    def __len__(self):
        return len(self.cub)

    def __getitem__(self, index):
        _, label, image_id = self.cub[index]
        path = self.cache_dir / f"{image_id:05d}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        value = torch.load(path, map_location="cpu", weights_only=True)
        return value, label, image_id


class PairedCacheDataset(Dataset):
    def __init__(self, cub: CUB200, feature_dir: str | Path, target_dir: str | Path):
        self.cub = cub
        self.feature_dir = Path(feature_dir)
        self.target_dir = Path(target_dir)

    def __len__(self):
        return len(self.cub)

    def __getitem__(self, index):
        _, label, image_id = self.cub[index]
        feature = torch.load(
            self.feature_dir / f"{image_id:05d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        target = torch.load(
            self.target_dir / f"{image_id:05d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        return feature, target, label, image_id


# ------------------------------- transforms -------------------------------- #

def train_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def eval_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


# --------------------------------- models ---------------------------------- #

class ResNet18Spatial(nn.Module):
    def __init__(self, num_classes: int = 200, imagenet_init: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if imagenet_init else None
        net = models.resnet18(weights=weights)
        self.stem_to_layer4 = nn.Sequential(*list(net.children())[:-2])
        self.classifier = nn.Linear(512, num_classes)

    def features(self, x):
        return self.stem_to_layer4(x)

    def forward(self, x):
        feature_map = self.features(x)
        pooled = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        return self.classifier(pooled)


class SpatialCBL(nn.Module):
    def __init__(self, feature_dim: int, num_concepts: int, grid_size: int):
        super().__init__()
        self.grid_size = grid_size
        self.proj = nn.Conv2d(feature_dim, num_concepts, kernel_size=1, bias=False)

    def forward(self, feature_map):
        feature_map = F.interpolate(
            feature_map,
            size=(self.grid_size, self.grid_size),
            mode="bilinear",
            align_corners=False,
        )
        return self.proj(feature_map)


class SALFCBM(nn.Module):
    def __init__(
        self,
        backbone: ResNet18Spatial,
        cbl: SpatialCBL,
        classifier: nn.Linear,
        concept_mean: torch.Tensor,
        concept_std: torch.Tensor,
    ):
        super().__init__()
        self.backbone = backbone
        self.cbl = cbl
        self.classifier = classifier
        self.register_buffer("concept_mean", concept_mean)
        self.register_buffer("concept_std", concept_std)

    @staticmethod
    def softmax_pool(maps: torch.Tensor) -> torch.Tensor:
        flat = maps.flatten(2)
        weights = torch.softmax(flat, dim=-1)
        return (weights * flat).sum(dim=-1)

    def forward(self, x):
        feature_map = self.backbone.features(x)
        concept_maps = self.cbl(feature_map)
        concepts = self.softmax_pool(concept_maps)
        normalized = (concepts - self.concept_mean) / self.concept_std.clamp_min(1e-6)
        logits = self.classifier(normalized)
        return logits, normalized, concept_maps


# ---------------------------- utility functions ---------------------------- #

@torch.no_grad()
def accuracy(model, loader, device, full_cbm: bool = False):
    model.eval()
    correct = total = 0
    for x, y, _ in tqdm(loader, desc="evaluate", leave=False):
        x, y = x.to(device), y.to(device)
        output = model(x)
        logits = output[0] if full_cbm else output
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_backbone(checkpoint: Path, device: torch.device) -> ResNet18Spatial:
    model = ResNet18Spatial(imagenet_init=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    return model.to(device)


def load_concepts(path: str | Path) -> list[str]:
    concepts = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not concepts:
        raise ValueError(f"No concepts found in {path}")
    return concepts


# --------------------------- stage 1: backbone ------------------------------ #

def cmd_train_backbone(args):
    seed_everything(args.seed)
    device = device_from_arg(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    train_ds = CUB200(args.data, "train", train_transform(args.image_size))
    test_ds = CUB200(args.data, "test", eval_transform(args.image_size))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = ResNet18Spatial(imagenet_init=True).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y, _ in tqdm(train_loader, desc=f"backbone {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * y.size(0)
        scheduler.step()

        acc = accuracy(model, test_loader, device)
        print(f"epoch={epoch} train_loss={running/len(train_ds):.4f} test_acc={acc:.4%}")
        payload = {"model": model.state_dict(), "epoch": epoch, "test_acc": acc}
        torch.save(payload, output / "backbone_last.pt")
        if acc > best:
            best = acc
            torch.save(payload, output / "backbone_best.pt")
    print(f"Best test accuracy: {best:.4%}")


# --------------------------- stage 2: concepts ------------------------------ #

def humanize_attribute(raw: str) -> str:
    # Example: has_bill_shape::curved_(up_or_down) -> a bird with a curved bill
    raw = raw.strip()
    raw = raw.replace("::", " ")
    raw = raw.replace("_", " ")
    raw = re.sub(r"\([^)]*\)", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = raw.replace("has ", "")
    return f"a bird with {raw}"


def cmd_make_concepts(args):
    root = Path(args.data)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    candidates: list[str] = []
    attributes = root / "attributes" / "attributes.txt"
    if attributes.exists():
        for line in attributes.read_text(encoding="utf-8").splitlines():
            _, raw = line.split(maxsplit=1)
            candidates.append(humanize_attribute(raw))

    # Additional general visual/context concepts. No per-image concept labels are used.
    candidates.extend(
        [
            "a bird head", "a bird eye", "a bird beak", "a bird crown",
            "a bird throat", "a bird breast", "a bird belly", "a bird back",
            "a bird wing", "a bird tail", "bird legs", "bird feet",
            "feathers", "plumage", "a perched bird", "a flying bird",
            "a bird on a branch", "a bird in water", "a bird on grass",
            "tree branches", "green leaves", "blue sky", "water",
            "grass", "rocks", "sand", "snow", "flowers",
        ]
    )

    seen = set()
    concepts = []
    for concept in candidates:
        key = concept.casefold()
        if key not in seen:
            seen.add(key)
            concepts.append(concept)

    if args.limit:
        concepts = concepts[: args.limit]
    output.write_text("\n".join(concepts) + "\n", encoding="utf-8")
    print(f"Wrote {len(concepts)} concepts to {output}")
    print(
        "Note: the paper reports 370 CUB concepts inherited from LF-CBM. "
        "Pass that exact list with --concept-file for strict comparison."
    )


# ------------------------ stage 3: CLIP targets ----------------------------- #

def add_red_circle(image: Image.Image, cx: float, cy: float, radius: int) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    box = (cx - radius, cy - radius, cx + radius, cy + radius)
    width = max(2, radius // 6)
    draw.ellipse(box, outline=(255, 0, 0), width=width)
    return result


@torch.no_grad()
def cmd_clip_targets(args):
    if open_clip is None:
        raise RuntimeError("Install open_clip_torch first.")
    seed_everything(args.seed)
    device = device_from_arg(args.device)
    concepts = load_concepts(args.concept_file)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = CUB200(args.data, "train", transform=None)

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model,
        pretrained=args.clip_pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    clip_model.eval()

    text_tokens = tokenizer(concepts).to(device)
    text_features = F.normalize(clip_model.encode_text(text_tokens), dim=-1)

    positions = [
        ((col + 0.5) / args.grid, (row + 0.5) / args.grid)
        for row in range(args.grid)
        for col in range(args.grid)
    ]

    for image, _, image_id in tqdm(dataset, desc="CLIP spatial targets"):
        save_path = output / f"{image_id:05d}.pt"
        if save_path.exists() and not args.overwrite:
            continue

        width, height = image.size
        prompted = [
            preprocess(
                add_red_circle(
                    image,
                    cx=fx * width,
                    cy=fy * height,
                    radius=args.radius,
                )
            )
            for fx, fy in positions
        ]

        all_features = []
        for start in range(0, len(prompted), args.prompt_batch):
            batch = torch.stack(prompted[start : start + args.prompt_batch]).to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                features = F.normalize(clip_model.encode_image(batch), dim=-1)
            all_features.append(features.float())
        image_features = torch.cat(all_features, dim=0)
        similarities = image_features @ text_features.T  # [G*G, M]
        target = similarities.T.reshape(len(concepts), args.grid, args.grid)
        torch.save(target.to(dtype=torch.float16).cpu(), save_path)

    save_json(
        output / "metadata.json",
        {
            "concept_file": str(Path(args.concept_file).resolve()),
            "num_concepts": len(concepts),
            "grid": args.grid,
            "radius": args.radius,
            "clip_model": args.clip_model,
            "clip_pretrained": args.clip_pretrained,
        },
    )


# ----------------------- stage 4: cache features ---------------------------- #

@torch.no_grad()
def cmd_cache_features(args):
    seed_everything(args.seed)
    device = device_from_arg(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    dataset = CUB200(args.data, "train", eval_transform(args.image_size))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    backbone = load_backbone(Path(args.backbone), device)
    backbone.eval()

    for x, _, image_ids in tqdm(loader, desc="cache backbone features"):
        x = x.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            features = backbone.features(x).float().cpu()
        for feature, image_id in zip(features, image_ids):
            save_path = output / f"{int(image_id):05d}.pt"
            if save_path.exists() and not args.overwrite:
                continue
            torch.save(feature.to(dtype=torch.float16), save_path)


# --------------------------- stage 5: CBL ---------------------------------- #

def cubic_cosine_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    """
    Practical minibatch estimate of Eq. (1).

    For every concept and grid position, correlation is computed across images
    in the minibatch after zero-centering and cubing.
    """
    pred = pred.float()
    target = target.float()
    pred = pred - pred.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    pred = pred.pow(3)
    target = target.pow(3)
    numerator = (pred * target).sum(dim=0)
    denominator = (
        pred.square().sum(dim=0).sqrt()
        * target.square().sum(dim=0).sqrt()
    ).clamp_min(eps)
    similarity = numerator / denominator
    return -similarity.mean()


def cmd_train_cbl(args):
    seed_everything(args.seed)
    device = device_from_arg(args.device)
    concepts = load_concepts(args.concept_file)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    cub = CUB200(args.data, "train", transform=None)
    dataset = PairedCacheDataset(cub, args.features, args.targets)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    cbl = SpatialCBL(512, len(concepts), args.grid).to(device)
    optimizer = torch.optim.AdamW(cbl.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        cbl.train()
        total = 0.0
        count = 0
        for features, targets, _, _ in tqdm(loader, desc=f"CBL {epoch}/{args.epochs}"):
            features = features.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            pred = cbl(features)
            loss = cubic_cosine_loss(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cbl.parameters(), 5.0)
            optimizer.step()
            total += loss.item()
            count += 1
        scheduler.step()
        mean_loss = total / max(count, 1)
        print(f"epoch={epoch} cbl_loss={mean_loss:.6f}")
        payload = {
            "model": cbl.state_dict(),
            "feature_dim": 512,
            "num_concepts": len(concepts),
            "grid": args.grid,
            "epoch": epoch,
            "loss": mean_loss,
        }
        torch.save(payload, output / "cbl_last.pt")
        if mean_loss < best:
            best = mean_loss
            torch.save(payload, output / "cbl_best.pt")

    weights = cbl.proj.weight.detach().cpu()
    torch.save(weights.squeeze(-1).squeeze(-1), output / "W_c.pt")


# ---------------------- stage 6: sparse classifier -------------------------- #

@torch.no_grad()
def compute_all_concepts(
    cbl: SpatialCBL,
    loader: DataLoader,
    device: torch.device,
):
    cbl.eval()
    activations, labels = [], []
    for features, y, _ in tqdm(loader, desc="pool concepts", leave=False):
        features = features.to(device=device, dtype=torch.float32)
        maps = cbl(features)
        pooled = SALFCBM.softmax_pool(maps)
        activations.append(pooled.cpu())
        labels.append(y)
    return torch.cat(activations), torch.cat(labels)


def proximal_l1_(parameter: torch.Tensor, threshold: float):
    with torch.no_grad():
        parameter.copy_(
            torch.sign(parameter) * torch.clamp(parameter.abs() - threshold, min=0.0)
        )


def load_cbl(path: Path, device: torch.device) -> SpatialCBL:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cbl = SpatialCBL(
        payload["feature_dim"],
        payload["num_concepts"],
        payload["grid"],
    )
    cbl.load_state_dict(payload["model"])
    return cbl.to(device)


def cmd_train_classifier(args):
    seed_everything(args.seed)
    device = device_from_arg(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    train_cub = CUB200(args.data, "train", transform=None)
    train_cache = TensorShardDataset(train_cub, args.features)
    train_loader = DataLoader(
        train_cache,
        batch_size=args.feature_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    cbl = load_cbl(Path(args.cbl), device)
    activations, labels = compute_all_concepts(cbl, train_loader, device)
    mean = activations.mean(0)
    std = activations.std(0).clamp_min(1e-6)
    x = (activations - mean) / std

    dataset = torch.utils.data.TensorDataset(x, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    head = nn.Linear(x.shape[1], 200).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        head.train()
        total = 0.0
        correct = count = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(xb)
            ce = F.cross_entropy(logits, yb)
            l2 = 0.5 * head.weight.square().sum()
            loss = ce + args.reg * (1.0 - args.alpha) * l2
            loss.backward()
            optimizer.step()

            # Proximal step approximates the L1 part of elastic net.
            proximal_l1_(head.weight, args.lr * args.reg * args.alpha)

            total += loss.item() * yb.numel()
            correct += (logits.argmax(1) == yb).sum().item()
            count += yb.numel()
        nonzero = (head.weight.detach().abs() > 0).float().mean().item()
        print(
            f"epoch={epoch} loss={total/count:.4f} "
            f"train_acc={correct/count:.4%} nonzero_fraction={nonzero:.4f}"
        )

    torch.save(mean, output / "proj_mean.pt")
    torch.save(std, output / "proj_std.pt")
    torch.save(head.weight.detach().cpu(), output / "W_g.pt")
    torch.save(head.bias.detach().cpu(), output / "b_g.pt")
    torch.save(
        {
            "model": head.state_dict(),
            "num_concepts": x.shape[1],
            "num_classes": 200,
            "reg": args.reg,
            "alpha": args.alpha,
        },
        output / "classifier.pt",
    )


# -------------------------- stage 7: evaluation ----------------------------- #

def cmd_evaluate(args):
    device = device_from_arg(args.device)
    backbone = load_backbone(Path(args.backbone), device)
    cbl = load_cbl(Path(args.cbl), device)

    mean = torch.load(Path(args.model_dir) / "proj_mean.pt", map_location="cpu", weights_only=True)
    std = torch.load(Path(args.model_dir) / "proj_std.pt", map_location="cpu", weights_only=True)
    W = torch.load(Path(args.model_dir) / "W_g.pt", map_location="cpu", weights_only=True)
    b = torch.load(Path(args.model_dir) / "b_g.pt", map_location="cpu", weights_only=True)
    classifier = nn.Linear(W.shape[1], W.shape[0])
    with torch.no_grad():
        classifier.weight.copy_(W)
        classifier.bias.copy_(b)

    model = SALFCBM(backbone, cbl, classifier, mean, std).to(device)
    test_ds = CUB200(args.data, "test", eval_transform(args.image_size))
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    acc = accuracy(model, loader, device, full_cbm=True)
    print(f"SALF-CBM CUB-200 test accuracy: {acc:.4%}")

    # Export files compatible with the released demo model convention.
    export = Path(args.model_dir) / "export"
    export.mkdir(parents=True, exist_ok=True)
    torch.save(cbl.proj.weight.detach().cpu().squeeze(-1).squeeze(-1), export / "W_c.pt")
    torch.save(W, export / "W_g.pt")
    torch.save(b, export / "b_g.pt")
    torch.save(mean, export / "proj_mean.pt")
    torch.save(std, export / "proj_std.pt")
    Path(export / "concepts.txt").write_text(
        Path(args.concept_file).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print(f"Exported model tensors to {export}")


# ---------------------------------- CLI ------------------------------------ #

def build_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--data", required=True, help="Path to CUB_200_2011")
        p.add_argument("--device", default="auto")
        p.add_argument("--workers", type=int, default=2)
        p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("train-backbone")
    common(p)
    p.add_argument("--output", default="runs/cub_backbone")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.set_defaults(func=cmd_train_backbone)

    p = sub.add_parser("make-concepts")
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="concepts/cub_concepts.txt")
    p.add_argument("--limit", type=int, default=370)
    p.set_defaults(func=cmd_make_concepts)

    p = sub.add_parser("clip-targets")
    common(p)
    p.add_argument("--concept-file", required=True)
    p.add_argument("--output", default="cache/clip_targets")
    p.add_argument("--grid", type=int, default=7)
    p.add_argument("--radius", type=int, default=32)
    p.add_argument("--clip-model", default="ViT-B-16")
    p.add_argument("--clip-pretrained", default="openai")
    p.add_argument("--prompt-batch", type=int, default=49)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_clip_targets)

    p = sub.add_parser("cache-features")
    common(p)
    p.add_argument("--backbone", required=True)
    p.add_argument("--output", default="cache/backbone_features")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_cache_features)

    p = sub.add_parser("train-cbl")
    common(p)
    p.add_argument("--features", required=True)
    p.add_argument("--targets", required=True)
    p.add_argument("--concept-file", required=True)
    p.add_argument("--output", default="runs/cub_salf")
    p.add_argument("--grid", type=int, default=7)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.set_defaults(func=cmd_train_cbl)

    p = sub.add_parser("train-classifier")
    common(p)
    p.add_argument("--features", required=True)
    p.add_argument("--cbl", required=True)
    p.add_argument("--output", default="runs/cub_salf")
    p.add_argument("--feature-batch", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--reg", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=0.99)
    p.set_defaults(func=cmd_train_classifier)

    p = sub.add_parser("evaluate")
    common(p)
    p.add_argument("--backbone", required=True)
    p.add_argument("--cbl", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--concept-file", required=True)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=128)
    p.set_defaults(func=cmd_evaluate)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
