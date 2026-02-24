import os
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm

from .model import MiniUNet


@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    epochs: int = 10
    batch: int = 8
    lr: float = 3e-4
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    in_ch: int = 1
    base: int = 32


class SegDataset(Dataset):
    def __init__(self, data_dir: str, split: str):
        self.img_dir = os.path.join(data_dir, split, "images")
        self.mask_dir = os.path.join(data_dir, split, "masks")
        self.ids = sorted([p[:-4] for p in os.listdir(self.img_dir) if p.endswith(".png")])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        img = Image.open(os.path.join(self.img_dir, f"{sid}.png")).convert("L")
        mask = Image.open(os.path.join(self.mask_dir, f"{sid}.png")).convert("L")
        x = TF.to_tensor(img)
        y = torch.from_numpy(np.array(mask, dtype=np.int64))
        return x, y


def _load_labels(data_dir: str) -> List[str]:
    p = os.path.join(data_dir, "dataset.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            spec = json.load(f)
        return spec["labels"]
    return ["background", "header", "experience", "education", "skills", "contacts"]


@torch.no_grad()
def _eval(model, loader, device: str, num_classes: int) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    inter = torch.zeros(num_classes, dtype=torch.float64, device=device)
    union = torch.zeros(num_classes, dtype=torch.float64, device=device)
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        total += y.numel()
        correct += (pred == y).sum().item()
        for c in range(num_classes):
            pc = pred == c
            yc = y == c
            iou_i = (pc & yc).sum()
            iou_u = (pc | yc).sum()
            inter[c] += iou_i
            union[c] += iou_u
    acc = correct / max(1, total)
    iou = (inter / torch.clamp(union, min=1.0)).detach().cpu().numpy()
    miou = float(np.mean(iou[1:])) if num_classes > 1 else float(iou.mean())
    return {"acc": float(acc), "miou": float(miou)}


def train(cfg: TrainConfig) -> Dict:
    os.makedirs(cfg.out_dir, exist_ok=True)

    labels = _load_labels(cfg.data_dir)
    num_classes = len(labels)

    tr = SegDataset(cfg.data_dir, "train")
    va = SegDataset(cfg.data_dir, "val")
    tr_loader = DataLoader(tr, batch_size=cfg.batch, shuffle=True, num_workers=cfg.num_workers)
    va_loader = DataLoader(va, batch_size=cfg.batch, shuffle=False, num_workers=cfg.num_workers)

    model = MiniUNet(in_ch=cfg.in_ch, num_classes=num_classes, base=cfg.base).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    history = {"labels": labels, "epochs": []}

    best = -1.0
    best_path = os.path.join(cfg.out_dir, "model.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(tr_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False)
        run_loss = 0.0
        n_batches = 0

        for x, y in pbar:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            run_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=run_loss / max(1, n_batches))

        metrics = _eval(model, va_loader, cfg.device, num_classes)
        row = {"epoch": epoch, "train_loss": run_loss / max(1, n_batches), **metrics}
        history["epochs"].append(row)

        if metrics["miou"] > best:
            best = metrics["miou"]
            torch.save(
                {"state_dict": model.state_dict(), "labels": labels, "in_ch": cfg.in_ch, "base": cfg.base},
                best_path,
            )

        with open(os.path.join(cfg.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)

    return history