import os
import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass
class SynthConfig:
    w: int = 768
    h: int = 1086
    margin: int = 48
    gap: int = 18
    min_block_h: int = 90
    max_block_h: int = 360
    p_two_col: float = 0.45
    p_noise: float = 0.6
    p_jpeg: float = 0.35
    p_blur: float = 0.25
    p_rotate: float = 0.25
    max_rotate_deg: float = 1.5


def _rng(seed: int | None) -> random.Random:
    r = random.Random()
    if seed is not None:
        r.seed(seed)
    return r


def _alloc_one_col(r: random.Random, cfg: SynthConfig, labels: List[str]) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    x1 = cfg.margin
    x2 = cfg.w - cfg.margin
    y = cfg.margin
    out = []
    n = r.randint(4, min(8, len(labels)))
    chosen = r.sample(labels, n)
    for lab in chosen:
        if y + cfg.min_block_h >= cfg.h - cfg.margin:
            break
        bh = r.randint(cfg.min_block_h, cfg.max_block_h)
        bh = min(bh, (cfg.h - cfg.margin) - y)
        b = (x1, y, x2, y + bh)
        out.append((lab, b))
        y = y + bh + cfg.gap
    return out


def _alloc_two_col(r: random.Random, cfg: SynthConfig, labels: List[str]) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    m = cfg.margin
    gap = cfg.gap
    col_gap = 28
    col_w = (cfg.w - 2 * m - col_gap)
    left_x1 = m
    left_x2 = m + col_w // 2
    right_x1 = left_x2 + col_gap
    right_x2 = cfg.w - m

    yL = m
    yR = m
    out = []

    header = None
    if "header" in labels and r.random() < 0.9:
        header = ("header", (m, m, cfg.w - m, m + r.randint(90, 170)))
        out.append(header)
        yL = header[1][3] + gap
        yR = header[1][3] + gap

    labs = [x for x in labels if x != "header"]
    r.shuffle(labs)
    for lab in labs:
        side = "L" if r.random() < 0.55 else "R"
        if side == "L":
            if yL + cfg.min_block_h >= cfg.h - m:
                continue
            bh = r.randint(cfg.min_block_h, cfg.max_block_h)
            bh = min(bh, (cfg.h - m) - yL)
            out.append((lab, (left_x1, yL, left_x2, yL + bh)))
            yL = yL + bh + gap
        else:
            if yR + cfg.min_block_h >= cfg.h - m:
                continue
            bh = r.randint(cfg.min_block_h, cfg.max_block_h)
            bh = min(bh, (cfg.h - m) - yR)
            out.append((lab, (right_x1, yR, right_x2, yR + bh)))
            yR = yR + bh + gap

    return out


def _draw_text_like(r: random.Random, draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    pad = r.randint(10, 16)
    tx1 = x1 + pad
    tx2 = x2 - pad
    y = y1 + pad
    line_h = r.randint(10, 14)
    max_lines = max(1, (y2 - y1 - 2 * pad) // (line_h + r.randint(3, 6)))
    n_lines = r.randint(max(1, max_lines // 2), max_lines)
    for i in range(n_lines):
        if y + line_h >= y2 - pad:
            break
        w = r.randint(int((tx2 - tx1) * 0.45), int((tx2 - tx1) * 0.98))
        x_end = tx1 + w
        th = r.randint(1, 2)
        if r.random() < 0.18:
            bsz = r.randint(2, 4)
            bx = tx1
            by = y + line_h // 2
            draw.ellipse((bx, by, bx + bsz, by + bsz), fill=0)
            bx2 = bx + bsz + r.randint(8, 12)
            draw.line((bx2, y, x_end, y), fill=0, width=th)
        else:
            draw.line((tx1, y, x_end, y), fill=0, width=th)
        y += line_h + r.randint(4, 7)


def synth_sample(labels: List[str], cfg: SynthConfig, seed: int | None = None) -> Tuple[Image.Image, Image.Image, List[Dict]]:
    r = _rng(seed)
    bg = r.randint(235, 255)
    img = Image.new("L", (cfg.w, cfg.h), color=bg)
    mask = Image.new("L", (cfg.w, cfg.h), color=0)
    draw = ImageDraw.Draw(img)
    mdraw = ImageDraw.Draw(mask)

    if r.random() < cfg.p_two_col:
        blocks = _alloc_two_col(r, cfg, labels)
    else:
        blocks = _alloc_one_col(r, cfg, labels)

    idx = {lab: i + 1 for i, lab in enumerate(labels)}
    meta = []

    for lab, b in blocks:
        x1, y1, x2, y2 = b
        fill = r.randint(245, 255)
        outline = r.randint(120, 170)
        draw.rounded_rectangle(b, radius=r.randint(8, 14), outline=outline, width=1, fill=fill)
        _draw_text_like(r, draw, b)
        mdraw.rectangle(b, fill=idx.get(lab, 0))
        meta.append({"label": lab, "bbox": [x1, y1, x2, y2]})

    if r.random() < cfg.p_noise:
        arr = np.array(img).astype(np.int16)
        noise = r.randint(2, 10)
        n = np.random.randint(-noise, noise + 1, size=arr.shape, dtype=np.int16)
        arr = np.clip(arr + n, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")

    if r.random() < cfg.p_blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=r.uniform(0.2, 0.8)))

    if r.random() < cfg.p_rotate:
        deg = r.uniform(-cfg.max_rotate_deg, cfg.max_rotate_deg)
        img = img.rotate(deg, resample=Image.BILINEAR, expand=False, fillcolor=bg)
        mask = mask.rotate(deg, resample=Image.NEAREST, expand=False, fillcolor=0)

    if r.random() < cfg.p_jpeg:
        from io import BytesIO
        buf = BytesIO()
        q = r.randint(50, 85)
        img.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        img = Image.open(buf).convert("L")

    return img, mask, meta


def _ensure_dirs(base: str, split: str) -> Tuple[str, str, str]:
    img_dir = os.path.join(base, split, "images")
    mask_dir = os.path.join(base, split, "masks")
    meta_dir = os.path.join(base, split, "meta")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    return img_dir, mask_dir, meta_dir


def generate_dataset(out_dir: str, n: int, labels: List[str], cfg: SynthConfig, val_ratio: float = 0.1, seed: int = 1) -> Dict:
    r = _rng(seed)
    n_val = int(round(n * val_ratio))
    n_train = n - n_val

    tr_img, tr_mask, tr_meta = _ensure_dirs(out_dir, "train")
    va_img, va_mask, va_meta = _ensure_dirs(out_dir, "val")

    for i in range(n_train):
        sid = f"{i+1:06d}"
        img, mask, meta = synth_sample(labels, cfg, seed=r.randint(0, 10**9))
        img.save(os.path.join(tr_img, f"{sid}.png"))
        mask.save(os.path.join(tr_mask, f"{sid}.png"))
        with open(os.path.join(tr_meta, f"{sid}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    for i in range(n_val):
        sid = f"{i+1:06d}"
        img, mask, meta = synth_sample(labels, cfg, seed=r.randint(0, 10**9))
        img.save(os.path.join(va_img, f"{sid}.png"))
        mask.save(os.path.join(va_mask, f"{sid}.png"))
        with open(os.path.join(va_meta, f"{sid}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    spec = {
        "w": cfg.w,
        "h": cfg.h,
        "labels": ["background"] + labels,
        "n_train": n_train,
        "n_val": n_val,
        "val_ratio": val_ratio,
        "seed": seed,
    }
    with open(os.path.join(out_dir, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    return spec