import os
import json
from typing import List, Dict, Tuple

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision.transforms import functional as TF

from .model import MiniUNet


def load_model(path: str, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=device)
    labels = ckpt["labels"]
    in_ch = ckpt.get("in_ch", 1)
    base = ckpt.get("base", 32)
    model = MiniUNet(in_ch=in_ch, num_classes=len(labels), base=base).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, labels, device


@torch.no_grad()
def predict_mask(model, img: Image.Image, device: str) -> Tuple[np.ndarray, np.ndarray]:
    x = TF.to_tensor(img.convert("L")).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred = probs.argmax(axis=0).astype(np.uint8)
    conf = probs.max(axis=0).astype(np.float32)
    return pred, conf


def mask_to_boxes(pred: np.ndarray, conf: np.ndarray, labels: List[str], min_area: int = 900) -> List[Dict]:
    out = []
    for c in range(1, len(labels)):
        m = (pred == c).astype(np.uint8)
        if m.sum() < min_area:
            continue
        n, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            score = float(conf[y1:y2, x1:x2].mean()) if (y2 > y1 and x2 > x1) else 0.0
            out.append({"label": labels[c], "x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": score})
    out.sort(key=lambda d: (d["y1"], d["x1"]))
    return out


def render_vis(img: Image.Image, pred: np.ndarray, boxes: List[Dict], out_path: str):
    base = np.array(img.convert("RGB"))
    m = pred.astype(np.uint8)
    color = cv2.applyColorMap((m * (255 // max(1, m.max()))).astype(np.uint8), cv2.COLORMAP_JET)
    vis = cv2.addWeighted(base[..., ::-1], 0.75, color, 0.25, 0.0)
    for b in boxes:
        cv2.rectangle(vis, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 0, 0), 2)
        txt = f'{b["label"]}:{b["score"]:.2f}'
        cv2.putText(vis, txt, (b["x1"], max(0, b["y1"] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    Image.fromarray(vis[..., ::-1]).save(out_path)


def infer_image(model_path: str, image_path: str, out_dir: str, min_area: int = 900) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    model, labels, device = load_model(model_path)
    img = Image.open(image_path).convert("L")
    pred, conf = predict_mask(model, img, device)
    boxes = mask_to_boxes(pred, conf, labels, min_area=min_area)

    payload = {"width": img.size[0], "height": img.size[1], "labels": labels, "sections": boxes}
    with open(os.path.join(out_dir, "sections.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    render_vis(img, pred, boxes, os.path.join(out_dir, "vis.png"))
    Image.fromarray(pred).save(os.path.join(out_dir, "pred_mask.png"))
    return payload