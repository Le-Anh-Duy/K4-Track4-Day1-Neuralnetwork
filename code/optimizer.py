"""optimizer.py — chọn bộ tối ưu, bộ lập lịch lr và cắt gradient để `train.py` gọn và mọi thí nghiệm công bằng.

Công thức (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import math

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")
SCHEDULERS = (None, "cosine")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer.

    weight_decay của Adam là L2 trộn vào gradient (bị chia cho sqrt(v_hat)), còn của AdamW là suy giảm tách riêng.
    """
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer phải thuộc {OPTIMIZERS}, nhận {name!r}")
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """Bộ lập lịch lr, gọi .step() sau MỖI bước cập nhật. None = lr cố định.

    "cosine": CosineAnnealingLR từ lr ban đầu về eta_min (mặc định 0) sau total_steps bước.
    """
    if name not in SCHEDULERS:
        raise ValueError(f"scheduler phải thuộc {SCHEDULERS}, nhận {name!r}")
    if name is None:
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=kwargs.get("eta_min", 0.0))


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    max_norm = None: chỉ đo (max_norm = inf nên hệ số cắt min(1, inf/‖g‖) = 1, gradient không đổi).
    Khi dùng FP16 + GradScaler: phải scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này.
    """
    total_norm = torch.nn.utils.clip_grad_norm_(params, math.inf if max_norm is None else max_norm)
    return float(total_norm)
