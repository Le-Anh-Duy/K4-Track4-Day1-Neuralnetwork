"""plots.py — ảnh biểu đồ (sản phẩm nộp, README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png,
và ảnh chồng figures/compare_<nhóm>.png cho từng nhóm.
"""
from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt

LABELS = {
    "train_loss": "train loss (eval mode)", "val_loss": "val loss", "val_acc": "val accuracy",
    "val_macro_f1": "val macro-F1", "grad_norm": "grad norm (trước clip, TB epoch)",
}


def _cfg_text(cfg: dict) -> str:
    hidden = "-".join(map(str, cfg["hidden"]))
    parts = [f"{cfg['loss']}", f"{cfg['optimizer']} lr={cfg['lr']:g}", f"bs={cfg['batch']}", f"h={hidden}",
             f"drop={cfg['dropout']}", f"init={cfg['init']}", f"clip={cfg['clip_norm']}", cfg["precision"],
             f"seed={cfg['seed']}"]
    if cfg.get("weight_decay"):
        parts.insert(2, f"wd={cfg['weight_decay']:g}")
    if cfg.get("scheduler"):
        parts.insert(2, f"sched={cfg['scheduler']}")
    return ", ".join(parts)


def plot_run(result: dict, path: str) -> None:
    """Một thí nghiệm -> một PNG 3 ô: (1) train/val loss, (2) val acc + macro-F1, (3) grad_norm trước clip.
    Đường đứt nét dọc = best_epoch (val loss thấp nhất).
    """
    cfg, h, s = result["cfg"], result["history"], result["summary"]
    ep = h["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    ax = axes[0]
    ax.plot(ep, h["train_loss"], "o-", ms=3, label=LABELS["train_loss"])
    ax.plot(ep, h["val_loss"], "o-", ms=3, label=LABELS["val_loss"])
    ax.axhline(s["step0_loss"], color="gray", ls=":", lw=1, label=f"step0 val loss = {s['step0_loss']:.3f}")
    ax.set(xlabel="epoch", ylabel=f"loss ({cfg['loss']})", title="Loss")

    ax = axes[1]
    ax.plot(ep, h["val_acc"], "o-", ms=3, label=LABELS["val_acc"])
    ax.plot(ep, h["val_macro_f1"], "o-", ms=3, label=LABELS["val_macro_f1"])
    ax.set(xlabel="epoch", ylabel="score", title="Val accuracy / macro-F1")

    ax = axes[2]
    ax.plot(ep, h["grad_norm"], "o-", ms=3, label="TB epoch")
    if "grad_norm_max" in h:
        ax.plot(ep, h["grad_norm_max"], "^--", ms=3, alpha=0.6, label="max epoch")
    if cfg.get("clip_norm") is not None:
        ax.axhline(cfg["clip_norm"], color="red", ls="--", lw=1, label=f"clip c = {cfg['clip_norm']:g}")
    ax.set(xlabel="epoch", ylabel="‖g‖₂", title="Grad norm (trước clip)")
    if any(v and v > 0 for v in h["grad_norm"]):
        ax.set_yscale("log")

    for ax in axes:
        if s["best_epoch"]:
            ax.axvline(s["best_epoch"], color="k", ls="--", lw=0.8, alpha=0.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    best = f"best ep {s['best_epoch']}: val_acc={s['val_acc']:.4f}, val_F1={s['val_macro_f1']:.4f}" \
        if s["best_epoch"] else "diverged"
    fig.suptitle(f"{cfg['exp_id']}  —  {_cfg_text(cfg)}\n{best}" + ("  [DIVERGED]" if s["diverged"] == "Y" else ""),
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric, path: str, title: str = "") -> None:
    """Vẽ chồng một hoặc nhiều chỉ số (str hoặc list[str]) của nhiều thí nghiệm, mỗi thí nghiệm một đường,
    chú thích bằng exp_id. Dùng cho figures/compare_<nhóm>.png.
    """
    metrics = [metric] if isinstance(metric, str) else list(metric)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 4.2), squeeze=False)
    colors = matplotlib.colormaps["tab10"]
    for ax, m in zip(axes[0], metrics):
        for i, r in enumerate(results):
            ax.plot(r["history"]["epoch"], r["history"][m], "o-", ms=2.5, color=colors(i % 10),
                    label=r["cfg"]["exp_id"])
        ax.set(xlabel="epoch", ylabel=LABELS.get(m, m), title=LABELS.get(m, m))
        if m == "grad_norm":
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(title or os.path.splitext(os.path.basename(path))[0])
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
