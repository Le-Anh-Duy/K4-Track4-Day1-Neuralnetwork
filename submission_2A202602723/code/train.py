"""train.py — đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.

Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).
Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients

N_CLASSES = 7

# Cấu hình mặc định = BASELINE (M-base). `lr` được chọn bằng val trong notebook (Part 2) rồi ghi đè.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    scheduler=None,            # None | "cosine" (ghi vào notes nếu dùng)
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, k: int = N_CLASSES) -> np.ndarray:
    """Ma trận nhầm lẫn (k, k), hàng = nhãn thật, cột = dự đoán (giống scripts/evaluate.py)."""
    cm = torch.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)
    return cm.cpu().numpy()


def per_class_scores(cm: np.ndarray):
    """precision, recall, F1 của từng lớp; bằng 0 khi mẫu số bằng 0."""
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0."""
    return float(per_class_scores(cm)[2].mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Nhãn dự đoán int64 (N,) = argmax của logits, ở chế độ eval()."""
    model.eval()
    return torch.cat([model(X[i:i + batch_size]).argmax(1) for i in range(0, len(X), batch_size)])


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y, như nn.MSELoss: không có hệ số 1/2,
               lấy trung bình trên MỌI phần tử (B * 7). Với reduction="sum" thì trả về tổng
               theo mẫu của trung bình trên 7 lớp, để chia cho N vẫn ra cùng thang đo.
    """
    logits = logits.float()  # loss luôn tính bằng FP32, kể cả dưới autocast
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name == "mse":
        target = F.one_hot(y, N_CLASSES).float()
        per_sample = ((logits - target) ** 2).mean(1)
        return per_sample.mean() if reduction == "mean" else per_sample.sum()
    raise ValueError(f"loss phải là 'ce' hoặc 'mse', nhận {loss_name!r}")


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad, luôn FP32."""
    model.eval()
    total_loss, preds = 0.0, []
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i + batch_size])
        total_loss += compute_loss(logits, y[i:i + batch_size], loss_name, reduction="sum").item()
        preds.append(logits.argmax(1))
    pred = torch.cat(preds)
    cm = confusion_matrix(y, pred)
    return dict(loss=total_loss / len(X), acc=float(np.trace(cm) / cm.sum()), macro_f1=macro_f1_from_confusion(cm))


def _autocast(device_type: str, precision: str):
    if precision == "fp32":
        return torch.autocast(device_type, enabled=False)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    return torch.autocast(device_type, dtype=dtype)


def _finite_or_none(v):
    return v if v is not None and math.isfinite(v) else None


def run_experiment(cfg: dict, data: dict, verbose: bool = True) -> dict:
    """Huấn luyện một cấu hình và trả về {"cfg", "history", "summary", "best_state"}.

    - train_loss đo ở chế độ eval() trên tập con CỐ ĐỊNH 50 000 mẫu của train (data["X_tr_sub"]),
      để so sánh được với val_loss (không bị dropout ảnh hưởng).
    - grad_norm là chuẩn L2 toàn cục TRƯỚC khi clip, trung bình theo epoch (bỏ các bước GradScaler
      phát hiện inf/NaN và tự bỏ qua). Thêm grad_norm_max và clip_frac (tỉ lệ bước bị cắt).
    - best_epoch = epoch có val_loss thấp nhất; val_acc / val_macro_f1 của summary lấy ở epoch đó.
    X_eval KHÔNG được dùng ở đây.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    cfg["hidden"] = tuple(cfg["hidden"])
    if cfg["lr"] is None:
        raise ValueError("cfg['lr'] chưa được đặt")
    X_tr, y_tr, X_val, y_val = data["X_tr"], data["y_tr"], data["X_val"], data["y_val"]
    device = X_tr.device
    dev_type = device.type
    loss_name, precision = cfg["loss"], cfg["precision"]

    set_seed(cfg["seed"])
    model = MLP(hidden=cfg["hidden"], dropout=cfg["dropout"], init=cfg["init"]).to(device)
    if cfg["hidden"] in EXPECTED_PARAMS:
        assert count_params(model) == EXPECTED_PARAMS[cfg["hidden"]], count_params(model)
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"], momentum=cfg["momentum"])
    steps_per_epoch = math.ceil(len(X_tr) / cfg["batch"])
    scheduler = build_scheduler(optimizer, cfg["scheduler"], total_steps=steps_per_epoch * cfg["epochs"])
    scaler = torch.amp.GradScaler(dev_type) if precision == "fp16" else None
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])
    if dev_type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    step0_loss = evaluate(model, X_val, y_val, loss_name)["loss"]
    hist = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
                            "grad_norm", "grad_norm_max", "clip_frac", "lr", "epoch_time_s")}
    best_val, best_epoch, best_state, diverged = math.inf, 0, None, False
    if verbose:
        print(f"[{cfg['exp_id']}] step0 val loss = {step0_loss:.4f}")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        if dev_type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        norms, n_clipped = [], 0
        lr_epoch = optimizer.param_groups[0]["lr"]
        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator=gen):
            with _autocast(dev_type, precision):  # chỉ bọc forward + loss
                logits = model(xb)
                loss = compute_loss(logits, yb, loss_name)
            if not torch.isfinite(loss):
                diverged = True
                break
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # luôn unscale để grad_norm đo trên gradient thật, trước khi clip
            else:
                loss.backward()
            gn = clip_gradients(model.parameters(), cfg["clip_norm"])
            if math.isfinite(gn):
                norms.append(gn)
                n_clipped += cfg["clip_norm"] is not None and gn > cfg["clip_norm"]
            if scaler is not None:
                scaler.step(optimizer)  # tự bỏ qua bước nếu gradient có inf/NaN
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
        if dev_type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0

        tr = evaluate(model, data["X_tr_sub"], data["y_tr_sub"], loss_name)
        va = evaluate(model, X_val, y_val, loss_name)
        if not math.isfinite(va["loss"]):
            diverged = True
        hist["epoch"].append(epoch)
        hist["train_loss"].append(tr["loss"])
        hist["val_loss"].append(va["loss"])
        hist["val_acc"].append(va["acc"])
        hist["val_macro_f1"].append(va["macro_f1"])
        hist["grad_norm"].append(float(np.mean(norms)) if norms else float("nan"))
        hist["grad_norm_max"].append(float(np.max(norms)) if norms else float("nan"))
        hist["clip_frac"].append(n_clipped / max(len(norms), 1))
        hist["lr"].append(lr_epoch)
        hist["epoch_time_s"].append(epoch_time)
        if math.isfinite(va["loss"]) and va["loss"] < best_val:
            best_val, best_epoch = va["loss"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  ep {epoch:2d} | train {tr['loss']:.4f} | val {va['loss']:.4f} acc {va['acc']:.4f} "
                  f"f1 {va['macro_f1']:.4f} | gn {hist['grad_norm'][-1]:.3f} | {epoch_time:.1f}s")
        if diverged:
            print(f"  [{cfg['exp_id']}] loss thành NaN/inf ở epoch {epoch} -> dừng sớm (diverged)")
            break

    if best_state is None:  # phân kỳ ngay epoch đầu: giữ trạng thái hiện tại để không crash
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bi = best_epoch - 1
    summary = dict(
        step0_loss=step0_loss,
        best_val_loss=_finite_or_none(best_val),
        best_epoch=best_epoch,
        final_train_loss=_finite_or_none(hist["train_loss"][-1]) if hist["epoch"] else None,
        final_val_loss=_finite_or_none(hist["val_loss"][-1]) if hist["epoch"] else None,
        val_acc=hist["val_acc"][bi] if best_epoch else None,
        val_macro_f1=hist["val_macro_f1"][bi] if best_epoch else None,
        time_per_epoch_s=float(np.mean(hist["epoch_time_s"])) if hist["epoch"] else None,
        peak_mem_MB=torch.cuda.max_memory_allocated(device) / 2**20 if dev_type == "cuda" else None,
        diverged="Y" if diverged else "N",
    )
    if verbose:
        print(f"[{cfg['exp_id']}] best epoch {best_epoch}: val_loss {summary['best_val_loss']}, "
              f"val_acc {summary['val_acc']}, val_macro_f1 {summary['val_macro_f1']}")
    return {"cfg": cfg, "history": hist, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """CSV `row_id,pred` cho scripts/evaluate.py; đủ mọi dòng eval, mỗi row_id đúng một lần."""
    row_id, preds = np.asarray(row_id), np.asarray(preds)
    assert len(row_id) == len(preds) and len(np.unique(row_id)) == len(row_id)
    assert preds.min() >= 0 and preds.max() <= N_CLASSES - 1
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write("row_id,pred\n")
        f.writelines(f"{r},{p}\n" for r, p in zip(row_id.tolist(), preds.tolist()))


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str,
               repo_root: str | None = None, out_json: str | None = None) -> dict | None:
    """Dùng cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Nếu có repo_root: chạy `python scripts/evaluate.py --pred <pred_path> --out <out_json>` từ repo_root,
    in kết quả và trả về dict đọc từ out_json.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    device = data["X_eval"].device
    model = MLP(hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"], init=cfg["init"]).to(device)
    model.load_state_dict(result["best_state"])
    preds = predict(model, data["X_eval"])  # fp32, eval mode
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    print(f"đã ghi {pred_path} ({len(preds):,} dòng)")
    if repo_root is None:
        return None
    out_json = out_json or os.path.splitext(pred_path)[0] + "_result.json"
    cmd = [sys.executable, "scripts/evaluate.py", "--pred", os.path.abspath(pred_path), "--out", os.path.abspath(out_json)]
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, encoding="utf-8",
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"evaluate.py lỗi:\n{proc.stdout}\n{proc.stderr}")
    with open(out_json) as f:
        return json.load(f)
