"""data.py — nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)
N_FEATURES, N_CLASSES = 54, 7
TRAIN_SUB_SIZE = 50_000  # tập con CỐ ĐỊNH của train để đo train_loss mỗi epoch


def _check(X, y):
    assert X.ndim == 2 and X.shape[1] == N_FEATURES and X.dtype == np.float32, (X.shape, X.dtype)
    assert y.shape == (len(X),) and y.dtype == np.int64, (y.shape, y.dtype)
    assert y.min() >= 0 and y.max() <= N_CLASSES - 1


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz. Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id."""
    tr = np.load(f"{processed_dir}/train.npz")
    ev = np.load(f"{processed_dir}/eval.npz")
    X_train, y_train = tr["X"], tr["y"]
    X_eval, y_eval, eval_row_id = ev["X"], ev["y"], ev["row_id"]
    _check(X_train, y_train)
    _check(X_eval, y_eval)
    assert len(eval_row_id) == len(X_eval)
    return X_train, y_train, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval), phân tầng theo nhãn. Trả về: X_tr, y_tr, X_val, y_val."""
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=val_fraction, stratify=y, random_state=seed)
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """mean, std của N_NUMERIC cột đầu, CHỈ tính trên phần train (sau khi tách val).

    Tính trên val/eval là rò rỉ thông tin: thống kê của dữ liệu dùng để đánh giá lọt vào bước tiền xử lý.
    """
    num = X_tr[:, :N_NUMERIC].astype(np.float64)
    return num.mean(0).astype(np.float32), num.std(0).astype(np.float32)


def apply_standardizer(X, mean, std):
    """Bản sao của X, 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên. std = 0 thì chia cho 1."""
    X = X.copy()
    safe_std = np.where(std > 0, std, 1.0).astype(np.float32)
    X[:, :N_NUMERIC] = (X[:, :N_NUMERIC] - mean) / safe_std
    return X


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed", verbose: bool = True) -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm tensor trên device: X_tr, y_tr, X_val, y_val, X_eval, y_eval,
    X_tr_sub, y_tr_sub (tập con cố định 50 000 mẫu của train để đo train_loss),
    và numpy: eval_row_id, mean, std.
    """
    X_full, y_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)
    mean, std = fit_standardizer(X_tr)
    X_tr, X_val, X_eval = (apply_standardizer(a, mean, std) for a in (X_tr, X_val, X_eval))

    sub_idx = np.random.default_rng(seed).choice(len(X_tr), size=min(TRAIN_SUB_SIZE, len(X_tr)), replace=False)

    def fx(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    def fy(a):
        return torch.tensor(a, dtype=torch.int64, device=device)

    data = dict(
        X_tr=fx(X_tr), y_tr=fy(y_tr), X_val=fx(X_val), y_val=fy(y_val),
        X_eval=fx(X_eval), y_eval=fy(y_eval),
        X_tr_sub=fx(X_tr[sub_idx]), y_tr_sub=fy(y_tr[sub_idx]),
        eval_row_id=eval_row_id, mean=mean, std=std,
    )
    if verbose:
        majority = np.bincount(y_tr, minlength=N_CLASSES).argmax()
        print(f"train full: {len(X_full):,} -> X_tr {X_tr.shape}, X_val {X_val.shape}; X_eval {X_eval.shape}")
        print(f"'luôn đoán lớp đa số' (lớp {majority}) trên val: acc = {(y_val == majority).mean():.4f}")
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Batch cuối nhỏ hơn batch_size vẫn được dùng (không bỏ), nên mỗi epoch duyệt đúng mọi mẫu một lần.
    `generator` phải nằm cùng thiết bị với X (torch.randperm yêu cầu vậy).
    """
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
