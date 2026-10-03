"""runner.py — đọc experiments.yaml và chạy toàn bộ kế hoạch thí nghiệm một lượt.

Mỗi thí nghiệm vẫn chỉ là một dict cfg đưa vào train.run_experiment; file này lo phần "điều phối":
giải biểu thức (=BASE_LR*10, =CLIP_C), lưu JSON/ảnh/checkpoint, nạp lại kết quả cũ, chọn cấu hình cuối bằng val.
"""
from __future__ import annotations

import fnmatch
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from plots import plot_compare, plot_run
from results_table import save_result
from train import DEFAULT_CFG, run_experiment

CFG_KEYS = set(DEFAULT_CFG)


def load_plan(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm_cfg(cfg: dict) -> dict:
    c = {**DEFAULT_CFG, **cfg}
    c["hidden"] = list(c["hidden"])
    return c


class Runner:
    def __init__(self, plan: dict, data: dict, fig_dir: str, res_dir: str, ckpt_dir, skip_existing: bool = True,
                 epochs_override: int | None = None):
        self.plan, self.data = plan, data
        self.fig_dir, self.res_dir, self.ckpt_dir = fig_dir, res_dir, Path(ckpt_dir)
        self.skip_existing, self.epochs_override = skip_existing, epochs_override
        self.results: dict[str, dict] = {}   # exp_id -> result, theo thứ tự chạy (có best_state)
        self.ctx: dict[str, float] = {}      # BASE_LR, CLIP_C
        self.noise: dict[str, float] = {}    # mean, std, two_sigma của val_macro_f1 baseline
        self.skipped: dict[str, str] = {}    # exp_id -> lý do bỏ qua
        self.base_ids: list[str] = []
        self.final_ids: list[str] = []
        self.final_desc = ""

    # ---------- tiện ích ----------
    def f1(self, eid: str) -> float:
        v = self.results[eid]["summary"]["val_macro_f1"]
        return -1.0 if v is None else v

    def match(self, patterns) -> list[str]:
        """Mẫu exp_id (wildcard) -> danh sách exp_id đã chạy (giữ thứ tự chạy, không lặp); "best:<mẫu>" -> exp_id tốt nhất."""
        out = []
        for p in patterns:
            if p.startswith("best:"):
                cands = [e for e in self.results if fnmatch.fnmatchcase(e, p[5:])]
                hits = [max(cands, key=self.f1)] if cands else []
            else:
                hits = [e for e in self.results if fnmatch.fnmatchcase(e, p)]
            out += [e for e in hits if e not in out]
        return out

    def _resolve(self, v):
        if isinstance(v, str) and v.startswith("="):
            return float(eval(v[1:], {"__builtins__": {}}, dict(self.ctx)))  # chỉ các biến BASE_LR, CLIP_C
        return v

    def make_cfg(self, spec: dict) -> dict:
        unknown = set(spec) - CFG_KEYS - {"requires"}
        if unknown:
            raise ValueError(f"{spec.get('exp_id')}: khoá không hợp lệ {unknown}")
        cfg = {**DEFAULT_CFG, **self.plan["baseline"], "lr": self.ctx.get("BASE_LR")}
        cfg.update({k: self._resolve(v) for k, v in spec.items() if k != "requires"})
        if self.epochs_override:
            cfg["epochs"] = self.epochs_override
        cfg["hidden"] = tuple(cfg["hidden"])
        return cfg

    # ---------- chạy một thí nghiệm ----------
    def run(self, cfg: dict, verbose: bool = False) -> dict:
        eid = cfg["exp_id"]
        jpath, ckpt = Path(self.res_dir) / f"{eid}.json", self.ckpt_dir / f"{eid}.pt"
        if self.skip_existing and jpath.exists() and ckpt.exists():
            old = json.loads(jpath.read_text(encoding="utf-8"))
            if _norm_cfg(old["cfg"]) == _norm_cfg(cfg):
                old["cfg"]["hidden"] = tuple(old["cfg"]["hidden"])
                old["best_state"] = torch.load(ckpt, map_location="cpu")
                self.results[eid] = old
                plot_run(old, f"{self.fig_dir}/{eid}.png")
                self._log(eid, old, "nạp lại")
                return old
        t0 = time.time()
        r = run_experiment(cfg, self.data, verbose=verbose)
        save_result(r, self.res_dir)
        torch.save(r["best_state"], ckpt)
        plot_run(r, f"{self.fig_dir}/{eid}.png")
        self.results[eid] = r
        self._log(eid, r, f"{time.time() - t0:5.1f}s")
        return r

    @staticmethod
    def _log(eid, r, tag):
        s = r["summary"]
        fmt = lambda v: "—" if v is None else f"{v:.4f}"
        print(f"[{eid}] {tag} | step0 {s['step0_loss']:.3f} | best ep {s['best_epoch']:2d} | val_loss {fmt(s['best_val_loss'])} "
              f"| val_acc {fmt(s['val_acc'])} | val_F1 {fmt(s['val_macro_f1'])} | diverged {s['diverged']}")

    # ---------- kế hoạch đầy đủ ----------
    def run_all(self, device: str) -> None:
        plan, t_all = self.plan, time.time()

        print("=== 1. Tìm lr baseline (SGD+momentum) bằng val ===")
        lr_ids = []
        for lr in plan["lr_search"]["lrs"]:
            eid = f"lr-sgdm-{lr:g}"
            self.run(self.make_cfg({"exp_id": eid, "group": "hparam", "lr": lr, "seed": 1,
                                    "description": f"Tìm lr baseline: SGD+momentum lr={lr:g}"}))
            lr_ids.append(eid)
        best_lr_id = max(lr_ids, key=self.f1)
        self.ctx["BASE_LR"] = self.results[best_lr_id]["cfg"]["lr"]
        print(f"=> BASE_LR = {self.ctx['BASE_LR']:g} (từ {best_lr_id})")

        print("=== 2. Baseline nhiều seed -> độ nhiễu ===")
        for s in plan["base_seeds"]:
            eid = f"base-s{s}"
            self.run(self.make_cfg({"exp_id": eid, "group": "baseline", "seed": s,
                                    "description": f"Baseline M-base, SGD+momentum lr={self.ctx['BASE_LR']:g}, seed {s}"}))
            self.base_ids.append(eid)
        f1s = np.array([self.f1(e) for e in self.base_ids])
        std = float(f1s.std(ddof=1)) if len(f1s) > 1 else float("nan")
        self.noise = {"mean": float(f1s.mean()), "std": std, "two_sigma": 2 * std}
        gn = np.array(self.results[self.base_ids[0]]["history"]["grad_norm"])
        self.ctx["CLIP_C"] = float(f"{np.nanmedian(gn):.2g}")
        print(f"=> val_macro_f1 baseline = {self.noise['mean']:.4f} ± {std:.4f}, 2σ = {self.noise['two_sigma']:.4f}; "
              f"CLIP_C = {self.ctx['CLIP_C']:g}")

        print("=== 3. Các thí nghiệm trong experiments.yaml ===")
        for spec in plan["experiments"]:
            req = spec.get("requires")
            if req == "cuda" and device != "cuda":
                self.skipped[spec["exp_id"]] = "không có GPU CUDA"
            elif req == "bf16" and not (device == "cuda" and torch.cuda.is_bf16_supported()):
                self.skipped[spec["exp_id"]] = "GPU không hỗ trợ BF16"
            if spec["exp_id"] in self.skipped:
                print(f"[{spec['exp_id']}] BỎ QUA: {self.skipped[spec['exp_id']]}")
                continue
            self.run(self.make_cfg(spec))

        print("=== 4. Cấu hình cuối (chọn bằng val) ===")
        self.run_final()
        self.make_compare_figures()
        print(f"Tổng thời gian: {(time.time() - t_all) / 60:.1f} phút, {len(self.results)} thí nghiệm")

    def beats_base(self, eid: str) -> bool:
        return math.isfinite(self.noise["two_sigma"]) and self.f1(eid) - self.noise["mean"] > self.noise["two_sigma"]

    def run_final(self) -> None:
        fp = self.plan["final"]
        best_opt = max(self.match(fp["optimizer_from"]), key=self.f1)
        oc = self.results[best_opt]["cfg"]
        over = {"optimizer": oc["optimizer"], "lr": oc["lr"], "weight_decay": oc["weight_decay"]}
        choices = [f"optimizer/lr từ {best_opt}"]
        for rule in fp.get("add_if_beats_noise", []):
            cands = self.match(rule["from"])
            if not cands:
                continue
            best = max(cands, key=self.f1)
            if self.beats_base(best):
                over.update({k: self.results[best]["cfg"][k] for k in rule["keys"]})
                choices.append(f"{'/'.join(rule['keys'])} từ {best}")
        epochs = self.epochs_override or fp["epochs"]
        self.final_desc = f"Cấu hình cuối ({epochs} epoch): " + "; ".join(choices)
        print(self.final_desc)
        for s in fp["seeds"]:
            eid = f"final-s{s}"
            cfg = self.make_cfg({"exp_id": eid, "group": "final", "seed": s, **over,
                                 "description": f"{self.final_desc}; seed {s}"})
            cfg["epochs"] = epochs
            self.run(cfg)
            self.final_ids.append(eid)

    def make_compare_figures(self) -> dict[str, str]:
        paths = {}
        for name, spec in self.plan.get("compare", {}).items():
            ids = self.match(spec["ids"])
            if len(ids) < 2:
                continue
            paths[name] = f"{self.fig_dir}/compare_{name}.png"
            plot_compare([self.results[e] for e in ids], spec["metrics"], paths[name], f"compare_{name}: " + ", ".join(ids))
        return paths

    # ---------- bảng hiển thị ----------
    def table(self, patterns) -> pd.DataFrame:
        """Bảng tóm tắt + chênh lệch val_macro_f1 so với trung bình baseline và ngưỡng 2σ."""
        rows = []
        for e in self.match(patterns):
            c, s = self.results[e]["cfg"], self.results[e]["summary"]
            row = dict(exp_id=e, optimizer=c["optimizer"], lr=c["lr"], wd=c["weight_decay"], batch=c["batch"],
                       hidden="-".join(map(str, c["hidden"])), dropout=c["dropout"], clip=c["clip_norm"],
                       prec=c["precision"], init=c["init"], loss=c["loss"], sched=c.get("scheduler"),
                       step0=s["step0_loss"], best_ep=s["best_epoch"], best_val_loss=s["best_val_loss"],
                       final_train=s["final_train_loss"], final_val=s["final_val_loss"],
                       val_acc=s["val_acc"], val_F1=s["val_macro_f1"], s_per_ep=s["time_per_epoch_s"],
                       mem_MB=s["peak_mem_MB"], diverged=s["diverged"])
            if self.noise and s["val_macro_f1"] is not None and math.isfinite(self.noise["two_sigma"]):
                row["ΔF1 vs base"] = s["val_macro_f1"] - self.noise["mean"]
                row["vượt 2σ?"] = "Có" if abs(row["ΔF1 vs base"]) > self.noise["two_sigma"] else "Không"
            rows.append(row)
        return pd.DataFrame(rows).set_index("exp_id") if rows else pd.DataFrame()
