# -*- coding: utf-8 -*-
"""单卡 RLCD 微调 Laya（改写自官方 Kaggle 2×T4 DDP notebook，去掉分布式，适配 Windows 单显卡）。

训练目标与官方一致：严格适当评分规则奖励（log + spherical + 有序题的 RPS）上的 GRPO 式策略梯度，
加软交叉熵引导；训练结束后在按 group 留出的校准集上按题型拟合温度。

需要：torch（CUDA 版）、laya、transformers、safetensors、huggingface_hub。
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

from .common import DATA, now, read_jsonl, sha

os.environ.setdefault("USE_TF", "0")


def resolve_base(base: str, subfolder: Optional[str]) -> str:
    """本地目录直接用；否则只下载所需子目录。返回包含 rl_agent_config.json 的目录。"""
    from laya.agent import _fix_tokenizer_config

    if os.path.isdir(base):
        d = os.path.join(base, subfolder) if subfolder else base
    else:
        from huggingface_hub import snapshot_download

        prefix = f"{subfolder}/" if subfolder else ""
        root = snapshot_download(base, allow_patterns=[prefix + n for n in (
            "rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")])
        d = os.path.join(root, subfolder) if subfolder else root
    if not os.path.isfile(os.path.join(d, "rl_agent_config.json")):
        raise FileNotFoundError(f"{d} 里没有 rl_agent_config.json，不是 Laya checkpoint")
    _fix_tokenizer_config(d)
    return d


def build_items(rows: List[dict], tok, cfg: dict) -> List[dict]:
    """与官方 notebook 的 build_training_item 相同，额外保留 group_id 以便按组留出校准集。"""
    from laya.common import QTYPES, build_sequence, render_options

    items, skipped = [], 0
    for row in rows:
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        for qid, q in questions.items():
            g = gold.get(qid)
            if g is None:
                continue
            t, crit = q["type"], q.get("criteria") or {}
            if t == "choice":
                target = [g["probabilities"].get(k, 0.0) for k in crit.keys()]
            elif t == "noul":
                target = [g["probabilities"].get("false", 0.5), g["probabilities"].get("true", 0.5)]
            else:
                target = [g["probabilities"].get(str(i), 0.0) for i in range(len(crit))]
            s = sum(target)
            target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
            internal = {"t": t, "ins": q["instructions"], "crit": crit}
            seq, markers = build_sequence(tok, state, internal, cfg["max_len"], cfg["head_max_len"])
            if len(markers) != len(render_options(internal)):
                skipped += 1
                continue
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[t], "target": target,
                          "label": target.index(max(target)), "group": row.get("group_id", row["id"]),
                          "workflow": row.get("workflow")})
    if skipped:
        print(f"[train] 跳过 {skipped} 题（选项超出 head_max_len）")
    return items


def collate(items, pad_id):
    import torch

    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
        att[i, :len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, :len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask, "target": target,
            "qtype": torch.tensor([it["qtype"] for it in items])}


def fit_one_temp(sel) -> float:
    import torch

    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.5, 5.0).item())  # 与 laya 运行时的 clamp 区间一致


def train(train_path: Path = DATA / "dataset" / "train.jsonl", out_dir: Optional[Path] = None,
          base: str = "convaiinnovations/laya", subfolder: Optional[str] = "multilingual", epochs: int = 4,
          micro_batch: int = 8, grad_accum: int = 8, lr_encoder: float = 2.5e-5, lr_head: float = 1e-4,
          group_size: int = 4, sigma_start: float = 0.4, sigma_end: float = 0.1, ce_weight: float = 1.0,
          calib_frac: float = 0.1, seed: int = 20260924, grad_ckpt: bool = True, max_len: Optional[int] = None,
          log=print) -> Path:
    import torch
    from safetensors.torch import load_file, save_file
    from laya.agent import _load_tokenizer
    from laya.common import build_model, proper_reward

    if not torch.cuda.is_available():
        log("[train] 警告：没有检测到 CUDA，将在 CPU 上训练（非常慢，只适合冒烟测试）。")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_amp = device.type == "cuda"

    random.seed(seed)
    torch.manual_seed(seed)
    model_dir = resolve_base(base, subfolder)
    with open(os.path.join(model_dir, "rl_agent_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if max_len:
        cfg["max_len"] = max_len
    cfg.setdefault("max_len", 512)
    cfg.setdefault("head_max_len", 192)
    tok = _load_tokenizer(os.path.join(model_dir, "tokenizer"), cfg)

    rows = list(read_jsonl(train_path))
    if not rows:
        raise ValueError(f"{train_path} 是空的。先运行 build。")
    items = build_items(rows, tok, cfg)
    # 按 group 留出校准集：同一组（配对/孪生样本）不能一半训练一半校准
    groups = sorted({it["group"] for it in items})
    random.Random(seed).shuffle(groups)
    calib_groups = set(groups[:max(1, int(len(groups) * calib_frac))])
    calib = [it for it in items if it["group"] in calib_groups]
    tr = [it for it in items if it["group"] not in calib_groups]
    log(f"[train] base={base}{'/' + subfolder if subfolder else ''} device={device} amp={amp_dtype if use_amp else 'off'}")
    log(f"[train] {len(rows)} 行 → {len(tr)} 训练题，{len(calib)} 校准题；max_len={cfg['max_len']} head_max_len={cfg['head_max_len']}")

    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"), pretrained=False)
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    try:
        model.encoder.config.reference_compile = False
    except Exception:  # noqa: BLE001
        pass
    if grad_ckpt:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model.to(device).train()

    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": lr_encoder}, {"params": head, "lr": lr_head}], weight_decay=0.01)
    steps = max(1, math.ceil(len(tr) / micro_batch / grad_accum) * epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not use_bf16)

    out_dir = Path(out_dir or DATA / "models" / f"laya-pol2-{time.strftime('%Y%m%d-%H%M')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for ep in range(epochs):
        random.Random(seed + ep).shuffle(tr)
        sigma = sigma_start + (sigma_end - sigma_start) * (ep / max(1, epochs - 1))
        tot, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for bi in range(0, len(tr), micro_batch):
            b = collate(tr[bi:bi + micro_batch], tok.pad_token_id)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits, act = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                                    b["marker_pos"].to(device), b["marker_mask"].to(device), b["qtype"].to(device))
            logits = logits.float()
            mask = b["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = b["target"].to(device)
            eps = torch.randn((group_size,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), b["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = (r - r.mean(0, keepdim=True)) / ((r - r.mean(0, keepdim=True)).std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + ce_weight * loss_ce) / grad_accum + 0.0 * act.sum()
            scaler.scale(loss).backward()
            nb += 1
            if nb % grad_accum == 0 or bi + micro_batch >= len(tr):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            tot += loss.item() * grad_accum
            if nb % 50 == 0:
                log(f"[train] epoch {ep + 1}/{epochs} step {nb} loss {loss.item() * grad_accum:.4f} "
                    f"reward {r.mean().item():.3f} lr {sched.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)")
        log(f"[train] === epoch {ep + 1}/{epochs} 完成，平均 loss {tot / max(1, nb):.4f}，用时 {time.time() - t0:.0f}s")
        _save(model, tok, cfg, out_dir / "checkpoint_latest", None)

    # 温度校准
    model.eval()
    preds = []
    with torch.no_grad():
        for ci in range(0, len(calib), 16):
            ch = calib[ci:ci + 16]
            cb = collate(ch, tok.pad_token_id)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                lg, _ = model(cb["input_ids"].to(device), cb["attention_mask"].to(device), cb["marker_pos"].to(device),
                              cb["marker_mask"].to(device), cb["qtype"].to(device))
            lg = lg.float().cpu().numpy()
            for j, it in enumerate(ch):
                preds.append((it["qtype"], lg[j, :len(it["markers"])].tolist(), it["target"]))
    temps = [fit_one_temp([(z, t) for qt, z, t in preds if qt == i]) for i in range(3)]
    log(f"[train] 拟合温度 (choice, score, noul) = {[round(t, 3) for t in temps]}")
    cfg.update(fine_tuned=True, model_name="laya-pol2-distilled", temperature=temps)
    cfg.pop("temperature_by_options", None)
    meta = {"created": now(), "base": base, "subfolder": subfolder, "train_file": str(train_path),
            "train_sha": sha([r["id"] for r in rows]), "n_rows": len(rows), "n_train_items": len(tr),
            "n_calib_items": len(calib), "epochs": epochs, "micro_batch": micro_batch, "grad_accum": grad_accum,
            "lr_encoder": lr_encoder, "lr_head": lr_head, "temperatures": temps,
            "honesty": "学生模型蒸馏自老师共识；上限是老师，偏差会一起学走。须用人工金标集评估。"}
    _save(model, tok, cfg, out_dir, meta)
    log(f"[train] 完成 → {out_dir}")
    return out_dir


def _save(model, tok, cfg, d: Path, meta: Optional[dict]):
    from safetensors.torch import save_file

    d.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, str(d / "model.safetensors"))
    model.encoder.config.save_pretrained(str(d / "encoder"))
    tok.save_pretrained(str(d / "tokenizer"))
    (d / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    if meta:
        (d / "distill_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
