# -*- coding: utf-8 -*-
"""共用工具：路径、JSONL、哈希、场景读取、分布的规范化与聚合。只用标准库。"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ROOT / "docs" / "scenarios.json"
DATA = ROOT / "distill_data"

# 发送给模型的问题只允许这三个字段（与 OpenJEV / Laya 的请求格式一致）
QUESTION_KEYS = ("type", "instructions", "criteria")


# ---------------------------------------------------------------- I/O -------
def read_jsonl(path: os.PathLike) -> Iterator[dict]:
    p = Path(path)
    if not p.is_file():
        return
    with p.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{n} 不是合法 JSON：{e}") from None


def write_jsonl(path: os.PathLike, rows: Iterable[dict]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: os.PathLike, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def canon(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(obj: Any, n: int = 16) -> str:
    return hashlib.sha256(canon(obj).encode("utf-8")).hexdigest()[:n]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def load_dotenv(path: os.PathLike = ROOT / ".env") -> None:
    """极简 .env 读取：只设置尚未存在的环境变量，不打印任何值。"""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


# ----------------------------------------------------------- scenarios ------
def load_scenarios(path: os.PathLike = SCENARIOS, group: Optional[str] = "pol2",
                   ids: Optional[List[str]] = None) -> Dict[str, dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for s in d["scenarios"]:
        if group and group != "all" and s.get("group") != group:
            continue
        if ids and s["id"] not in ids:
            continue
        out[s["id"]] = s
    if not out:
        raise ValueError(f"没有匹配的场景（group={group}, ids={ids}）")
    return out


def clean_questions(questions: dict) -> dict:
    """只保留会发给模型的字段；本地注释字段绝不外发。"""
    return {qid: {k: q[k] for k in QUESTION_KEYS if k in q} for qid, q in questions.items()}


def option_keys(q: dict) -> List[str]:
    """规范的选项键顺序：choice 用选项名，score 用 "0".."n-1"，noul 固定 ["false","true"]。"""
    t = q["type"]
    if t == "choice":
        crit = q["criteria"]
        return list(crit.keys()) if isinstance(crit, dict) else [str(c) for c in crit]
    if t == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    if t == "noul":
        return ["false", "true"]
    raise ValueError(f"未知题型 {t!r}")


# -------------------------------------------------------- distributions -----
def normalize(p: Dict[str, float], keys: List[str]) -> Dict[str, float]:
    v = [max(0.0, float(p.get(k, 0.0) or 0.0)) for k in keys]
    s = sum(v)
    if s <= 0 or not math.isfinite(s):
        return {k: 1.0 / len(keys) for k in keys}
    return {k: x / s for k, x in zip(keys, v)}


def smooth(p: Dict[str, float], eps: float) -> Dict[str, float]:
    """(1-eps)·p + eps·均匀。防止某个老师给真实标签 0 概率时把学生逼向极端。"""
    k = len(p)
    return {key: (1 - eps) * v + eps / k for key, v in p.items()}


def answer_to_dist(q: dict, ans: dict) -> Dict[str, float]:
    """把 Jev / Laya 风格的答案转成规范分布。"""
    keys = option_keys(q)
    t = q["type"]
    if t == "noul":
        if "noul" in ans and ans["noul"] is not None:
            p = float(ans["noul"])
        else:
            p = float((ans.get("probabilities") or {}).get("true", 0.5))
        p = min(1.0, max(0.0, p))
        return {"false": 1.0 - p, "true": p}
    probs = ans.get("probabilities") or {}
    if t == "score" and probs and not any(k in probs for k in keys):
        # 个别实现用 legend 文本作键
        crit = q["criteria"]
        probs = {str(crit.index(k)) if k in crit else k: v for k, v in probs.items()}
    if not probs:
        label = ans.get("choice") if t == "choice" else None
        if t == "score" and ans.get("score") is not None:
            label = str(int(round(float(ans["score"]))))
        probs = {label: 1.0} if label is not None else {}
    return normalize(probs, keys)


def argmax(p: Dict[str, float]) -> str:
    return max(p, key=p.get)


def entropy_conf(p: Dict[str, float]) -> float:
    """1 − H(p)/log k，与 Laya/Jev 的 confidence 定义一致。"""
    k = len(p)
    if k < 2:
        return 1.0
    h = -sum(v * math.log(v) for v in p.values() if v > 0)
    return max(0.0, min(1.0, 1.0 - h / math.log(k)))


def js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Jensen–Shannon 散度（以 2 为底，取值 0–1）。"""
    keys = list(p)
    m = {k: 0.5 * (p[k] + q.get(k, 0.0)) for k in keys}

    def kl(a, b):
        return sum(a[k] * math.log2(a[k] / b[k]) for k in keys if a[k] > 0 and b[k] > 0)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def expected_score(p: Dict[str, float]) -> float:
    return sum(int(k) * v for k, v in p.items())


def gold_entry(q: dict, p: Dict[str, float]) -> dict:
    """生成 typed-decisions 数据集 gold 字段中单题的条目。"""
    t = q["type"]
    lab = argmax(p)
    e = {"type": t, "label": lab, "probabilities": {k: round(v, 6) for k, v in p.items()},
         "confidence": round(entropy_conf(p), 6)}
    if t == "score":
        e["score"] = round(expected_score(p), 6)
    if t == "noul":
        e["noul"] = round(p["true"], 6)
    return e
