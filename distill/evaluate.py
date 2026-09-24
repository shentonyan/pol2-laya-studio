# -*- coding: utf-8 -*-
"""在一个 gold 文件上比较：Laya 基础 checkpoint、蒸馏后的学生、各个老师。

gold 文件可以是 build 产出的 test.jsonl（老师共识），也可以是 annotate import 产出的
human_gold.jsonl（人工真值）。**只有后者能回答「判得对不对」**；前者只回答「学得像不像老师」。
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .common import DATA, answer_to_dist, argmax, expected_score, normalize, now, read_jsonl, sha, clean_questions
from .build import teacher_diagnostics, _fmt
from .label import teacher_file

BASES = {"base:english": ("convaiinnovations/laya", None),
         "base:multilingual": ("convaiinnovations/laya", "multilingual"),
         "base:typed-decisions": ("convaiinnovations/laya", "typed-decisions")}


def predict_model(spec: str, rows: List[dict], batch: int = 16, log=print) -> Dict[str, dict]:
    """返回 {row_id: {qid: 分布}}。spec 是 base:<名字> 或本地 checkpoint 目录。"""
    os.environ.setdefault("USE_TF", "0")
    import laya

    if spec in BASES:
        repo, sub = BASES[spec]
        agent = laya.load(repo, subfolder=sub)
    else:
        agent = laya.Agent(os.path.abspath(spec))
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["questions"]].append(r)
    out = {}
    for qjson, rs in by_q.items():
        qs = json.loads(qjson)
        for i in range(0, len(rs), batch):
            part = rs[i:i + batch]
            res = agent.predict_batch([json.loads(r["state"]) for r in part], qs)
            for r, x in zip(part, res):
                out[r["id"]] = {qid: answer_to_dist(q, x["answers"][qid]) for qid, q in qs.items()}
    log(f"[eval] {spec}: 预测 {len(out)} 条")
    return out


def teacher_preds(name: str, rows: List[dict], labels_dir: Path = DATA / "labels") -> Dict[str, dict]:
    want = {r["id"]: sha(json.loads(r["questions"])) for r in rows}
    out = {}
    for rec in read_jsonl(teacher_file(name, labels_dir)):
        if want.get(rec["state_id"]) == rec["qhash"]:
            out[rec["state_id"]] = rec["dists"]
    return out


def ece(conf: List[float], correct: List[float], bins: int = 15) -> Optional[float]:
    if not conf:
        return None
    e, n = 0.0, len(conf)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [i for i, c in enumerate(conf) if (c >= lo if b == 0 else c > lo) and c <= hi]
        if sel:
            e += len(sel) / n * abs(sum(conf[i] for i in sel) / len(sel) - sum(correct[i] for i in sel) / len(sel))
    return e


def score(rows: List[dict], preds: Dict[str, dict], skip_disputed: bool = False) -> dict:
    agg = defaultdict(lambda: {"n": 0, "acc": 0.0, "brier": 0.0, "conf": [], "corr": [], "mae": [], "missing": 0})
    for r in rows:
        gold = json.loads(r["gold"])
        qs = json.loads(r["questions"])
        disp = r.get("disputed") or {}
        p = preds.get(r["id"])
        for qid, g in gold.items():
            if skip_disputed and qid in disp:
                continue
            for key in ("all", f"type:{qs[qid]['type']}", f"{r['workflow']}.{qid}"):
                a = agg[key]
                if p is None or qid not in p:
                    a["missing"] += 1
                    continue
                keys = list(g["probabilities"])
                pd = normalize(p[qid], keys)
                gd = normalize(g["probabilities"], keys)
                c = float(argmax(pd) == g["label"])
                a["n"] += 1
                a["acc"] += c
                a["brier"] += sum((pd[k] - gd[k]) ** 2 for k in keys)
                a["conf"].append(max(pd.values()))
                a["corr"].append(c)
                if qs[qid]["type"] == "score":
                    a["mae"].append(abs(expected_score(pd) - expected_score(gd)))
    res = {}
    for k, a in agg.items():
        n = a["n"]
        res[k] = {"n": n, "missing": a["missing"], "acc": a["acc"] / n if n else None,
                  "brier": a["brier"] / n if n else None, "ece": ece(a["conf"], a["corr"]),
                  "score_mae": sum(a["mae"]) / len(a["mae"]) if a["mae"] else None}
    return res


def evaluate(gold_path: Path, models: List[str], teachers: List[str], states_path: Optional[Path] = None,
             out_path: Optional[Path] = None, skip_disputed: bool = False, labels_dir: Path = DATA / "labels",
             log=print) -> dict:
    if not Path(gold_path).is_file():
        raise SystemExit(f"找不到 {gold_path}。人工金标要先在 to_label.csv 里填好 human_label，"
                         f"另存后用 annotate import 导入。")
    rows = list(read_jsonl(gold_path))
    if not rows:
        raise SystemExit(f"{gold_path} 是空的：导入的表格里没有填写任何 human_label。")
    systems = {}
    for t in teachers:
        systems[t] = teacher_preds(t, rows, labels_dir)
    for m in models:
        systems[m] = predict_model(m, rows, log=log)
    results = {name: score(rows, p, skip_disputed) for name, p in systems.items()}

    diag = {}
    if states_path:
        states = [s for s in read_jsonl(states_path)]
        ids = {r["id"] for r in rows}
        states = [s for s in states if s["id"] in ids]
        diag = teacher_diagnostics(states, {n: p for n, p in systems.items()}, list(systems), {})

    name = Path(gold_path).name
    if "human" in name:
        note = "gold 是人工标注：这里的准确率回答的是「判得对不对」。"
    elif "design" in name:
        note = ("gold 按构造得到（主张真假 × 回复类型），不来自任何模型或人工判断。"
                "局限：生成模型写错回复类型时，gold 也会跟着错。")
    elif "judge" in name:
        note = ("gold 来自一个没有参与训练标签的独立模型：这里的准确率回答「与独立裁判是否一致」，"
                "仍不等于判得对。")
    else:
        note = "gold 是老师共识：这里的准确率只回答「学得像不像老师」，不代表判得对。"
    L = [f"# 评估报告", "", f"时间：{now()}", f"gold：`{gold_path}`（{len(rows)} 条）", "", "> " + note, ""]
    keys = sorted({k for r in results.values() for k in r}, key=lambda k: (k != "all", not k.startswith("type:"), k))
    for k in keys:
        L += [f"## `{k}`", "", "| 系统 | n | 缺失 | 准确率 | Brier | ECE | score MAE |", "|---|---:|---:|---:|---:|---:|---:|"]
        for name, r in results.items():
            x = r.get(k)
            if not x:
                continue
            L.append(f"| `{name}` | {x['n']} | {x['missing']} | {_fmt(x['acc'])} | {_fmt(x['brier'])} | "
                     f"{_fmt(x['ece'])} | {_fmt(x['score_mae'])} |")
        L.append("")
    if diag:
        L += ["## 不变性诊断", "", "| 系统 | syco 配对 | 归属效应 ΔP(谄媚) | 平均 |Δ| | arena 配对 | A/B 互换一致率 |",
              "|---|---:|---:|---:|---:|---:|"]
        for name, d in diag.items():
            L.append(f"| `{name}` | {d['syco_pairs']} | {_fmt(d['syco_mean_delta'])} | {_fmt(d['syco_mean_abs_delta'])} | "
                     f"{d['arena_pairs']} | {_fmt(d['arena_position_consistency'])} |")
        L.append("")
    out_path = Path(out_path or DATA / "eval" / f"eval_{Path(gold_path).stem}_{now()[:10]}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L), encoding="utf-8")
    out_path.with_suffix(".json").write_text(json.dumps({"results": results, "diagnostics": diag}, ensure_ascii=False,
                                                        indent=1), encoding="utf-8")
    log(f"[eval] 报告 → {out_path}")
    return {"results": results, "diagnostics": diag}
