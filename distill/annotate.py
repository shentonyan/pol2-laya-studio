# -*- coding: utf-8 -*-
"""人工金标集：导出盲标表格 → 人工填写 → 导入为 gold。

- 只从测试集（按 group 留出、从未进入训练）里抽样。
- 盲标：表格里**不显示**任何老师或模型的答案，以免人被锚定。
- 可以多人各填一份；导入时合并成投票分布，并报告人与人之间的一致率。
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List

from .common import DATA, gold_entry, normalize, option_keys, read_jsonl, write_jsonl

NOUL_TRUE = {"true", "t", "yes", "y", "1", "是", "对", "真"}
NOUL_FALSE = {"false", "f", "no", "n", "0", "否", "不是", "错", "假"}


def export(test_path: Path = DATA / "dataset" / "test.jsonl", out: Path = DATA / "annotation" / "to_label.csv",
           per_scenario: int = 40, seed: int = 7, log=print) -> Path:
    rows = list(read_jsonl(test_path))
    by_w = defaultdict(list)
    for r in rows:
        by_w[r["workflow"]].append(r)
    rng = random.Random(seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "scenario", "question", "type", "instructions", "options", "state", "human_label", "note"])
        for wf, rs in sorted(by_w.items()):
            rng.shuffle(rs)
            for r in rs[:per_scenario]:
                qs = json.loads(r["questions"])
                for qid, q in qs.items():
                    if q["type"] == "choice":
                        opts = " | ".join(f"{k}: {v}" if v else k for k, v in q["criteria"].items())
                    elif q["type"] == "score":
                        opts = " | ".join(f"{i}: {c}" for i, c in enumerate(q["criteria"]))
                    else:
                        opts = "true | false"
                    w.writerow([r["id"], wf, qid, q["type"], q["instructions"], opts,
                                json.dumps(json.loads(r["state"]), ensure_ascii=False, indent=1), "", ""])
                    n += 1
    log(f"[annotate] 导出 {n} 道题 → {out}。在 human_label 列填写：choice 填选项名，score 填数字，noul 填 true/false。")
    return out


def _parse(q: dict, v: str):
    v = (v or "").strip()
    if not v:
        return None
    t = q["type"]
    if t == "noul":
        lv = v.lower()
        return "true" if lv in NOUL_TRUE else "false" if lv in NOUL_FALSE else ValueError(v)
    keys = option_keys(q)
    if t == "score":
        try:
            s = str(int(float(v)))
        except ValueError:
            return ValueError(v)
        return s if s in keys else ValueError(v)
    return v if v in keys else ValueError(v)


def import_(csv_paths: List[Path], test_path: Path = DATA / "dataset" / "test.jsonl",
            out: Path = DATA / "annotation" / "human_gold.jsonl", log=print) -> Path:
    rows = {r["id"]: r for r in read_jsonl(test_path)}
    votes: Dict[tuple, List[str]] = defaultdict(list)
    per_file: Dict[str, Dict[tuple, str]] = {}
    bad = []
    for p in csv_paths:
        if not Path(p).is_file():
            raise SystemExit(f"找不到 {p}。先用 Excel 打开 to_label.csv，填好 human_label 列，另存为这个文件名。")
        mine = {}
        with Path(p).open(encoding="utf-8-sig", newline="") as f:
            for rec in csv.DictReader(f):
                r = rows.get(rec["row_id"])
                if not r:
                    continue
                q = json.loads(r["questions"]).get(rec["question"])
                if not q:
                    continue
                lab = _parse(q, rec.get("human_label", ""))
                if lab is None:
                    continue
                if isinstance(lab, Exception):
                    bad.append(f"{p}: {rec['row_id']}.{rec['question']} = {rec.get('human_label')!r}")
                    continue
                key = (rec["row_id"], rec["question"])
                votes[key].append(lab)
                mine[key] = lab
        per_file[str(p)] = mine
    if bad:
        log("[annotate] 以下答案无法识别，已跳过：\n  " + "\n  ".join(bad[:30]))

    out_rows = defaultdict(dict)
    for (rid, qid), vs in votes.items():
        q = json.loads(rows[rid]["questions"])[qid]
        keys = option_keys(q)
        c = Counter(vs)
        out_rows[rid][qid] = gold_entry(q, normalize({k: c.get(k, 0) for k in keys}, keys))
        out_rows[rid][qid]["n_annotators"] = len(vs)
    res = []
    for rid, gold in out_rows.items():
        r = dict(rows[rid])
        r["gold"] = json.dumps(gold, ensure_ascii=False)
        r.pop("disputed", None)
        r["source"] = "human"
        res.append(r)
    write_jsonl(out, res)
    log(f"[annotate] 人工金标 {sum(len(g) for g in out_rows.values())} 题 / {len(res)} 条 → {out}")
    for a, b in combinations(per_file, 2):
        common = set(per_file[a]) & set(per_file[b])
        if common:
            agree = sum(per_file[a][k] == per_file[b][k] for k in common) / len(common)
            log(f"[annotate] 人际一致率 {Path(a).name} vs {Path(b).name}：{agree:.3f}（{len(common)} 题）")
    return out
