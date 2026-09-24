# -*- coding: utf-8 -*-
"""把多个老师的标签聚合成 Laya 微调数据（LocalLLaMA/typed-decisions 的行格式）。

输出（distill_data/dataset/）：
    train.jsonl   训练用；老师有分歧的题目不进 gold
    test.jsonl    按 group_id 留出的测试集（老师共识，含分歧标记）
    review.csv    老师有分歧的题目，交人工看（Excel 可直接打开）
    report.md     数据与老师一致性报告，包括老师自身的偏差诊断

行格式与 Laya 官方微调 notebook 读取的一致：
    {"id", "workflow", "state": "<JSON 字符串>", "questions": "<JSON 字符串>", "gold": "<JSON 字符串>"}
"""
from __future__ import annotations

import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .common import (DATA, argmax, clean_questions, gold_entry, js_divergence, normalize, now, read_jsonl,
                     sha, smooth, write_jsonl)
from .label import teacher_file

# ---------------------------------------------------------- 结构性对称 ------
ARENA_QSWAP = {"A_keeps_truth": "B_keeps_truth", "B_keeps_truth": "A_keeps_truth"}


def _swap_ab(d: Dict[str, float]) -> Dict[str, float]:
    return {("B" if k == "A" else "A" if k == "B" else k): v for k, v in d.items()}


def canon_q(st: dict, qid: str, dist: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
    """把一条 state 上的答案映射到它所在「对称组」的规范朝向。映射是对合的，反向用同一个函数。"""
    if st["scenario"] == "arena" and st.get("meta", {}).get("swapped"):
        qid = ARENA_QSWAP.get(qid, qid)
        if set(dist) >= {"A", "B"}:
            dist = _swap_ab(dist)
    return qid, dist


def tie_key(st: dict, tie_arena: bool, tie_syco: bool) -> tuple:
    m = st.get("meta", {})
    if tie_arena and st["scenario"] == "arena":
        return ("arena", st["group_id"])
    if tie_syco and st["scenario"] == "syco" and m.get("reply_kind"):
        # 同一主张、同一回复，只是归属不同：事实判断不应随归属变化
        return ("syco", st["group_id"], m["reply_kind"])
    return ("state", st["id"])


def mean_dist(ds: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list(ds[0])
    return normalize({k: sum(d.get(k, 0.0) for d in ds) / len(ds) for k in keys}, keys)


def disputed(q: dict, gold: Dict[str, float], votes: List[str], agree_min: float, margin_min: float,
             noul_band: Tuple[float, float]) -> Optional[str]:
    if votes:
        top = Counter(votes).most_common(1)[0][1] / len(votes)
        if top < agree_min - 1e-9:
            return f"老师多数票比例 {top:.2f} < {agree_min:.2f}"
    if q["type"] == "noul":
        p = gold["true"]
        if noul_band[0] <= p <= noul_band[1]:
            return f"共识 P(true)={p:.2f} 落在不确定区间"
    else:
        s = sorted(gold.values(), reverse=True)
        if len(s) > 1 and s[0] - s[1] < margin_min:
            return f"前两名概率差 {s[0] - s[1]:.2f} < {margin_min:.2f}"
    return None


def load_teacher_labels(names: List[str], labels_dir: Path, qhashes: Dict[str, str]) -> Dict[str, Dict[str, dict]]:
    """{teacher: {state_id: dists}}；问题集已改动的旧标签（qhash 不符）会被忽略。"""
    out = {}
    for n in names:
        p = teacher_file(n, labels_dir)
        if not p.is_file():
            raise FileNotFoundError(f"找不到老师 {n} 的标签文件 {p}。先运行 label。")
        m = {}
        for r in read_jsonl(p):
            if qhashes.get(r["scenario"]) == r["qhash"]:
                m[r["state_id"]] = r["dists"]
        out[n] = m
    return out


def build(states: List[dict], scenarios: Dict[str, dict], teachers: List[str], out_dir: Path = DATA / "dataset",
          labels_dir: Path = DATA / "labels", weights: Optional[Dict[str, float]] = None, eps: float = 0.02,
          min_teachers: Optional[int] = None, agree_min: float = 2 / 3, margin_min: float = 0.10,
          noul_band: Tuple[float, float] = (0.35, 0.65), test_frac: float = 0.15, tie_arena: bool = True,
          tie_syco: bool = True, split_seed: str = "pol2", exclude: Optional[Dict[str, set]] = None,
          log=print) -> dict:
    """exclude：{老师: {"syco.sycophantic", "eap.*", ...}}，这些题目不使用该老师的标签。"""
    weights = weights or {}
    exclude = exclude or {}

    def excluded(t, scen, qid):
        ex = exclude.get(t, ())
        return f"{scen}.{qid}" in ex or f"{scen}.*" in ex
    qsets = {sid: clean_questions(s["questions"]) for sid, s in scenarios.items()}
    qhashes = {sid: sha(q) for sid, q in qsets.items()}
    labels = load_teacher_labels(teachers, labels_dir, qhashes)
    min_teachers = min_teachers or len(teachers)
    states = [s for s in states if s["scenario"] in qsets]
    by_id = {s["id"]: s for s in states}

    # 1) 每个老师在每个对称组、每道规范题上的分布（组内平均）
    acc: Dict[tuple, Dict[str, Dict[str, List[dict]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for t in teachers:
        for sid_, dists in labels[t].items():
            st = by_id.get(sid_)
            if st is None:
                continue
            tk = tie_key(st, tie_arena, tie_syco)
            for qid, d in dists.items():
                if excluded(t, st["scenario"], qid):
                    continue
                cq, cd = canon_q(st, qid, d)
                acc[tk][cq][t].append(cd)

    # 2) 聚合为 gold，并判断分歧
    tk_scen = {tie_key(st, tie_arena, tie_syco): st["scenario"] for st in states}
    gold_c: Dict[tuple, Dict[str, dict]] = {}
    for tk, qmap in acc.items():
        for cq, per_t in qmap.items():
            t_dists = {t: smooth(mean_dist(ds), eps) for t, ds in per_t.items()}
            gold_c.setdefault(tk, {})[cq] = {"t_dists": t_dists}

    for tk, qmap in gold_c.items():
        scen = tk_scen.get(tk)
        if scen is None:
            continue
        for cq, info in qmap.items():
            q = qsets[scen].get(cq)
            if q is None:
                info["missing"] = True
                continue
            td = info["t_dists"]
            need = min(min_teachers, sum(1 for t in teachers if not excluded(t, scen, cq)))
            if len(td) < max(1, need):
                info["missing"] = True
                continue
            tot = sum(weights.get(t, 1.0) for t in td)
            keys = list(next(iter(td.values())))
            g = normalize({k: sum(weights.get(t, 1.0) * d[k] for t, d in td.items()) / tot for k in keys}, keys)
            votes = [argmax(d) for d in td.values()]
            info.update(gold=g, votes=dict(zip(td, votes)),
                        reason=disputed(q, g, votes, agree_min, margin_min, noul_band))

    # 3) 映射回每条 state，切分并写出
    train, test, review = [], [], []
    seen_review = set()
    stat_q = defaultdict(lambda: Counter())
    label_dist = defaultdict(Counter)
    for st in states:
        tk = tie_key(st, tie_arena, tie_syco)
        qs = qsets[st["scenario"]]
        gold, disp = {}, {}
        for qid, q in qs.items():
            cq, _ = canon_q(st, qid, {})
            info = gold_c.get(tk, {}).get(cq)
            key = f"{st['scenario']}.{qid}"
            if not info or info.get("missing"):
                stat_q[key]["missing"] += 1
                continue
            _, g = canon_q(st, qid, info["gold"])
            gold[qid] = gold_entry(q, g)
            if info["reason"]:
                disp[qid] = info["reason"]
                stat_q[key]["disputed"] += 1
                if (tk, cq) not in seen_review:  # 同一对称组只交人工看一次
                    seen_review.add((tk, cq))
                    review.append((st, qid, q, info, g))
            else:
                stat_q[key]["clean"] += 1
        if not gold:
            continue
        in_test = int(sha([split_seed, st["group_id"]], 8), 16) / 16 ** 8 < test_frac
        row = {"id": st["id"], "workflow": st["scenario"], "group_id": st["group_id"], "source": st.get("source"),
               "state": json.dumps(st["state"], ensure_ascii=False),
               "questions": json.dumps(qs, ensure_ascii=False)}
        if in_test:
            row["gold"] = json.dumps(gold, ensure_ascii=False)
            row["disputed"] = disp
            test.append(row)
        else:
            clean = {k: v for k, v in gold.items() if k not in disp}
            if not clean:
                continue
            row["gold"] = json.dumps(clean, ensure_ascii=False)
            train.append(row)
            for k, v in clean.items():
                label_dist[f"{st['scenario']}.{k}"][v["label"]] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    n_train = write_jsonl(out_dir / "train.jsonl", train)
    n_test = write_jsonl(out_dir / "test.jsonl", test)
    _write_review(out_dir / "review.csv", review, teachers)
    diag = teacher_diagnostics(states, labels, teachers, qsets)
    degen = degenerate_teachers(states, labels, teachers)
    report = _report(teachers, weights, eps, agree_min, margin_min, noul_band, test_frac, n_train, n_test,
                     stat_q, label_dist, diag, labels, qsets, by_id, degen, exclude)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    log(f"[build] train {n_train} 行，test {n_test} 行，待人工复核 {len(review)} 题 → {out_dir}")
    return {"train": n_train, "test": n_test, "review": len(review), "diag": diag}


def _write_review(path: Path, review, teachers: List[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state_id", "scenario", "group_id", "question", "type", "instructions", "options", "reason",
                    "consensus", *[f"{t}" for t in teachers], "state", "human_label", "note"])
        for st, qid, q, info, g in review:
            opts = list(g)
            teacher_cols = []
            for t in teachers:
                d = info["t_dists"].get(t)
                if d is None:
                    teacher_cols.append("")
                    continue
                _, d2 = canon_q(st, qid, d)
                teacher_cols.append(" ".join(f"{k}={v:.2f}" for k, v in d2.items()))
            w.writerow([st["id"], st["scenario"], st["group_id"], qid, q["type"], q["instructions"], "|".join(opts),
                        info["reason"], " ".join(f"{k}={v:.2f}" for k, v in g.items()), *teacher_cols,
                        json.dumps(st["state"], ensure_ascii=False), "", ""])


# --------------------------------------------------------- 老师自身诊断 -----
def teacher_diagnostics(states, labels, teachers, qsets) -> dict:
    """先检验尺子：
    - syco：同一主张、同一回复，只改归属时，老师的 P(sycophantic) 变化多少（理想为 0）
    - arena：A/B 互换后，老师的「哪个更好」是否跟着互换（位置一致性，理想为 1）
    """
    by_id = {s["id"]: s for s in states}
    out = {}
    for t in teachers:
        L = labels[t]
        # syco 归属敏感度
        pairs = defaultdict(dict)
        for sid_, d in L.items():
            st = by_id.get(sid_)
            if not st or st["scenario"] != "syco":
                continue
            m = st.get("meta", {})
            if m.get("reply_kind") and m.get("attribution"):
                pairs[(st["group_id"], m["reply_kind"])][m["attribution"]] = d
        deltas = [p["user"]["sycophantic"]["true"] - p["third"]["sycophantic"]["true"]
                  for p in pairs.values() if {"user", "third"} <= set(p) and "sycophantic" in p["user"]]
        # arena 位置一致性
        twins = defaultdict(dict)
        for sid_, d in L.items():
            st = by_id.get(sid_)
            if not st or st["scenario"] != "arena" or "better" not in d:
                continue
            twins[st["group_id"]]["swapped" if st.get("meta", {}).get("swapped") else "orig"] = d["better"]
        cons, tvs = [], []
        for tw in twins.values():
            if {"orig", "swapped"} <= set(tw):
                back = _swap_ab(tw["swapped"])
                cons.append(argmax(tw["orig"]) == argmax(back))
                tvs.append(0.5 * sum(abs(tw["orig"][k] - back.get(k, 0.0)) for k in tw["orig"]))
        out[t] = {
            "syco_pairs": len(deltas),
            "syco_mean_delta": sum(deltas) / len(deltas) if deltas else None,
            "syco_mean_abs_delta": sum(abs(x) for x in deltas) / len(deltas) if deltas else None,
            "arena_pairs": len(cons),
            "arena_position_consistency": sum(cons) / len(cons) if cons else None,
            "arena_mean_tv": sum(tvs) / len(tvs) if tvs else None,
        }
    return out


def degenerate_teachers(states, labels, teachers, others_min=0.05, ratio=0.2, n_min=30):
    """找出在某道题上几乎只给一个答案的老师：它给出「非最常见答案」的比例，
    不到其他老师平均比例的 ratio 倍（而其他老师至少有 others_min 的答案不是最常见的那个）。"""
    by_id = {s["id"]: s for s in states}
    votes = defaultdict(lambda: defaultdict(list))
    for t in teachers:
        for sid_, dists in labels[t].items():
            st = by_id.get(sid_)
            if not st:
                continue
            for qid, d in dists.items():
                votes[f"{st['scenario']}.{qid}"][t].append(argmax(d))
    out = []
    for key, per_t in votes.items():
        top = {t: (Counter(v).most_common(1)[0][1] / len(v), len(v)) for t, v in per_t.items() if v}
        for t, (own, n) in top.items():
            rest = [1 - x for u, (x, _) in top.items() if u != t]
            if n < n_min or not rest:
                continue
            others_rest = sum(rest) / len(rest)
            if others_rest >= others_min and (1 - own) <= ratio * others_rest:
                out.append((t, key, own, 1 - others_rest, n))
    return sorted(out)


def _fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def _report(teachers, weights, eps, agree_min, margin_min, noul_band, test_frac, n_train, n_test, stat_q,
            label_dist, diag, labels, qsets, by_id, degen=None, exclude=None) -> str:
    L = [f"# 蒸馏数据报告", "", f"生成时间：{now()}", "",
         "> 诚实条款：gold 是老师模型的共识分布，不是人工真值。学生模型学到的上限是老师，老师的偏差也会一起学走。"
         "判断质量必须用独立的人工金标集评估（`python -m distill annotate`）。", "",
         "## 设置", "",
         f"- 老师：{', '.join(f'`{t}`（权重 {weights.get(t, 1.0)}）' for t in teachers)}",
         f"- 平滑 ε = {eps}；多数票下限 {agree_min}；choice/score 前两名差下限 {margin_min}；noul 不确定区间 {list(noul_band)}",
         f"- 按 group_id 留出测试集比例 {test_frac}", f"- 训练 {n_train} 行，测试 {n_test} 行",
         f"- 排除：{'；'.join(f'`{t}` 不参与 ' + '、'.join(sorted(v)) for t, v in (exclude or {}).items()) or '无'}", "",
         "## 每道题的分歧率", "", "| 题目 | 共识 | 分歧 | 缺失 | 分歧率 |", "|---|---:|---:|---:|---:|"]
    for k in sorted(stat_q):
        c = stat_q[k]
        tot = c["clean"] + c["disputed"]
        L.append(f"| `{k}` | {c['clean']} | {c['disputed']} | {c['missing']} | {_fmt(c['disputed'] / tot if tot else None, 2)} |")
    L += ["", "## 训练集标签分布（检查类别是否失衡）", ""]
    for k in sorted(label_dist):
        tot = sum(label_dist[k].values())
        L.append(f"- `{k}`：" + "，".join(f"{lab} {n}（{n / tot:.0%}）" for lab, n in label_dist[k].most_common()))
    L += ["", "## 老师两两一致率（argmax 相同的比例）", "", "| 老师 A | 老师 B | 题数 | 一致率 | 平均 JS 散度 |", "|---|---|---:|---:|---:|"]
    for a, b in itertools.combinations(teachers, 2):
        same = n = 0
        js = []
        for sid_, da in labels[a].items():
            db = labels[b].get(sid_)
            if not db:
                continue
            for qid in da:
                if qid in db:
                    n += 1
                    same += argmax(da[qid]) == argmax(db[qid])
                    js.append(js_divergence(normalize(da[qid], list(da[qid])), normalize(db[qid], list(da[qid]))))
        L.append(f"| `{a}` | `{b}` | {n} | {_fmt(same / n if n else None)} | {_fmt(sum(js) / len(js) if js else None)} |")
    if degen:
        L += ["", "## ⚠ 疑似「常数答案」的老师", "",
              "某个老师在一道题上几乎总给同一个答案，而其他老师的答案有变化：它在这道题上可能没有真正在判断，"
              "只是把票数往一边拉。考虑用 `build --exclude 老师=场景.题目` 排除。", "",
              "| 老师 | 题目 | 该老师最常见答案占比 | 其他老师平均 | n |", "|---|---|---:|---:|---:|"]
        for t, key, own, others, n in degen:
            L.append(f"| `{t}` | `{key}` | {own:.2f} | {others:.2f} | {n} |")
    L += ["", "## 先检验尺子：老师自身的偏差", "",
          "| 老师 | syco 配对数 | 归属效应 ΔP(谄媚)（本人−第三方） | 平均 |Δ| | arena 配对数 | A/B 互换一致率 | 平均 TV |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for t, d in diag.items():
        L.append(f"| `{t}` | {d['syco_pairs']} | {_fmt(d['syco_mean_delta'])} | {_fmt(d['syco_mean_abs_delta'])} | "
                 f"{d['arena_pairs']} | {_fmt(d['arena_position_consistency'])} | {_fmt(d['arena_mean_tv'])} |")
    L += ["", "理想裁判：归属效应为 0，互换一致率为 1。构建 gold 时已把同一回复的两种归属、A/B 互换孪生合并平均，"
          "因此这些偏差不会直接进入训练标签；但它们说明了老师本身的可靠程度，应当如实报告。", ""]
    return "\n".join(L)
