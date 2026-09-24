# -*- coding: utf-8 -*-
"""用一个或多个老师给 state 打标签。结果按老师分别追加到 distill_data/labels/<老师>.jsonl。

可中断、可续跑：已经存在的 (state_id, 问题集哈希) 会跳过。
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from .common import DATA, append_jsonl, clean_questions, now, read_jsonl, sha


def teacher_file(name: str, labels_dir: Path = DATA / "labels") -> Path:
    return labels_dir / (re.sub(r"[^A-Za-z0-9._-]+", "_", name) + ".jsonl")


def done_keys(path: Path) -> set:
    return {(r["state_id"], r["qhash"]) for r in read_jsonl(path)}


def run_label(teacher, states: List[dict], scenarios: Dict[str, dict], labels_dir: Path = DATA / "labels",
              workers: int = 1, limit: int = 0, max_consecutive_errors: int = 5, log=print) -> dict:
    out = teacher_file(teacher.name, labels_dir)
    err_file = labels_dir / "errors.jsonl"
    done = done_keys(out)
    todo = []
    for st in states:
        sc = scenarios.get(st["scenario"])
        if sc is None:
            continue
        qs = clean_questions(sc["questions"])
        key = (st["id"], sha(qs))
        if key not in done:
            todo.append((st, qs, key[1]))
    if limit:
        todo = todo[:limit]
    log(f"[label] {teacher.name}: 已完成 {len(done)}，本次待标 {len(todo)} → {out}")

    stats = {"ok": 0, "err": 0}
    every = 1 if len(todo) <= 50 else 25  # 待标很少时逐条显示进度，避免看起来像卡住
    consecutive = 0
    t0 = time.time()

    def work(item):
        st, qs, qh = item
        dists = teacher.label(st["state"], qs)
        return st, qh, dists, getattr(teacher, "last_meta", {})

    def record(st, qh, dists, meta):
        append_jsonl(out, {"state_id": st["id"], "scenario": st["scenario"], "teacher": teacher.name,
                           "qhash": qh, "dists": dists, "meta": meta, "ts": now()})

    if workers <= 1:
        for i, item in enumerate(todo, 1):
            try:
                record(*work(item))
                stats["ok"] += 1
                consecutive = 0
            except Exception as e:  # noqa: BLE001 — 记录后继续，连续失败才停
                stats["err"] += 1
                consecutive += 1
                append_jsonl(err_file, {"teacher": teacher.name, "state_id": item[0]["id"], "error": str(e)[:500],
                                        "ts": now()})
                log(f"[label] {teacher.name} {item[0]['id']}: {e}")
                if consecutive >= max_consecutive_errors:
                    log(f"[label] 连续 {consecutive} 次失败，停止。检查 key / 网络 / Ollama 是否在运行。")
                    break
            if i % every == 0 or i == len(todo):
                rate = i / max(1e-9, time.time() - t0)
                log(f"[label] {teacher.name}: {i}/{len(todo)}  ({rate:.2f}/s)")
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(work, it): it for it in todo}
            for i, f in enumerate(as_completed(futs), 1):
                it = futs[f]
                try:
                    record(*f.result())
                    stats["ok"] += 1
                    consecutive = 0
                except Exception as e:  # noqa: BLE001
                    stats["err"] += 1
                    consecutive += 1
                    append_jsonl(err_file, {"teacher": teacher.name, "state_id": it[0]["id"],
                                            "error": str(e)[:500], "ts": now()})
                    log(f"[label] {teacher.name} {it[0]['id']}: {e}")
                    if consecutive >= max_consecutive_errors:
                        log("[label] 连续失败过多，取消剩余任务。")
                        for g in futs:
                            g.cancel()
                        break
                if i % every == 0 or i == len(todo):
                    rate = i / max(1e-9, time.time() - t0)
                    log(f"[label] {teacher.name}: {i}/{len(todo)}  ({rate:.2f}/s)")
    log(f"[label] {teacher.name}: 成功 {stats['ok']}，失败 {stats['err']}")
    return stats
