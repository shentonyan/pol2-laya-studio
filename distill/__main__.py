# -*- coding: utf-8 -*-
"""python -m distill <命令>   —— 用 Jev + 本地 LLM 当老师，蒸馏出 PoL2 专用的 Laya 判别模型。

    seeds      从 docs/scenarios.json 导出种子 state
    generate   用本地 Ollama 生成合成 state
    label      让老师打标签（可续跑）
    build      聚合成 Laya 微调数据 + 分歧表 + 报告
    annotate   人工金标集：export / import
    train      单卡 RLCD 微调（需要 torch + laya）
    eval       在 gold 上比较基础模型 / 学生 / 老师
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .common import DATA, SCENARIOS, load_dotenv, load_scenarios, read_jsonl, write_jsonl


def _scen(a):
    return load_scenarios(a.scenarios, a.group, a.only.split(",") if a.only else None)


def _states(paths):
    rows, seen = [], set()
    for p in paths:
        for r in read_jsonl(p):
            if r["id"] not in seen:
                seen.add(r["id"])
                rows.append(r)
    if not rows:
        sys.exit(f"没有读到任何 state：{[str(p) for p in paths]}")
    return rows


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台中文
    load_dotenv()
    ap = argparse.ArgumentParser(prog="python -m distill", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", type=Path, default=SCENARIOS)
    ap.add_argument("--group", default="pol2", help="pol2 / general / all（默认 pol2）")
    ap.add_argument("--only", default="", help="只处理这些场景，逗号分隔，如 lovelang,syco")
    # --scenarios / --group / --only 写在子命令前后都可以
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--scenarios", type=Path, default=argparse.SUPPRESS)
    common.add_argument("--group", default=argparse.SUPPRESS)
    common.add_argument("--only", default=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add = sub.add_parser
    sub.add_parser = lambda name, **kw: _add(name, parents=[common], **kw)

    s = sub.add_parser("seeds")
    s.add_argument("--out", type=Path, default=DATA / "states" / "seeds.jsonl")

    s = sub.add_parser("generate")
    s.add_argument("--model", required=True, help="Ollama 模型，例如 gemma4:26b")
    s.add_argument("--per-scenario", type=int, default=200)
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--temperature", type=float, default=0.9)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--host", default="http://127.0.0.1:11434")
    s.add_argument("--out", type=Path, default=None)

    s = sub.add_parser("label")
    s.add_argument("--teacher", action="append", required=True,
                   help="可重复：jev、ollama:qwen2.5:7b、ollama:gemma4:26b、mock:a")
    s.add_argument("--states", type=Path, action="append", default=None,
                   help="默认 distill_data/states/ 下全部 .jsonl")
    s.add_argument("--workers", type=int, default=1,
                   help="并发数。Jev 可设 4–8；Ollama 需先设 OLLAMA_NUM_PARALLEL 才能真正并行")
    s.add_argument("--max-retries", type=int, default=5, help="Jev 遇到 429/529 等繁忙错误时的重试次数")
    s.add_argument("--samples", type=int, default=3, help="Ollama 每条采样次数，投票成分布")
    s.add_argument("--temperature", type=float, default=0.7)
    s.add_argument("--min-interval", type=float, default=0.0, help="Jev 两次调用的最小间隔（秒）")
    s.add_argument("--limit", type=int, default=0, help="只标前 N 条（试跑用）")

    s = sub.add_parser("build")
    s.add_argument("--teacher", action="append", required=True)
    s.add_argument("--weight", action="append", default=[], help="老师=权重，例如 jev:openjev=1.5")
    s.add_argument("--states", type=Path, action="append", default=None)
    s.add_argument("--eps", type=float, default=0.02)
    s.add_argument("--min-teachers", type=int, default=0, help="默认要求所有老师都标过")
    s.add_argument("--agree-min", type=float, default=2 / 3, help="多数票比例下限，默认 2/3（三个老师里两个一致即可）")
    s.add_argument("--margin-min", type=float, default=0.10)
    s.add_argument("--test-frac", type=float, default=0.15)
    s.add_argument("--no-tie-arena", action="store_true")
    s.add_argument("--no-tie-syco", action="store_true")
    s.add_argument("--exclude", action="append", default=[],
                   help="老师=场景.题目，该题不用这个老师；可重复，也可写 场景.*。例：ollama:qwen2.5:7b=syco.sycophantic")
    s.add_argument("--out", type=Path, default=DATA / "dataset")

    s = sub.add_parser("annotate")
    s.add_argument("action", choices=["export", "import"])
    s.add_argument("--per-scenario", type=int, default=40)
    s.add_argument("--csv", type=Path, action="append", default=None, help="import 时的已填表格，可多份")

    s = sub.add_parser("autogold", help="不依赖人工的独立 gold：design（按构造）/ judge（未参与训练的模型）")
    s.add_argument("action", choices=["design", "judge"])
    s.add_argument("--judge", default="ollama:gemma4:26b", help="独立裁判，必须不在 --used 里")
    s.add_argument("--used", action="append", default=[], help="参与了训练标签的老师，可重复")
    s.add_argument("--samples", type=int, default=3)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--test", type=Path, default=DATA / "dataset" / "test.jsonl")
    s.add_argument("--fresh", type=Path, default=None,
                   help="design：改用一批全新生成、从未打标签也未训练过的 state（放在 states 目录之外）")

    s = sub.add_parser("train")
    s.add_argument("--train", type=Path, default=DATA / "dataset" / "train.jsonl")
    s.add_argument("--base", default="convaiinnovations/laya")
    s.add_argument("--subfolder", default="multilingual", help="multilingual / typed-decisions / none(英文)")
    s.add_argument("--epochs", type=int, default=4)
    s.add_argument("--micro-batch", type=int, default=8)
    s.add_argument("--grad-accum", type=int, default=8)
    s.add_argument("--lr-encoder", type=float, default=2.5e-5)
    s.add_argument("--lr-head", type=float, default=1e-4)
    s.add_argument("--max-len", type=int, default=0)
    s.add_argument("--no-grad-ckpt", action="store_true")
    s.add_argument("--out", type=Path, default=None)

    s = sub.add_parser("eval")
    s.add_argument("--gold", type=Path, default=DATA / "dataset" / "test.jsonl")
    s.add_argument("--model", action="append", default=[],
                   help="base:multilingual / base:english / base:typed-decisions / 本地 checkpoint 目录")
    s.add_argument("--teacher", action="append", default=[])
    s.add_argument("--states", type=Path, action="append", default=None)
    s.add_argument("--skip-disputed", action="store_true")
    s.add_argument("--out", type=Path, default=None)

    a = ap.parse_args(argv)
    states_default = sorted((DATA / "states").glob("*.jsonl"))

    if a.cmd == "seeds":
        from .states import seeds
        n = write_jsonl(a.out, seeds(_scen(a)))
        print(f"[seeds] {n} 条 → {a.out}")

    elif a.cmd == "generate":
        from .states import generate
        rows = generate(_scen(a), a.model, a.per_scenario, a.batch, a.host, temperature=a.temperature, seed=a.seed)
        out = a.out or DATA / "states" / f"gen_{a.model.replace(':', '_').replace('/', '_')}_s{a.seed}.jsonl"
        n = write_jsonl(out, rows)
        print(f"[generate] {n} 条 → {out}")

    elif a.cmd == "label":
        from .label import run_label
        from .teachers import make_teacher
        scen = _scen(a)
        states = _states(a.states or states_default)
        for spec in a.teacher:
            t = make_teacher(spec, samples=a.samples, temperature=a.temperature, min_interval=a.min_interval,
                             max_retries=a.max_retries)
            run_label(t, states, scen, workers=a.workers, limit=a.limit)

    elif a.cmd == "build":
        from .build import build
        names = [_teacher_name(t) for t in a.teacher]
        weights = {}
        for w in a.weight:
            k, _, v = w.rpartition("=")
            weights[_teacher_name(k)] = float(v)
        exclude = {}
        for e in a.exclude:
            k, _, v = e.rpartition("=")
            exclude.setdefault(_teacher_name(k), set()).add(v)
        build(_states(a.states or states_default), _scen(a), names, out_dir=a.out, weights=weights, eps=a.eps,
              exclude=exclude,
              min_teachers=a.min_teachers or None, agree_min=a.agree_min, margin_min=a.margin_min,
              test_frac=a.test_frac, tie_arena=not a.no_tie_arena, tie_syco=not a.no_tie_syco)

    elif a.cmd == "annotate":
        from . import annotate
        if a.action == "export":
            annotate.export(per_scenario=a.per_scenario)
        else:
            if not a.csv:
                sys.exit("import 需要 --csv <已填表格>")
            annotate.import_(a.csv)

    elif a.cmd == "autogold":
        from . import autogold
        states = _states(states_default)
        if a.action == "design":
            if a.fresh:
                fresh = _states([a.fresh])
                autogold.design_gold(autogold.rows_from_states(fresh, _scen(a)), fresh,
                                     DATA / "autogold" / "design_gold_fresh.jsonl")
            else:
                autogold.design_gold(a.test, states, DATA / "autogold" / "design_gold.jsonl")
        else:
            from .label import run_label
            from .teachers import make_teacher
            used = [_teacher_name(t) for t in a.used]
            judge = _teacher_name(a.judge)
            if not used:
                sys.exit("请用 --used 列出参与训练标签的老师，以确认裁判是独立的。")
            if judge in used:
                sys.exit(f"{judge} 参与了训练标签，不能当独立裁判。")
            tfile = autogold.test_states(a.test, states, DATA / "autogold" / "test_states.jsonl")
            t = make_teacher(a.judge, samples=a.samples)
            run_label(t, _states([tfile]), _scen(a), workers=a.workers)
            safe = judge.replace(":", "_").replace("/", "_")
            autogold.judge_gold(a.test, judge, used, DATA / "autogold" / f"judge_gold_{safe}.jsonl")

    elif a.cmd == "train":
        from .train import train
        sub_ = None if a.subfolder in ("", "none", "english") else a.subfolder
        train(a.train, a.out, a.base, sub_, a.epochs, a.micro_batch, a.grad_accum, a.lr_encoder, a.lr_head,
              grad_ckpt=not a.no_grad_ckpt, max_len=a.max_len or None)

    elif a.cmd == "eval":
        _eval(a, a.states or states_default)


def _eval(a, states_paths):
    from .evaluate import evaluate
    tmp = None
    if states_paths:  # 不变性诊断需要 state 的 meta（归属、A/B 互换）
        tmp = DATA / "eval" / "_states.tmp.jsonl"
        write_jsonl(tmp, _states(states_paths))
    try:
        evaluate(a.gold, a.model, [_teacher_name(t) for t in a.teacher], states_path=tmp, out_path=a.out,
                 skip_disputed=a.skip_disputed)
    finally:
        if tmp:
            tmp.unlink(missing_ok=True)


def _teacher_name(spec: str) -> str:
    """命令行写法 → 标签文件里的老师名（不创建客户端，不需要 key）。"""
    kind, _, rest = spec.partition(":")
    if kind == "jev":
        return f"jev:{rest or 'openjev'}"
    if kind == "mock":
        return f"mock:{(rest or 'a').partition('@')[0]}"
    return spec


if __name__ == "__main__":
    main()
