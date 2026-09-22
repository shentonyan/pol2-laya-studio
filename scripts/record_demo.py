# -*- coding: utf-8 -*-
"""
用真实 laya 模型跑一遍 docs/scenarios.json 里的所有内置样例，
把结果写入 docs/results.json，供 GitHub Pages 静态演示回放。

    python scripts/record_demo.py

每个样例 × 4 种模型设置（auto / english / multilingual / typed-decisions）。
修改 scenarios.json 后需要重新录制，否则演示页会提示找不到对应结果。
"""
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import MODELS, _default, load_router, predict, warmup  # noqa: E402


def main():
    cfg = json.loads((ROOT / "docs" / "scenarios.json").read_text(encoding="utf-8"))
    router, info = load_router()
    print("[record] 预热中 …")
    warmup(router)

    try:
        from importlib.metadata import version
        laya_ver = version("laya")
    except Exception:  # noqa: BLE001
        laya_ver = "unknown"

    results = {}
    total = sum(len(s["samples"]) for s in cfg["scenarios"]) * (len(MODELS) + 1)
    done = 0
    for sc in cfg["scenarios"]:
        for i, sample in enumerate(sc["samples"]):
            for m in ("auto",) + MODELS:
                res, ms = predict(router, sample["state"], sc["questions"], m)
                results[f"{sc['id']}|{i}|{m}"] = {"result": res, "ms": ms}
                done += 1
                print(f"[record] {done}/{total}  {sc['id']} #{i} {m:<16} {ms:7.1f} ms")

    out = {
        "meta": {
            "recorded_at": dt.date.today().isoformat(),
            "gpu": info["gpu"],
            "device": info["device"],
            "torch": info["torch"],
            "laya": laya_ver,
            "note": "Real outputs of the base laya checkpoints on the built-in samples. "
                    "Demonstration only; not evidence for any claim.",
        },
        "results": results,
    }
    path = ROOT / "docs" / "results.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=_default), encoding="utf-8")
    print(f"[record] 已写入 {path}（{len(results)} 条）")


if __name__ == "__main__":
    main()
