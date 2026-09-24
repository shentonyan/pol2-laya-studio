# -*- coding: utf-8 -*-
"""离线测试：用本地假服务模拟 Ollama 和 OpenJEV，跑通 seeds → generate → label → build → annotate。
不需要 key、不需要网络、不需要显卡。   python -m pytest tests -q
"""
import csv
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from distill import build as B
from distill import states as S
from distill import annotate as A
from distill.common import clean_questions, load_scenarios, read_jsonl, write_jsonl, answer_to_dist, option_keys
from distill.label import run_label, teacher_file
from distill.teachers import JevTeacher, MockTeacher, OllamaTeacher, answer_schema


# ------------------------------------------------------------ 假服务 ------
class Fake(BaseHTTPRequestHandler):
    rng = random.Random(0)
    n_gen = 0

    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        if self.path.endswith("/systemone"):
            assert self.headers["Authorization"] == "Bearer test-key"
            assert set(body) == {"model", "state", "questions"}
            dists = MockTeacher("jev", noise=0.3).label(body["state"], body["questions"])
            ans = {}
            for qid, q in body["questions"].items():
                d = dists[qid]
                if q["type"] == "noul":
                    ans[qid] = {"type": "noul", "noul": d["true"]}
                elif q["type"] == "score":
                    ans[qid] = {"type": "score", "score": sum(int(k) * v for k, v in d.items()),
                                "probabilities": d, "confidence": 0.5}
                else:
                    ans[qid] = {"type": "choice", "choice": max(d, key=d.get), "probabilities": d, "confidence": 0.5}
            return self._send({"model": "openjev", "answers": ans, "usage": {}})
        if self.path == "/api/chat":
            fmt = body["format"]
            props = fmt["properties"]
            if "items" in props:  # 生成
                Fake.n_gen += 1
                it = props["items"]["items"]
                n = 4
                if it.get("type") == "string":
                    items = [f"合成样本 {Fake.n_gen}-{i}：你这样做让我很失望，但我相信你能改。" for i in range(n)]
                elif "claim" in it["properties"]:
                    items = [{"claim": f"claim {Fake.n_gen}-{i} is true", "claim_is_true": i % 2 == 0,
                              "evidence": "Evidence text.", "reply_agree": "Yes, correct.",
                              "reply_disagree": "No, that's a myth.", "reply_evasive": "Hard to say."}
                             for i in range(n)]
                else:
                    items = [{k: f"{k} {Fake.n_gen}-{i}" for k in it["properties"]} for i in range(n)]
                return self._send({"message": {"content": json.dumps({"items": items}, ensure_ascii=False)}})
            ans = {}
            for qid, sch in props.items():
                if sch["type"] == "boolean":
                    ans[qid] = Fake.rng.random() < 0.5
                else:
                    ans[qid] = Fake.rng.choice(sch["enum"])
            return self._send({"message": {"content": json.dumps(ans)}})
        self.send_response(404)
        self.end_headers()


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture()
def scen():
    return load_scenarios()


# ------------------------------------------------------------ 单元 ------
def test_answer_to_dist_shapes(scen):
    q = scen["lovelang"]["questions"]
    assert answer_to_dist(q["truthful"], {"noul": 0.8}) == pytest.approx({"false": 0.2, "true": 0.8})
    d = answer_to_dist(q["warmth"], {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}})
    assert list(d) == ["0", "1", "2"]
    d = answer_to_dist(q["stance"], {"choice": "flattery"})
    assert d["flattery"] == 1.0 and set(d) == set(q["stance"]["criteria"])


def test_answer_schema_is_constrained(scen):
    sch = answer_schema(clean_questions(scen["syco"]["questions"]))
    assert sch["properties"]["sycophantic"] == {"type": "boolean"}
    assert "sycophantic_agree" in sch["properties"]["reply_type"]["enum"]


def test_seeds_make_arena_twins(scen):
    rows = S.seeds(scen)
    arena = [r for r in rows if r["scenario"] == "arena"]
    assert len(arena) == 4 and sum(bool(r["meta"].get("swapped")) for r in arena) == 2
    g = {}
    for r in arena:
        g.setdefault(r["group_id"], []).append(r)
    for pair in g.values():
        a, b = pair
        assert a["state"]["rewrite_A"] == b["state"]["rewrite_B"]


def test_syco_expand_pairs():
    rows = S.syco_expand({"claim": "水在 100 度沸腾", "claim_is_true": True, "evidence": "e",
                          "reply_agree": "对", "reply_disagree": "不对", "reply_evasive": "不好说"}, "zh", "t", "m")
    assert len(rows) == 6 and len({r["group_id"] for r in rows}) == 1
    assert {r["state"]["user_claim"][:3] for r in rows} == {"我觉得", "我同事"}


# ------------------------------------------------------------ 端到端 ------
def test_end_to_end(tmp_path, scen, server, monkeypatch):
    monkeypatch.setenv("OPENJEV_API_KEY", "test-key")
    states = S.seeds(scen) + S.generate(scen, "fake-gen", per_scenario=8, batch=4, host=server, log=lambda *a: None)
    ids = [s["id"] for s in states]
    assert len(ids) == len(set(ids))
    assert any(s["scenario"] == "syco" and s["source"] == "generated" for s in states)
    labels = tmp_path / "labels"

    teachers = [JevTeacher(base_url=server + "/v1"), OllamaTeacher("fake", host=server, samples=3),
                MockTeacher("a", noise=0.3)]
    for t in teachers:
        st = run_label(t, states, scen, labels_dir=labels, workers=2 if t.kind == "jev" else 1, log=lambda *a: None)
        assert st["err"] == 0
    # 续跑：不会重复标
    st = run_label(teachers[2], states, scen, labels_dir=labels, log=lambda *a: None)
    assert st["ok"] == 0
    names = [t.name for t in teachers]
    out = tmp_path / "dataset"
    res = B.build(states, scen, names, out_dir=out, labels_dir=labels, test_frac=0.3, log=lambda *a: None)
    assert res["train"] > 0 and res["test"] > 0

    # 格式与 Laya notebook 读取的一致
    for r in read_jsonl(out / "train.jsonl"):
        st_, qs, gold = json.loads(r["state"]), json.loads(r["questions"]), json.loads(r["gold"])
        assert r["workflow"] in scen
        for qid, g in gold.items():
            q = qs[qid]
            assert set(g["probabilities"]) == set(option_keys(q))
            assert abs(sum(g["probabilities"].values()) - 1) < 1e-4
            assert g["label"] in g["probabilities"]
            if q["type"] == "score":
                assert "score" in g
            if q["type"] == "noul":
                assert g["label"] in ("true", "false") and "noul" in g
    # group 不跨越 train/test
    tr = {r["group_id"] for r in read_jsonl(out / "train.jsonl")}
    te = {r["group_id"] for r in read_jsonl(out / "test.jsonl")}
    assert not tr & te
    assert (out / "report.md").read_text(encoding="utf-8").startswith("# 蒸馏数据报告")
    with (out / "review.csv").open(encoding="utf-8-sig") as f:
        head = next(csv.reader(f))
    assert "human_label" in head and names[0] in head

    assert res["review"] > 0  # 随机的假 Ollama 必然与其他老师有分歧

    # 对称性检查用不设分歧门槛的构建，保证每道题都有 gold
    out2 = tmp_path / "dataset_all"
    B.build(states, scen, names, out_dir=out2, labels_dir=labels, agree_min=0.0, margin_min=0.0,
            noul_band=(1.0, 0.0), log=lambda *a: None)
    # arena 孪生：gold 必须严格互为镜像
    rows = {r["id"]: r for r in list(read_jsonl(out2 / "train.jsonl")) + list(read_jsonl(out2 / "test.jsonl"))}
    by_id = {s["id"]: s for s in states}
    groups = {}
    for rid, r in rows.items():
        if r["workflow"] == "arena":
            groups.setdefault(r["group_id"], []).append(rid)
    checked = 0
    for g, rids in groups.items():
        if len(rids) != 2:
            continue
        a, b = (json.loads(rows[x]["gold"]) for x in rids)
        if "better" in a and "better" in b:
            assert a["better"]["probabilities"]["A"] == pytest.approx(b["better"]["probabilities"]["B"])
            checked += 1
    assert checked > 0

    # syco：同一回复在两种归属下 gold 相同
    by_key = {}
    for rid, r in rows.items():
        s = by_id[rid]
        if s["scenario"] == "syco" and s["meta"].get("reply_kind"):
            k = (s["group_id"], s["meta"]["reply_kind"])
            by_key.setdefault(k, []).append(json.loads(r["gold"]).get("sycophantic"))
    pairs = [v for v in by_key.values() if len(v) == 2 and all(v)]
    assert pairs and all(a["noul"] == pytest.approx(b["noul"]) for a, b in pairs)

    # 人工标注：导出 → 填写 → 导入
    csv_path = A.export(out / "test.jsonl", tmp_path / "ann" / "to_label.csv", per_scenario=3, log=lambda *a: None)
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        recs = list(csv.DictReader(f))
    assert recs and all(r["human_label"] == "" for r in recs)
    assert not any(n in ",".join(recs[0].keys()) for n in names)  # 盲标：不显示老师
    for r in recs:
        r["human_label"] = {"noul": "是", "score": "1"}.get(r["type"]) or r["options"].split(":")[0].split(" |")[0]
    filled = tmp_path / "ann" / "me.csv"
    with filled.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    hg = A.import_([filled, filled], out / "test.jsonl", tmp_path / "ann" / "human_gold.jsonl", log=lambda *a: None)
    got = list(read_jsonl(hg))
    assert got and all(json.loads(r["gold"]) for r in got)
    one = next(iter(json.loads(got[0]["gold"]).values()))
    assert one["n_annotators"] == 2 and max(one["probabilities"].values()) == 1.0

    # 老师对人工金标的评估（不需要 laya）
    from distill.evaluate import evaluate
    allst = tmp_path / "all_states.jsonl"
    write_jsonl(allst, states)
    rep = evaluate(hg, [], names, states_path=allst, out_path=tmp_path / "eval.md", labels_dir=labels,
                   log=lambda *a: None)
    assert rep["results"][names[0]]["all"]["n"] > 0
    assert "人工标注" in (tmp_path / "eval.md").read_text(encoding="utf-8")

