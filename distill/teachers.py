# -*- coding: utf-8 -*-
"""老师模型：Jev（OpenJEV API）、本地 Ollama LLM、离线 Mock。

每个老师实现 label(state, questions) -> {qid: 分布}，分布的键见 common.option_keys。
只用标准库。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional

from .common import answer_to_dist, canon, clean_questions, normalize, option_keys

RETRYABLE = {429, 500, 502, 503, 504, 529}


class TeacherError(RuntimeError):
    pass


def _http_json(url: str, payload: Optional[dict], headers: dict, timeout: float,
               max_retries: int, sleep: Callable[[float], None] = time.sleep) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    hdr = {"Accept": "application/json", **headers}
    if data is not None:
        hdr["Content-Type"] = "application/json"
    attempt = 0
    while True:
        req = urllib.request.Request(url, data=data, headers=hdr, method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:500]
            if e.code in RETRYABLE and attempt < max_retries:
                ra = e.headers.get("Retry-After")
                try:
                    wait = float(ra) if ra else None
                except ValueError:
                    wait = None
                wait = wait if wait is not None else min(30.0, 0.5 * 2 ** attempt + random.random() * 0.25)
                print(f"  [重试] HTTP {e.code}（{'服务器繁忙' if e.code in (429, 529, 503) else '服务器错误'}），"
                      f"{wait:.0f} 秒后重试 {attempt + 1}/{max_retries}", flush=True)
                sleep(wait)
                attempt += 1
                continue
            raise TeacherError(f"HTTP {e.code} @ {url}: {body}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < max_retries:
                print(f"  [重试] 网络错误，{min(30.0, 0.5 * 2 ** attempt):.0f} 秒后重试 {attempt + 1}/{max_retries}", flush=True)
                sleep(min(30.0, 0.5 * 2 ** attempt))
                attempt += 1
                continue
            raise TeacherError(f"网络错误 @ {url}: {e}") from None


def ollama_chat(host: str, payload: dict, timeout: float, max_retries: int) -> str:
    """调用 Ollama /api/chat，返回 message.content。

    - 默认发送 think=false：gemma4 / qwen3 等「思考型」模型开启思考时，经常把 token 花在思考上，
      content 为空或被截断。旧版 Ollama 或不支持该参数的模型报错时，自动去掉它重试。
    - content 为空时报告 done_reason，方便判断是被截断还是模型没有输出。
    """
    payload = {"think": False, **payload}
    try:
        resp = _http_json(f"{host}/api/chat", payload, {}, timeout, max_retries)
    except TeacherError as e:
        if "think" not in str(e).lower():
            raise
        payload.pop("think", None)
        resp = _http_json(f"{host}/api/chat", payload, {}, timeout, max_retries)
    msg = resp.get("message") or {}
    text = msg.get("content") or ""
    if not text.strip():
        raise TeacherError(f"{payload.get('model')} 输出为空（done_reason={resp.get('done_reason')!r}，"
                           f"{'有' if msg.get('thinking') else '无'}思考内容）")
    if resp.get("done_reason") == "length":
        text += "\n"  # 标记：被 num_predict 截断，交给调用方尝试抢救
    return text


# ------------------------------------------------------------------ Jev -----
class JevTeacher:
    """POST {base}/systemone，Authorization: Bearer $OPENJEV_API_KEY。一次调用回答全部问题。"""

    kind = "jev"

    def __init__(self, model: str = "openjev", base_url: Optional[str] = None,
                 min_interval: float = 0.0, timeout: float = 60.0, max_retries: int = 5):
        self.api_key = os.environ.get("OPENJEV_API_KEY")
        if not self.api_key:
            raise TeacherError("缺少 OPENJEV_API_KEY：请写进仓库根目录的 .env，或在 PowerShell 里 "
                               "$env:OPENJEV_API_KEY = '...'。不要贴进聊天或提交记录。")
        self.base = (base_url or os.environ.get("OPENJEV_BASE_URL") or "https://api.openjev.sh/v1").rstrip("/")
        self.model = model
        self.name = f"jev:{model}"
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self._last = 0.0

    def label(self, state: Any, questions: dict) -> Dict[str, Dict[str, float]]:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        qs = clean_questions(questions)
        resp = _http_json(f"{self.base}/systemone", {"model": self.model, "state": state, "questions": qs},
                          {"Authorization": f"Bearer {self.api_key}"}, self.timeout, self.max_retries)
        answers = resp.get("answers")
        if not isinstance(answers, dict):
            raise TeacherError(f"Jev 响应里没有 answers：{str(resp)[:300]}")
        out = {}
        for qid, q in qs.items():
            if qid not in answers:
                raise TeacherError(f"Jev 响应缺少问题 {qid}")
            out[qid] = answer_to_dist(q, answers[qid])
        self.last_meta = {"usage": resp.get("usage"), "model": resp.get("model")}
        return out


# --------------------------------------------------------------- Ollama -----
SYSTEM_PROMPT = (
    "You are a careful annotator. You receive a STATE (a text or a JSON document) and a list of typed "
    "questions about it. Answer every question strictly from the content of the STATE. Do not reward "
    "politeness that hides untruth, and do not punish honest criticism for being direct. "
    "Return only the JSON object requested."
)


def render_question(qid: str, q: dict) -> str:
    t, crit = q["type"], q.get("criteria")
    lines = [f"- `{qid}` ({t}): {q['instructions']}"]
    if t == "choice":
        items = crit.items() if isinstance(crit, dict) else [(c, "") for c in crit]
        lines.append("  Choose exactly one option key:")
        for k, v in items:
            desc = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) if v else ""
            lines.append(f"    * {k}" + (f": {desc}" if desc else ""))
    elif t == "score":
        lines.append("  Answer with one integer level:")
        for i, c in enumerate(crit):
            desc = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
            lines.append(f"    * {i}: {desc}")
    else:
        c = crit or {}
        tdesc = c.get("true") or "yes, the statement holds"
        fdesc = c.get("false") or "no, the statement does not hold"
        lines.append(f"  Answer true ({tdesc}) or false ({fdesc}).")
    return "\n".join(lines)


def answer_schema(questions: dict) -> dict:
    props = {}
    for qid, q in questions.items():
        t = q["type"]
        if t == "choice":
            props[qid] = {"type": "string", "enum": option_keys(q)}
        elif t == "score":
            props[qid] = {"type": "integer", "enum": list(range(len(q["criteria"])))}
        else:
            props[qid] = {"type": "boolean"}
    return {"type": "object", "properties": props, "required": list(questions)}


class OllamaTeacher:
    """本地 LLM 裁判。用 JSON Schema 约束解码保证答案合法；
    以温度采样 `samples` 次，用投票频率作为分布（这是模型自身的输出波动，不是校准过的概率）。"""

    kind = "ollama"

    def __init__(self, model: str, host: Optional[str] = None, samples: int = 3,
                 temperature: float = 0.7, timeout: float = 300.0, max_retries: int = 2, seed: int = 0):
        self.model = model
        self.name = f"ollama:{model}"
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.samples = max(1, samples)
        self.temperature = temperature if self.samples > 1 else 0.0
        self.timeout = timeout
        self.max_retries = max_retries
        self.seed = seed

    def _ask(self, state: Any, qs: dict, i: int) -> dict:
        st = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
        user = ("STATE:\n" + st + "\n\nQUESTIONS:\n" + "\n".join(render_question(k, q) for k, q in qs.items())
                + "\n\nReturn a JSON object with exactly these keys: " + ", ".join(qs))
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "format": answer_schema(qs),
            "stream": False,
            "options": {"temperature": self.temperature, "seed": self.seed + i},
        }
        text = ollama_chat(self.host, payload, self.timeout, self.max_retries)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise TeacherError(f"{self.name} 返回的不是 JSON：{text[:200]}") from None

    def label(self, state: Any, questions: dict) -> Dict[str, Dict[str, float]]:
        qs = clean_questions(questions)
        counts = {qid: {k: 0.0 for k in option_keys(q)} for qid, q in qs.items()}
        valid = {qid: 0 for qid in qs}
        for i in range(self.samples):
            ans = self._ask(state, qs, i)
            for qid, q in qs.items():
                v = ans.get(qid)
                if q["type"] == "noul":
                    if isinstance(v, str):
                        v = v.strip().lower() in ("true", "yes", "1")
                    key = "true" if v is True else "false" if v is False else None
                else:
                    key = str(v) if v is not None else None
                if key in counts[qid]:
                    counts[qid][key] += 1
                    valid[qid] += 1
        bad = [qid for qid, n in valid.items() if n == 0]
        if bad:
            raise TeacherError(f"{self.name} 对 {bad} 没有给出任何合法答案")
        self.last_meta = {"samples": self.samples, "valid": valid}
        return {qid: normalize(c, list(c)) for qid, c in counts.items()}


# ----------------------------------------------------------------- Mock -----
def _h(*parts: Any) -> float:
    d = hashlib.sha256("\x1f".join(canon(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big") / 2 ** 64


class MockTeacher:
    """离线确定性伪老师，只用于测试流程，不代表任何判断质量。
    `noise` 越大，与其他 mock 老师的分歧越多。"""

    kind = "mock"

    def __init__(self, name: str = "a", noise: float = 0.0, sharp: float = 4.0):
        self.name = f"mock:{name}"
        self.noise = noise
        self.sharp = sharp
        self._tag = name

    def label(self, state: Any, questions: dict) -> Dict[str, Dict[str, float]]:
        out = {}
        for qid, q in clean_questions(questions).items():
            keys = option_keys(q)
            logits = []
            for k in keys:
                base = _h("shared", state, qid, k)          # 所有 mock 老师共享的「信号」
                own = _h(self._tag, state, qid, k)           # 各自的噪声
                logits.append(self.sharp * ((1 - self.noise) * base + self.noise * own))
            m = max(logits)
            ex = [math.exp(x - m) for x in logits]
            s = sum(ex)
            out[qid] = {k: e / s for k, e in zip(keys, ex)}
        self.last_meta = {}
        return out


def make_teacher(spec: str, **kw):
    """spec 形如 `jev`、`jev:openjev`、`ollama:qwen2.5:7b`、`mock:a`、`mock:b@0.5`。"""
    kind, _, rest = spec.partition(":")
    if kind == "jev":
        return JevTeacher(model=rest or "openjev", min_interval=kw.get("min_interval", 0.0),
                          max_retries=kw.get("max_retries", 5))
    if kind == "ollama":
        if not rest:
            raise ValueError("ollama 老师需要模型名，例如 ollama:qwen2.5:7b")
        return OllamaTeacher(rest, samples=kw.get("samples", 3), temperature=kw.get("temperature", 0.7))
    if kind == "mock":
        name, _, noise = (rest or "a").partition("@")
        return MockTeacher(name, float(noise or 0.0))
    raise ValueError(f"未知老师 {spec!r}（可用：jev / ollama:<模型> / mock:<名字>[@噪声]）")
