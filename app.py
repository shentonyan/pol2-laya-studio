# -*- coding: utf-8 -*-
"""
PoL2 Laya Studio — 本地实时推理服务
    python app.py              # 默认 http://127.0.0.1:7860 ，自动打开浏览器
    python app.py --port 8000 --no-browser
页面文件在 docs/（与 GitHub Pages 静态演示共用同一套前端）。
"""
import argparse
import json
import os
import threading
import time
import traceback
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("USE_TF", "0")  # 避免 transformers 探测 TensorFlow 时卡住

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "docs"
MODELS = ("english", "multilingual", "typed-decisions")


def load_router():
    """加载 laya Router，三个 checkpoint 常驻显存/内存。"""
    import torch
    from laya import Router

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"[studio] device={device} ({gpu})，正在加载三个 checkpoint …")
    router = Router(preload=True, device=device)
    info = {"device": device, "gpu": gpu, "torch": torch.__version__}
    return router, info


def predict(router, state, questions, model=None):
    """返回 (result, 毫秒)。model 为 None 或 'auto' 时由 Router 自动路由。"""
    kw = {} if not model or model == "auto" else {"model": model}
    t0 = time.perf_counter()
    try:
        res = router.predict(state, questions, **kw)
    except (TypeError, AttributeError, KeyError):
        if isinstance(state, str):  # 部分版本只接受 dict 形式的 state
            res = router.predict({"text": state}, questions, **kw)
        else:
            raise
    return res, (time.perf_counter() - t0) * 1000


def warmup(router):
    q = {"x": {"type": "noul", "instructions": "Is this a test?"}}
    for m in MODELS:
        try:
            predict(router, {"body": "warm up"}, q, m)
        except Exception as e:  # noqa: BLE001
            print(f"[studio] 预热 {m} 失败（可忽略）: {e}")


def _default(o):
    try:
        return float(o)
    except Exception:  # noqa: BLE001
        return str(o)


def main():
    ap = argparse.ArgumentParser(description="PoL2 Laya Studio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    router, info = load_router()
    print("[studio] 预热中 …")
    warmup(router)
    lock = threading.Lock()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(WEB), **kw)

        def log_message(self, *a):
            pass

        def _json(self, code, body):
            data = json.dumps(body, ensure_ascii=False, default=_default).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/api/status":
                return self._json(200, info)
            return super().do_GET()

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
                if self.path != "/api/predict":
                    return self._json(404, {"error": "not found"})
                with lock:
                    res, ms = predict(router, req["state"], req["questions"], req.get("model"))
                self._json(200, {"result": res, "ms": ms, "model": req.get("model")})
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

    url = f"http://{args.host}:{args.port}"
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[studio] 就绪 → {url}   Ctrl+C 退出")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[studio] 已退出")


if __name__ == "__main__":
    main()
