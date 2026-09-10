#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ChatLaw 仿真推理服务端（**仅用于客户端联调**，不是生产服务端）。

用途：在没有部署真实 `chatlaw/server/server_pt.py` 的情况下，验证安卓端的
「连接 → 心跳 → /infer 流式输出 → 中断」整条链路。它实现的 HTTP 协议与
`chatlaw/network/http_transport.py` 完全一致：

- 6005：`POST /health`、`POST /heartbeat`（JSON，统计 PING/STOP 次数）
- 6006：`POST /health`；`POST /infer`
        —— 读取 pickle 请求体并打印其结构（校验客户端负载格式），
           随后按「8 字节大端长度 + pickle」分帧慢速吐字，最后发 `<END>`

用法（推荐配合 adb reverse，免去配置手机与电脑同网段）：

    # 1) 本机启动仿真服务端（默认 6005/6006）
    python scripts/fake_chatlaw_server.py

    # 2) 让手机的 127.0.0.1:6005/6006 转发到本机
    adb reverse tcp:6005 tcp:6005
    adb reverse tcp:6006 tcp:6006

    # 3) APP 里把「服务器 IP」设为 127.0.0.1，然后发问

可调参数：`--interval` 每块吐字间隔秒数（默认 1.2，放慢便于观察流式渲染）。
"""

import argparse
import json
import pickle
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 逐块吐出的文本（模拟模型流式输出）
CHUNKS = [
    "根据", "《中华人民共和国消防法》", "第五十一条", "，", "消防救援机构",
    "有权根据需要", "封闭火灾现场", "，负责调查火灾原因。", "\n\n",
    "结合《中华人民共和国民法典》", "第一千二百三十九条", "，",
    "过失导致他人财产损失的", "，应当承担相应侵权责任。", "\n\n",
    "以上为仿真服务端输出，", "用于验证安卓端流式渲染。",
]

stats = {"pings": 0, "stops": 0, "infer": 0}
chunk_interval = 1.2


def frame(chunk):
    data = pickle.dumps(chunk)
    return len(data).to_bytes(8, "big") + data


class Base(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # 静默访问日志
        pass

    def send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()


class HeartbeatHandler(Base):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001
            payload = {}
        action = str(payload.get("action", "PING")).upper()
        if action == "STOP":
            stats["stops"] += 1
            print(f"[hb] STOP #{stats['stops']}", flush=True)
            return self.send_json({"ok": True, "reply": "STOPPED"})
        stats["pings"] += 1
        if self.path == "/health":
            return self.send_json({"ok": True, "service": "heartbeat"})
        return self.send_json({"ok": True, "reply": "PONG"})


class DataHandler(Base):
    def do_POST(self):
        if self.path == "/health":
            return self.send_json({"ok": True, "service": "data"})
        if self.path != "/infer":
            return self.send_json({"ok": False, "error": "not found"}, status=404)

        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        stats["infer"] += 1
        print(
            f"[infer] #{stats['infer']} body={len(body)}B content-length={length}",
            flush=True,
        )
        try:
            payload = pickle.loads(body)
            keys = sorted(payload) if hasattr(payload, "keys") else type(payload).__name__
            shapes = {
                k: getattr(v, "shape", None)
                for k, v in (payload.items() if hasattr(payload, "items") else [])
            }
            print(f"[infer] 负载 keys={keys} shapes={shapes}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[infer] 反序列化失败: {type(exc).__name__}: {exc}", flush=True)

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            for chunk in CHUNKS:
                self.wfile.write(frame(chunk))
                self.wfile.flush()
                time.sleep(chunk_interval)
            self.wfile.write(frame("<END>"))
            self.wfile.flush()
            print("[infer] done", flush=True)
        except (ConnectionError, OSError) as exc:
            print(f"[infer] 客户端提前断开（已中止）: {type(exc).__name__}", flush=True)


def serve(port, handler):
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    server.serve_forever()


def main():
    global chunk_interval
    parser = argparse.ArgumentParser(description="ChatLaw 仿真推理服务端（联调用）")
    parser.add_argument("--heartbeat-port", type=int, default=6005)
    parser.add_argument("--data-port", type=int, default=6006)
    parser.add_argument("--interval", type=float, default=1.2, help="每块吐字间隔（秒）")
    args = parser.parse_args()
    chunk_interval = args.interval

    print(
        f"[fake] heartbeat :{args.heartbeat_port}  data :{args.data_port}"
        f"  吐字间隔 {chunk_interval}s",
        flush=True,
    )
    threading.Thread(
        target=serve, args=(args.heartbeat_port, HeartbeatHandler), daemon=True
    ).start()
    serve(args.data_port, DataHandler)


if __name__ == "__main__":
    main()
