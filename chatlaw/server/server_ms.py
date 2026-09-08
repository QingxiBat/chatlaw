import os
import subprocess
import threading
import time

import mindspore as ms
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from mindspore import context
from mindnlp.transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from chatlaw.configuration import config
from chatlaw.dataloader import download_resources
from chatlaw.network.http_transport import (
    read_json_request,
    read_pickle_request,
    send_json_response,
    send_pickled_frame,
)
from launcher import get_resources_path


HEARTBEAT_PORT = config.HEARTBEAT_PORT
DATA_PORT = config.DATA_PORT
HEALTH_PATH = getattr(config, "HEALTH_PATH", "/health")
HEARTBEAT_PATH = getattr(config, "HEARTBEAT_PATH", "/heartbeat")
INFER_PATH = getattr(config, "INFER_PATH", "/infer")

stop_flag = threading.Event()
state_lock = threading.Lock()
last_heartbeat_time = 0.0
alive = False

device_map = "auto"
USE_ORANGE_PI = False


class StopNow(StoppingCriteria):
    def __init__(self, stop_event):
        super().__init__()
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_event.is_set()


try:
    user = subprocess.run(
        ["whoami"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    context.set_context(mode=context.GRAPH_MODE)

    if user == "HwHiAiUser":
        context.set_context(jit_config={"jit_level": "O2"})
        ms.set_device("Ascend")
        device_map = "npu"
        USE_ORANGE_PI = True
    else:
        ms.set_device("Ascend")
        device_map = "npu"
except Exception:
    try:
        context.set_context(mode=context.GRAPH_MODE)
        ms.set_device("GPU")
        device_map = "cuda"
    except Exception as exc:
        raise RuntimeError("No compatible backend (Ascend/GPU) Found") from exc


print("[Model] Loading Qwen model...")
resource_path = get_resources_path()
download_resources(resource_type="tokenizer")
download_resources(resource_type="llm")

tokenizer_path = os.path.join(resource_path, "tokenizer").replace("\\", "/")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

model_path = os.path.join(resource_path, "llm").replace("\\", "/")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    dtype="auto",
    device_map=device_map,
)
print("[Model] Qwen loaded successfully.")


def _set_session_state(*, active=None, heartbeat=None, clear_stop=False):
    global alive, last_heartbeat_time
    with state_lock:
        if active is not None:
            alive = active
        if heartbeat is not None:
            last_heartbeat_time = heartbeat
        if clear_stop:
            stop_flag.clear()


def heartbeat_watchdog():
    while True:
        try:
            time.sleep(0.5)
            with state_lock:
                session_active = alive
                last_seen = last_heartbeat_time

            if session_active and last_seen and (time.time() - last_seen > config.HB_TIMEOUT):
                print("[Heartbeat] Timeout, stopping current session.")
                stop_flag.set()
                _set_session_state(active=False)
        except Exception as exc:
            print(f"[Heartbeat] Watchdog error: {exc}")
            time.sleep(1)


class ChatLawHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class BaseHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # noqa: A003
        return

    def _not_found(self):
        send_json_response(self, {"ok": False, "error": "not found"}, status=404)


class HeartbeatHandler(BaseHandler):
    def do_POST(self):
        if self.path not in (HEARTBEAT_PATH, "/health"):
            return self._not_found()

        if self.path == "/health":
            return send_json_response(self, {"ok": True, "service": "heartbeat"})

        try:
            payload = read_json_request(self)
        except Exception as exc:
            return send_json_response(self, {"ok": False, "error": str(exc)}, status=400)

        action = str(payload.get("action", "PING")).upper()

        if action == "PING":
            stop_flag.clear()
            _set_session_state(active=True, heartbeat=time.time())
            return send_json_response(self, {"ok": True, "reply": "PONG"})

        if action == "STOP":
            stop_flag.set()
            _set_session_state(active=False)
            return send_json_response(self, {"ok": True, "reply": "STOPPED"})

        return send_json_response(self, {"ok": False, "error": "unsupported action"}, status=400)


def _prepare_model_inputs(model_inputs):
    fixed = {}
    for key, value in model_inputs.items():
        if isinstance(value, np.ndarray):
            value = ms.Tensor(value.astype(np.int32))
        fixed[key] = value.to(model.device)
    return fixed


def stream_generate(model_inputs):
    fixed = _prepare_model_inputs(model_inputs)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    stopping = StoppingCriteriaList([StopNow(stop_flag)])

    if USE_ORANGE_PI:
        gen_kwargs = dict(
            **fixed,
            streamer=streamer,
            max_new_tokens=4096,
            temperature=0.7,
            do_sample=True,
            stopping_criteria=stopping,
        )
    else:
        gen_kwargs = dict(
            **fixed,
            streamer=streamer,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            stopping_criteria=stopping,
        )

    def _run_generate():
        try:
            model.generate(**gen_kwargs)
        except Exception as exc:
            print(f"[Data] Generation error: {exc}")

    thread = threading.Thread(target=_run_generate, daemon=True)
    thread.start()

    try:
        for text in streamer:
            if stop_flag.is_set():
                print("[Data] Generation stopped by client request.")
                break
            if text.strip():
                yield text
    finally:
        thread.join(timeout=1)
        stop_flag.clear()


class DataHandler(BaseHandler):
    def do_POST(self):
        if self.path == "/health":
            return send_json_response(self, {"ok": True, "service": "data"})

        if self.path != INFER_PATH:
            return self._not_found()

        try:
            model_inputs = read_pickle_request(self)
        except Exception as exc:
            return send_json_response(self, {"ok": False, "error": str(exc)}, status=400)

        stop_flag.clear()
        _set_session_state(active=True, heartbeat=time.time(), clear_stop=True)

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            for chunk in stream_generate(model_inputs):
                send_pickled_frame(self, chunk)

            send_pickled_frame(self, "<END>")
            print("[Data] Stream finished.")

        except Exception as exc:
            err_msg = f"[ServerError] {type(exc).__name__}: {exc}"
            print(err_msg)
            try:
                send_pickled_frame(self, err_msg)
                send_pickled_frame(self, "<END>")
            except Exception:
                pass

        finally:
            _set_session_state(active=False)
            stop_flag.clear()


def heartbeat_server():
    server = ChatLawHTTPServer(("0.0.0.0", HEARTBEAT_PORT), HeartbeatHandler)
    print(f"[Heartbeat] HTTP listening on port {HEARTBEAT_PORT}")
    server.serve_forever()


def data_server():
    server = ChatLawHTTPServer(("0.0.0.0", DATA_PORT), DataHandler)
    print(f"[Data] HTTP listening on port {DATA_PORT}")
    server.serve_forever()


if __name__ == "__main__":
    threading.Thread(target=heartbeat_watchdog, daemon=True).start()
    threading.Thread(target=heartbeat_server, daemon=True).start()
    data_server()
