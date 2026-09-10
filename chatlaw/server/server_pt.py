import os
import threading
import time

import torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from transformers import (
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
from chatlaw.network.tensor_utils import to_device_tensors
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


class StopFlagCriteria(StoppingCriteria):
    def __init__(self, flag):
        self.flag = flag

    def __call__(self, input_ids, scores, **kwargs):
        return self.flag.is_set()


print("[Backend] Checking available backend hardware...")

device = None
device_map = "auto"
use_npu = False

if hasattr(torch, "npu"):
    try:
        if torch.npu.is_available():
            use_npu = True
            print("[Backend] Ascend NPU available.")
    except Exception:
        pass

if use_npu:
    try:
        import torch_npu  # pylint: disable=unused-import

        device = "npu"
        device_map = {"": torch.npu.current_device()}
        print("[Backend] Using Ascend NPU backend.")
    except ImportError as exc:
        raise RuntimeError(
            "Ascend NPU detected but torch_npu package is missing."
        ) from exc
elif torch.cuda.is_available():
    device = "cuda"
    device_map = "auto"
    print("[Backend] Using NVIDIA GPU.")
else:
    raise RuntimeError("No compatible backend (Ascend/GPU) Found")


print("[Model] Loading Qwen model...")
resource_path = get_resources_path()
download_resources(resource_type="tokenizer")
download_resources(resource_type="llm")

tokenizer_path = os.path.join(resource_path, "tokenizer").replace("\\", "/")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

model_path = os.path.join(resource_path, "llm").replace("\\", "/")
if device_map == "npu":
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype="auto",
        device_map=device_map,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        device_map="auto",
        dtype="auto",
        attn_implementation="sdpa",
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
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


def stream_generate(model_inputs):
    # 客户端（含 Android 端）可能发送 numpy 数组而非 torch 张量，统一在此转换。
    input_ids, attention_mask = to_device_tensors(model_inputs, model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        streamer=streamer,
        max_new_tokens=4096,
        temperature=0.7,
        do_sample=True,
        stopping_criteria=StoppingCriteriaList([StopFlagCriteria(stop_flag)]),
    )

    def _run_generate():
        try:
            if device == "cuda":
                with torch.inference_mode(), torch.autocast("cuda"):
                    model.generate(**generation_kwargs)
            else:
                with torch.inference_mode():
                    model.generate(**generation_kwargs)
        except Exception as exc:
            print(f"[Data] Generation error: {exc}")

    thread = threading.Thread(target=_run_generate, daemon=True)
    thread.start()

    try:
        for new_text in streamer:
            if stop_flag.is_set():
                print("[Data] Generation stopped by client request.")
                break
            if new_text.strip():
                yield new_text
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
