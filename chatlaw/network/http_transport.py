"""Shared HTTP transport helpers used by ChatLaw client and server."""

from __future__ import annotations

import http.client
import json
import pickle
from contextlib import suppress


def _set_timeout(stream, timeout):
    if timeout is None:
        return

    if hasattr(stream, "settimeout"):
        with suppress(Exception):
            stream.settimeout(timeout)
        return

    raw_sock = getattr(getattr(getattr(stream, "fp", None), "raw", None), "_sock", None)
    if raw_sock is not None and hasattr(raw_sock, "settimeout"):
        with suppress(Exception):
            raw_sock.settimeout(timeout)


def read_exact(stream, n, timeout=None):
    """Read exactly ``n`` bytes from a socket-like or file-like stream."""
    if n < 0:
        raise ValueError("n must be non-negative")

    if timeout is not None:
        _set_timeout(stream, timeout)

    reader = getattr(stream, "recv", None) or getattr(stream, "read", None)
    if reader is None:
        raise TypeError("stream must provide recv() or read()")

    buf = bytearray()
    while len(buf) < n:
        chunk = reader(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed while receiving")
        buf.extend(chunk)
    return bytes(buf)


def read_request_body(handler):
    content_length = int(handler.headers.get("Content-Length", "0") or 0)
    if content_length <= 0:
        return b""
    return handler.rfile.read(content_length)


def read_json_request(handler):
    body = read_request_body(handler)
    if not body:
        return {}
    return json.loads(body.decode("utf-8"))


def read_pickle_request(handler):
    body = read_request_body(handler)
    if not body:
        raise ValueError("missing request body")
    return pickle.loads(body)


def send_json_response(handler, payload, status=200):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(data)
    handler.wfile.flush()


def send_text_response(handler, text, status=200):
    data = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(data)
    handler.wfile.flush()


def send_pickled_frame(handler, chunk):
    data = pickle.dumps(chunk)
    handler.wfile.write(len(data).to_bytes(8, "big"))
    handler.wfile.write(data)
    handler.wfile.flush()


def post_json(server_ip, port, path, payload, timeout=5.0, headers=None):
    conn = http.client.HTTPConnection(server_ip, port, timeout=timeout)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request_headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
    }
    if headers:
        request_headers.update(headers)
    try:
        conn.request("POST", path, body=body, headers=request_headers)
        return conn, conn.getresponse()
    except Exception:
        close_quietly(conn)
        raise


def post_pickle(server_ip, port, path, payload, timeout=5.0, headers=None):
    conn = http.client.HTTPConnection(server_ip, port, timeout=timeout)
    body = pickle.dumps(payload)
    request_headers = {
        "Content-Type": "application/octet-stream",
        "Accept": "application/octet-stream",
    }
    if headers:
        request_headers.update(headers)
    try:
        conn.request("POST", path, body=body, headers=request_headers)
        return conn, conn.getresponse()
    except Exception:
        close_quietly(conn)
        raise


def read_pickled_frames(response, timeout=None):
    while True:
        try:
            header = read_exact(response, 8, timeout=timeout)
        except ConnectionError:
            break

        if not header:
            break

        data_len = int.from_bytes(header, "big")
        if data_len <= 0:
            continue

        data = read_exact(response, data_len, timeout=timeout)
        yield pickle.loads(data)


def close_quietly(*resources):
    for resource in resources:
        with suppress(Exception):
            if resource is not None:
                resource.close()
