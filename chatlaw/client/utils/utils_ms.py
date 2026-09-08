import pickle
import time

from chatlaw.network.http_transport import close_quietly, post_json, post_pickle


def heartbeat_client_ms(server_ip, hb_port, hb_interval, hb_timeout,
                        alive_flag, stop_event, recv_exact_fn, heartbeat_path="/heartbeat"):
    """
    Send periodic HTTP POST heartbeat requests to the server.
    """
    print("[HB] HTTP heartbeat started.", flush=True)
    last_ok = time.time()

    try:
        while alive_flag():
            action = "STOP" if stop_event.is_set() else "PING"

            try:
                conn, resp = post_json(
                    server_ip,
                    hb_port,
                    heartbeat_path,
                    {"action": action},
                    timeout=3.0,
                )
                try:
                    if resp.status != 200:
                        body = resp.read().decode("utf-8", "ignore")
                        raise ConnectionError(
                            body or f"heartbeat failed with HTTP {resp.status}"
                        )
                    resp.read()
                finally:
                    close_quietly(resp, conn)

                if action == "STOP":
                    print("[HB] STOP sent.")
                    break

                last_ok = time.time()
                time.sleep(hb_interval)

            except Exception as e:
                if time.time() - last_ok > hb_timeout:
                    print(f"[HB] timeout, marking dead: {e}")
                    stop_event.set()
                    break
                time.sleep(min(0.2, hb_interval))

    finally:
        print("[HB] Closed.")


def stream_from_server_ms(
        server_ip, data_port,
        recv_exact_fn,
        stop_event,
        input_tensor,
        infer_path="/infer"):
    """
    Stream inference chunks from the HTTP POST inference endpoint.
    """
    conn = None
    resp = None
    timeout_count = 0
    MAX_TIMEOUT = 3

    try:
        conn, resp = post_pickle(
            server_ip,
            data_port,
            infer_path,
            input_tensor,
            timeout=5.0,
        )

        if resp.status != 200:
            body = resp.read().decode("utf-8", "ignore")
            raise ConnectionError(body or f"unexpected HTTP {resp.status}")

        while True:
            suppress_output = stop_event.is_set()

            try:
                hdr = recv_exact_fn(resp, 8, timeout=2.0)
                if not hdr:
                    break

                data_len = int.from_bytes(hdr, "big")
                data = recv_exact_fn(resp, data_len, timeout=2.0)
                timeout_count = 0

            except Exception as e:
                if "timed out" in str(e).lower():
                    timeout_count += 1
                    if timeout_count >= MAX_TIMEOUT:
                        yield "[ClientError] Connection timed out"
                        break
                    continue
                raise

            chunk = pickle.loads(data)
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8", "ignore")

            if str(chunk).strip() == "<END>":
                yield "<END>"
                break

            if not suppress_output:
                yield chunk

    except Exception as e:
        yield f"[ClientError] {e}"

    finally:
        close_quietly(resp, conn)
        print("[DATA] Closed.")
