#!/usr/bin/env python3
"""Front proxy + supervisor for two-process TP2 ninfer-serve (stdlib only).

* Starts rank 0 and rank 1 (one GPU each), waits until both answer /v1/models.
* Runs up to --max-inflight generation requests at once (the ranks need --max-concurrency of at
  least that); each request body is sent unchanged to both ranks, in the same order for both, so
  their FIFOs agree. Rank 0's response is relayed to the client; rank 1's is drained and discarded.
* A client disconnect closes only the rank-0 upstream connection. The engines agree on
  cancellation once per GPU unit (runtime/engine/tp_lockstep.h), so rank 1 stops at the same unit.
* Watchdog: if either process exits, or a request is in flight and the lockstep unit counters do
  not advance for --stall-seconds, both processes are killed and restarted and the client gets a
  503 (or a closed stream if headers were already sent).
"""
import argparse, http.client, http.server, json, mmap, os, signal, socket, socketserver
import struct, subprocess, sys, threading, time

HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "te", "trailer",
       "upgrade", "proxy-connection"}


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "[tp2-proxy]", msg, flush=True)


class Supervisor:
    def __init__(self, args):
        self.args = args
        self.procs = [None, None]
        self.ready = threading.Event()
        self.slots = threading.BoundedSemaphore(args.max_inflight)  # generation requests in flight
        self.inflight = 0
        self.dispatch_lock = threading.Lock()  # both ranks must receive requests in the same order
        self.restart_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.generation = 0
        self.restarts = 0
        self.waiting = 0
        self.count_lock = threading.Lock()

    # ---- process management -------------------------------------------------------------
    def rank_cmd(self, r):
        a = self.args
        return [a.binary, f"{a.model_prefix}.rank{r}.ninfer", "--host", "127.0.0.1",
                "--port", str(a.rank_ports[r]), "--api-key", a.probe_key] + a.serve_args

    def rank_env(self, r):
        env = dict(os.environ)
        env.update({"CUDA_VISIBLE_DEVICES": str(self.args.gpus[r]), "NINFER_TP_RANK": str(r),
                    "NINFER_TP_ID_FILE": self.args.id_file,
                    "NINFER_TP_LOCKSTEP_FILE": self.args.lockstep_file,
                    "NINFER_TP_LOCKSTEP_TIMEOUT_S": str(self.args.lockstep_timeout),
                    "NINFER_TP_MAILBOX_FILE": self.args.lockstep_file + ".mailbox"})
        if self.args.nccl_p2p == "off":
            env["NCCL_P2P_DISABLE"] = "1"   # GPU-to-GPU direct access is broken on the dev machine
        if r == 1:
            for kv in self.args.rank1_env:   # test hooks only
                k, v = kv.split("=", 1)
                env[k] = v
        return env

    def start(self):
        with self.state_lock:
            self.ready.clear()
            for f in (self.args.id_file, self.args.id_file + ".tmp", self.args.lockstep_file):
                try:
                    os.unlink(f)
                except FileNotFoundError:
                    pass
            self.generation += 1
            for r in (0, 1):
                logf = open(f"{self.args.log_dir}/tp2_rank{r}.log", "ab")
                self.procs[r] = subprocess.Popen(self.rank_cmd(r), env=self.rank_env(r),
                                                 stdout=logf, stderr=subprocess.STDOUT,
                                                 stdin=subprocess.DEVNULL, start_new_session=True)
            log(f"started ranks gen={self.generation} pids={[p.pid for p in self.procs]}")
        deadline = time.time() + self.args.start_timeout
        while time.time() < deadline:
            if any(p.poll() is not None for p in self.procs):
                log("a rank exited during startup")
                return False
            if all(self.probe(r) for r in (0, 1)):
                log(f"both ranks ready gen={self.generation}")
                self.ready.set()
                return True
            time.sleep(1)
        log("startup timed out")
        return False

    def kill_all(self):
        with self.state_lock:
            for p in self.procs:
                if p is not None and p.poll() is None:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            for p in self.procs:
                if p is not None:
                    try:
                        p.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass

    def restart(self, reason, gen=None):
        with self.restart_lock:
            if gen is not None and self.generation != gen:
                return  # another in-flight request already restarted this generation
            log(f"RESTART both ranks: {reason}")
            self.restarts += 1
            self.ready.clear()
            self.kill_all()
            time.sleep(2)
            while not self.start():
                self.kill_all()
                time.sleep(10)

    def probe(self, r):
        try:
            c = http.client.HTTPConnection("127.0.0.1", self.args.rank_ports[r], timeout=5)
            c.request("GET", "/v1/models", headers={"Authorization": f"Bearer {self.args.probe_key}"})
            ok = c.getresponse().status == 200
            c.close()
            return ok
        except OSError:
            return False

    def alive(self):
        return all(p is not None and p.poll() is None for p in self.procs)

    # ---- lockstep progress ----------------------------------------------------------------
    def units(self):
        try:
            with open(self.args.lockstep_file, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                a, b = struct.unpack_from("<QQ", mm, 64)
                mm.close()
                return a, b
        except (OSError, ValueError, struct.error):
            return None


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    sup: Supervisor = None

    def log_message(self, fmt, *a):
        pass

    def _error(self, code, msg):
        body = json.dumps({"error": {"message": msg, "type": "tp2_proxy_error"}}).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            pass
        self.close_connection = True

    def _headers_in(self):
        return {k: v for k, v in self.headers.items() if k.lower() not in HOP and k.lower() != "host"}

    def do_GET(self):
        self._passthrough("GET", None)

    def do_DELETE(self):
        self._passthrough("DELETE", None)

    def _passthrough(self, method, body):
        sup = self.sup
        if not sup.ready.wait(timeout=sup.args.ready_wait):
            return self._error(503, "TP2 backend is restarting")
        try:
            c = http.client.HTTPConnection("127.0.0.1", sup.args.rank_ports[0], timeout=60)
            c.request(method, self.path, body=body, headers=self._headers_in())
            r = c.getresponse()
            data = r.read()
            self.send_response(r.status)
            for k, v in r.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            c.close()
        except OSError as e:
            self._error(502, f"rank0 unreachable: {e}")

    def do_POST(self):
        sup = self.sup
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        if not sup.ready.wait(timeout=sup.args.ready_wait):
            return self._error(503, "TP2 backend is restarting")
        with sup.count_lock:
            if sup.waiting >= sup.args.max_waiting + sup.args.max_inflight:
                return self._error(429, "too many queued requests")
            sup.waiting += 1
        try:
            if not sup.slots.acquire(timeout=sup.args.pending_timeout):
                return self._error(503, "request expired while waiting for admission")
            try:
                if not sup.ready.is_set() or not sup.alive():
                    return self._error(503, "TP2 backend is restarting")
                with sup.count_lock:
                    sup.inflight += 1
                try:
                    self._generate(body)
                finally:
                    with sup.count_lock:
                        sup.inflight -= 1
            finally:
                sup.slots.release()
        finally:
            with sup.count_lock:
                sup.waiting -= 1

    def _generate(self, body):
        sup = self.sup
        gen = sup.generation
        headers = self._headers_in()
        headers["Content-Length"] = str(len(body))
        conns = [http.client.HTTPConnection("127.0.0.1", sup.args.rank_ports[r], timeout=None)
                 for r in (0, 1)]
        with sup.dispatch_lock:
            for r in (0, 1):
                conns[r].request("POST", self.path, body=body, headers=headers)
            # Let both engines queue this request before the next one is dispatched, so the two
            # FIFOs hold the same order (the engines agree on how many to admit, not on which).
            time.sleep(sup.args.dispatch_gap)
        failure = {"reason": None}
        done = [threading.Event(), threading.Event()]
        upstream_socks = [conns[0].sock, conns[1].sock]

        def drain_rank1():
            try:
                r1 = conns[1].getresponse()
                while r1.read1(65536):
                    pass
            except (OSError, http.client.HTTPException) as e:
                failure["reason"] = failure["reason"] or f"rank1 stream error: {e}"
            finally:
                done[1].set()

        def watchdog():
            last = sup.units()
            last_t = time.time()
            while not (done[0].is_set() and done[1].is_set()):
                time.sleep(2)
                if sup.generation != gen:
                    return
                if not sup.alive():
                    failure["reason"] = failure["reason"] or "a rank process exited"
                    break
                u = sup.units()
                if u != last:
                    last, last_t = u, time.time()
                elif time.time() - last_t > sup.args.stall_seconds:
                    failure["reason"] = failure["reason"] or \
                        f"no lockstep progress for {sup.args.stall_seconds}s (units {u})"
                    break
            if failure["reason"]:
                for s in upstream_socks:
                    try:
                        s.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass

        cancel_state = {"client_gone": False}

        def client_watch():
            # A non-streaming client that times out is only noticed here: nothing is written to it
            # until rank 0 finishes. EOF on the client socket cancels rank 0 immediately.
            import select
            sock = self.connection
            while not done[0].is_set():
                try:
                    readable, _, _ = select.select([sock], [], [], 0.5)
                    if readable and sock.recv(1, socket.MSG_PEEK) == b"":
                        cancel_state["client_gone"] = True
                        # rank 0 may already have finished and http.client dropped its socket
                        s0 = conns[0].sock
                        if s0 is not None:
                            try:
                                s0.shutdown(socket.SHUT_RDWR)
                            except OSError:
                                pass
                        return
                except (OSError, ValueError):
                    return

        tc = threading.Thread(target=client_watch, daemon=True)
        tc.start()
        t1 = threading.Thread(target=drain_rank1, daemon=True)
        tw = threading.Thread(target=watchdog, daemon=True)
        t1.start(); tw.start()

        headers_sent = False
        client_gone = False
        try:
            r0 = conns[0].getresponse()
            self.send_response(r0.status)
            length = r0.getheader("Content-Length")
            for k, v in r0.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            if length is not None:
                self.send_header("Content-Length", length)
            else:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            headers_sent = True
            while True:
                chunk = r0.read1(65536)
                if not chunk:
                    break
                if not client_gone:
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except OSError:
                        client_gone = True
                        # Cancel on rank 0 only; lockstep carries it to rank 1.
                        try:
                            conns[0].sock.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                        break
        except (OSError, http.client.HTTPException) as e:
            client_gone = client_gone or cancel_state["client_gone"]
            failure["reason"] = failure["reason"] or (None if client_gone else f"rank0 stream error: {e}")
        finally:
            done[0].set()

        # Wait for rank 1 to finish the same request (bounded by the watchdog).
        t1.join()
        tw.join(timeout=5)
        for c in conns:
            c.close()
        if client_gone:
            log(f"client disconnected on {self.path}; rank0 cancelled")
        if failure["reason"] and sup.generation == gen:
            if not headers_sent:
                self._error(503, f"TP2 backend failed: {failure['reason']}")
            else:
                self.close_connection = True
            sup.restart(failure["reason"], gen)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="127.0.0.1:18881")
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model-prefix", required=True)
    ap.add_argument("--rank-ports", default="18940,18941")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--id-file", default="/tmp/ninfer_tp2.id")
    ap.add_argument("--lockstep-file", default="/dev/shm/ninfer_tp2_lockstep")
    ap.add_argument("--lockstep-timeout", type=int, default=300)
    ap.add_argument("--stall-seconds", type=int, default=180)
    ap.add_argument("--start-timeout", type=int, default=900)
    ap.add_argument("--ready-wait", type=int, default=600)
    ap.add_argument("--api-key-file", required=True,
                    help="bearer key: passed to both ranks and used for health probes")
    ap.add_argument("--max-inflight", type=int, default=1,
                    help="generation requests running at once; start the ranks with "
                         "--max-concurrency >= this")
    ap.add_argument("--dispatch-gap", type=float, default=0.005,
                    help="seconds between dispatching consecutive requests to the ranks")
    ap.add_argument("--max-waiting", type=int, default=4,
                    help="generation requests allowed to queue behind the running ones")
    ap.add_argument("--pending-timeout", type=int, default=600,
                    help="seconds a queued generation request may wait")
    ap.add_argument("--log-dir", default=".")
    ap.add_argument("--rank1-env", action="append", default=[], help="KEY=VAL for rank 1 (tests)")
    ap.add_argument("--nccl-p2p", choices=["off", "auto"], default="off",
                    help="off: export NCCL_P2P_DISABLE=1 to both ranks (default; the machine this "
                         "was developed on has broken GPU-to-GPU direct access). auto: leave the "
                         "decision to NCCL, for NVLink machines or a working PCIe P2P path")
    ap.add_argument("serve_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    a.rank_ports = [int(x) for x in a.rank_ports.split(",")]
    a.gpus = [int(x) for x in a.gpus.split(",")]
    a.probe_key = open(a.api_key_file).read().strip()
    if a.serve_args and a.serve_args[0] == "--":
        a.serve_args = a.serve_args[1:]
    sup = Supervisor(a)
    Handler.sup = sup
    host, port = a.listen.rsplit(":", 1)
    srv = Server((host, int(port)), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log(f"listening on {a.listen}")
    while not sup.start():
        sup.kill_all()
        time.sleep(10)

    def stop(*_):
        log("shutting down")
        sup.kill_all()
        os._exit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    # Idle watchdog: restart if a rank dies while no request holds the lock.
    while True:
        time.sleep(5)
        if sup.ready.is_set() and not sup.alive():
            with sup.count_lock:
                idle = sup.inflight == 0
            if idle:
                sup.restart("a rank process exited while idle")


if __name__ == "__main__":
    main()
