"""Applies one netem setting to both relay containers, on request from party 0's tab.

GET /shape?delay_ms=<float>&rate_mbit=<int>&loss_pct=<float> sets every relay's eth0 to
`netem delay <delay_ms>ms rate <rate_mbit>mbit loss <loss_pct>%` (netem's default 1000-packet queue)
and answers with what `tc qdisc show` then reports. Each relay carries ONE direction, so delay_ms is
the one-way delay and the round trip is twice it.

Bound to 127.0.0.1, accepts only these three numbers in fixed ranges, and runs only fixed commands -
it exists for the length of a measurement. CORS is open because the bench page is another origin.
"""

import json
import re
import subprocess
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

RELAYS = ("qsh-netem-ab", "qsh-netem-ba")
RESULTS = Path(__file__).resolve().parent / "results"
LIMITS = {"delay_ms": (0.0, 500.0), "rate_mbit": (1, 1000), "loss_pct": (0.0, 10.0)}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path != "/shape":
            return self.reply(404, {"error": "only /shape"})
        q = parse_qs(url.query)
        try:
            v = {k: (int if k == "rate_mbit" else float)(q[k][0]) for k in LIMITS}
        except (KeyError, ValueError):
            return self.reply(400, {"error": "need delay_ms, rate_mbit, loss_pct"})
        for k, (lo, hi) in LIMITS.items():
            if not lo <= v[k] <= hi:
                return self.reply(400, {"error": f"{k} out of range"})
        shown = {}
        for relay in RELAYS:
            subprocess.run(["docker", "exec", relay, "tc", "qdisc", "replace", "dev", "eth0", "root",
                            "netem", "delay", f"{v['delay_ms']}ms", "rate", f"{v['rate_mbit']}mbit",
                            "loss", f"{v['loss_pct']}%"], check=True)
            shown[relay] = subprocess.run(["docker", "exec", relay, "tc", "qdisc", "show", "dev", "eth0"],
                                          check=True, capture_output=True, text=True).stdout.strip()
        self.reply(200, {"applied": v, "tc": shown})

    def do_POST(self):
        # POST /save?name=<lowercase-name> with a JSON body writes results/<name>.json, and nothing else:
        # the name is a fixed pattern and the directory is this harness's own results/.
        url = urlparse(self.path)
        name = parse_qs(url.query).get("name", [""])[0]
        if url.path != "/save" or not re.fullmatch(r"[a-z0-9-]{1,64}", name):
            return self.reply(400, {"error": "POST /save?name=<a-z0-9->"})
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            data = json.loads(body)
        except ValueError:
            return self.reply(400, {"error": "body is not JSON"})
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{name}.json").write_text(json.dumps(data, indent=1) + "\n")
        self.reply(200, {"saved": f"results/{name}.json", "bytes": len(body)})

    def reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8779), Handler).serve_forever()
