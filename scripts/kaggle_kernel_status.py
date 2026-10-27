"""Check the FANUC kernel's Kaggle status and latest output summary.

Uses the same vault authentication as scripts/kaggle_push_kernel.py
(dynamic_credentials surrogate, host www.kaggle.com only; the real token
never appears in arguments, files, or logs).

Usage:
  python3 scripts/kaggle_kernel_status.py            # status + output files
  python3 scripts/kaggle_kernel_status.py --log-tail # also print log tail

Endpoints (verified 2026-10-26 against the live API):
  GET /api/v1/kernels/status?userName=<u>&kernelSlug=<s>  -> {"status": ...}
  GET /api/v1/kernels/output?userName=<u>&kernelSlug=<s>  -> {"files": [...], "log": ...}
The path forms (/kernels/<u>/<s>/status) return 403/404; use the query forms.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request, read_json_response

BASE = "https://www.kaggle.com/api/v1"
HOSTS = ["www.kaggle.com"]
USER = "pascalcoimbra"
SLUG = "fanuc-conveyor-pick-place"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        raise RuntimeError("redirect refused; check the endpoint")


def get(path: str):
    req = urllib.request.Request(BASE + path, method="GET")
    add_surrogate_to_request(req, "custom.kaggle", allowed_hosts=HOSTS)
    try:
        with urllib.request.build_opener(NoRedirect()).open(req, timeout=60) as resp:
            return resp.status, read_json_response(resp)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")[:1000]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-tail", action="store_true", help="print the last log lines")
    parser.add_argument("--tail-chars", type=int, default=6000)
    parser.add_argument("--slug", default=SLUG, help="kernel slug (default: the training kernel)")
    args = parser.parse_args()

    code, status = get(f"/kernels/status?userName={USER}&kernelSlug={args.slug}")
    print(json.dumps({"http": code, "status": status}, ensure_ascii=False))

    code, output = get(f"/kernels/output?userName={USER}&kernelSlug={args.slug}")
    if isinstance(output, dict):
        files = [
            {key: item.get(key) for key in ("fileName", "size", "url") if key in item}
            for item in output.get("files", [])
        ]
        print(json.dumps({"http": code, "files": files}, ensure_ascii=False))
        if args.log_tail and output.get("log"):
            try:
                entries = json.loads(output["log"])
                text = "\n".join(str(entry.get("data", "")) for entry in entries)
            except (ValueError, AttributeError):
                text = str(output["log"])
            print("--- log tail ---")
            print(text[-args.tail_chars:])
    else:
        print(json.dumps({"http": code, "output": str(output)[:1000]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
