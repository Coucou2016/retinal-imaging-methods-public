"""Push HEAD to GitHub via Git Data API when HTTPS git:443 fails.

Preserves local commit SHAs. Non-force ref update only.
Requires GH_TOKEN or `gh auth token`.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

REPO = "Coucou2016/retinal-imaging-methods-public"
ROOT = r"E:\Projects\20260522-retinal-imaging"
SAFE = ["-c", "safe.directory=E:/Projects/20260522-retinal-imaging"]
API = "https://api.github.com"


def log(msg: str) -> None:
    print(msg, flush=True)


def token() -> str:
    t = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if t:
        return t.strip()
    try:
        return subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    except Exception as e:
        raise SystemExit(f"Need GH_TOKEN or gh auth token: {e}") from e


TOKEN = token()


def git(*args: str, binary: bool = False) -> bytes | str:
    out = subprocess.check_output(["git", *SAFE, *args], cwd=ROOT)
    return out if binary else out.decode().rstrip("\n")


def api(method: str, path: str, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {TOKEN}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "retinal-imaging-push",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = resp.read().decode()
            return resp.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            parsed = json.loads(body) if body else {}
        except Exception:
            parsed = {"raw": body}
        return e.code, parsed


def fmt_date(ts: int, tz: str) -> str:
    sign = 1 if tz[0] == "+" else -1
    hh, mm = int(tz[1:3]), int(tz[3:5])
    offset = timezone(sign * timedelta(hours=hh, minutes=mm))
    return datetime.fromtimestamp(ts, tz=offset).isoformat()


def parse_ident(s: str):
    left, right = s.rsplit(">", 1)
    name, email = (left + ">").rsplit(" <", 1)
    email = email[:-1]
    ts, tz = right.strip().split()
    return name, email, int(ts), tz


def main() -> int:
    code, ref = api("GET", f"/repos/{REPO}/git/ref/heads/main")
    if code != 200:
        log(f"get ref failed: {code} {ref}")
        return 1
    remote_sha = ref["object"]["sha"]
    local = git("rev-parse", "HEAD")
    log(f"remote main={remote_sha}")
    log(f"local  HEAD={local}")
    if remote_sha == local:
        log("already up to date")
        return 0

    base = remote_sha
    # Ensure local contains remote tip (fast-forward only).
    mb = git("merge-base", remote_sha, local)
    if mb != remote_sha:
        log(f"not a fast-forward (merge-base={mb}); refuse non-force push")
        return 1

    commits = git("rev-list", "--reverse", f"{base}..HEAD").splitlines()
    if not commits:
        log("no commits to upload")
        return 0
    log(f"commits to upload: {len(commits)}")

    obj_lines = git("rev-list", "--objects", f"{base}..HEAD").splitlines()
    shas = []
    for line in obj_lines:
        if not line.strip():
            continue
        shas.append(line.split(" ", 1)[0])
    ordered = list(dict.fromkeys(shas))
    typed = [(sha, git("cat-file", "-t", sha)) for sha in ordered]
    blobs = [s for s, t in typed if t == "blob"]
    trees = [s for s, t in typed if t == "tree"]
    log(f"objects: {len(blobs)} blobs, {len(trees)} trees")

    for sha in blobs:
        raw = git("cat-file", "-p", sha, binary=True)
        code, res = api(
            "POST",
            f"/repos/{REPO}/git/blobs",
            {
                "content": base64.b64encode(raw).decode("ascii"),
                "encoding": "base64",
            },
        )
        if code not in (200, 201):
            gcode, _ = api("GET", f"/repos/{REPO}/git/blobs/{sha}")
            if gcode == 200:
                log(f"skip blob {sha[:12]} (already remote)")
                continue
            log(f"blob failed {sha}: {code} {res}")
            return 1
        if res.get("sha") and res.get("sha") != sha:
            log(f"blob sha mismatch {sha} vs {res.get('sha')}")
            return 1
        log(f"blob {sha[:12]} ({len(raw)} bytes)")

    remaining = set(trees)
    guard = 0
    while remaining:
        guard += 1
        if guard > 10000:
            log(f"tree resolve stuck; left={remaining}")
            return 1
        progress = False
        for sha in list(remaining):
            entries = []
            ready = True
            for line in git("ls-tree", sha).splitlines():
                mode, etype, rest = line.split(" ", 2)
                esha, path = rest.split("\t", 1)
                if etype == "tree" and esha in remaining:
                    ready = False
                    break
                entries.append(
                    {"path": path, "mode": mode, "type": etype, "sha": esha}
                )
            if not ready:
                continue
            code, res = api("POST", f"/repos/{REPO}/git/trees", {"tree": entries})
            if code not in (200, 201):
                gcode, _ = api("GET", f"/repos/{REPO}/git/trees/{sha}")
                if gcode == 200:
                    log(f"skip tree {sha[:12]} (already remote)")
                    remaining.remove(sha)
                    progress = True
                    continue
                log(f"tree failed {sha}: {code} {res}")
                return 1
            elif res.get("sha") != sha:
                log(f"tree sha mismatch {sha} vs {res.get('sha')}: {res}")
                return 1
            else:
                log(f"tree {sha[:12]} ({len(entries)} entries)")
            remaining.remove(sha)
            progress = True
        if not progress:
            for sha in list(remaining):
                gcode, _ = api("GET", f"/repos/{REPO}/git/trees/{sha}")
                if gcode == 200:
                    log(f"skip tree {sha[:12]} (already remote)")
                    remaining.remove(sha)
                    progress = True
            if not progress:
                log(f"cannot resolve trees: {remaining}")
                return 1

    for sha in commits:
        text = git("cat-file", "-p", sha)
        lines = text.splitlines()
        tree = None
        parents = []
        author_line = committer_line = ""
        i = 0
        while i < len(lines) and lines[i]:
            line = lines[i]
            if line.startswith("tree "):
                tree = line[5:]
            elif line.startswith("parent "):
                parents.append(line[7:])
            elif line.startswith("author "):
                author_line = line[7:]
            elif line.startswith("committer "):
                committer_line = line[10:]
            i += 1
        message = git("log", "-1", "--format=%B", sha)
        if not message.endswith("\n"):
            message += "\n"
        an, ae, at, atz = parse_ident(author_line)
        cn, ce, ct, ctz = parse_ident(committer_line)
        payload = {
            "message": message,
            "tree": tree,
            "parents": parents,
            "author": {"name": an, "email": ae, "date": fmt_date(at, atz)},
            "committer": {"name": cn, "email": ce, "date": fmt_date(ct, ctz)},
        }
        code, res = api("POST", f"/repos/{REPO}/git/commits", payload)
        if code not in (200, 201):
            gcode, _ = api("GET", f"/repos/{REPO}/git/commits/{sha}")
            if gcode == 200:
                log(f"skip commit {sha[:12]} (already remote)")
                continue
            log(f"commit failed {sha}: {code} {res}")
            return 1
        if res.get("sha") != sha:
            log(f"commit sha mismatch want={sha} got={res.get('sha')} res={res}")
            return 1
        log(f"commit {sha[:12]} ok")

    tip = commits[-1]
    code, res = api(
        "PATCH",
        f"/repos/{REPO}/git/refs/heads/main",
        {"sha": tip, "force": False},
    )
    if code != 200:
        log(f"update ref failed: {code} {res}")
        return 1
    log(f"updated main -> {tip}")
    code, ref = api("GET", f"/repos/{REPO}/git/ref/heads/main")
    remote = ref["object"]["sha"]
    local = git("rev-parse", "HEAD")
    log(f"confirm remote={remote} local={local}")
    if remote == local == tip:
        log("SUCCESS")
        return 0
    log("MISMATCH")
    return 1


if __name__ == "__main__":
    sys.exit(main())
