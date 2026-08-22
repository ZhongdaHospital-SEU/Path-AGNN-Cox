# -*- coding: utf-8 -*-
"""Push a local commit to GitHub via REST API (works when git push to github.com is unstable).
Usage: py -3.10 work/api_push.py <local-commit>
The API normalizes committer dates to UTC, so the remote commit hash differs from the local one;
trees are identical. Run `git fetch origin main && git reset --hard origin/main` later to align.
"""
import base64, hashlib, json, subprocess, sys, urllib.request, datetime
from pathlib import Path

import os as _os
def _load_token():
    fp = Path(__file__).resolve().parent / ".gh_token"
    if fp.exists():
        return fp.read_text(encoding="utf-8").strip()
    tok = _os.environ.get("GH_TOKEN", "")
    if not tok:
        print("set GH_TOKEN env or create work/.gh_token")
        sys.exit(2)
    return tok
TOKEN = _load_token()
REPO = "ZhongdaHospital-SEU/Path-AGNN-Cox"
API = "https://api.github.com/repos/%s" % REPO
HEADERS = {"Authorization": "token " + TOKEN, "Accept": "application/vnd.github+json",
           "User-Agent": "codex-push", "X-GitHub-Api-Version": "2022-11-28"}
COMMIT = sys.argv[1]

def call(method, url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=HEADERS, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print("HTTPError", e.code, url, body[:800])
        raise

def git_blob(sha):
    out = subprocess.run(["git", "cat-file", "blob", sha], capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode(errors="replace"))
    return out.stdout

def git_sha(content):
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()

def git(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()

meta = git(["git", "cat-file", "-p", COMMIT])
lines = meta.splitlines()
tree = next(l.split()[1] for l in lines if l.startswith("tree "))
msg = meta[meta.find("\n\n")+2:].rstrip("\n")
def parse(l, tag):
    rest = l[len(tag):]
    nm, em, ts, tz = rest.rsplit(" ", 3)
    return nm.replace(" <", ""), em.strip("<>"), int(ts), tz
an, ae, ats, atz = parse(next(l for l in lines if l.startswith("author ")), "author ")
cn, ce, cts, ctz = parse(next(l for l in lines if l.startswith("committer ")), "committer ")
def iso(ts, tz):
    sign = 1 if tz[0] == "+" else -1
    dt = datetime.datetime.fromtimestamp(ts, datetime.timezone(datetime.timedelta(hours=sign*int(tz[1:3]), minutes=sign*int(tz[3:5]))))
    return dt.isoformat(timespec="seconds")
author = {"name": an, "email": ae, "date": iso(ats, atz)}
committer = {"name": cn, "email": ce, "date": iso(cts, ctz)}

# current remote head must be the parent (or its remote-equivalent)
head = call("GET", API + "/git/refs/heads/main")["object"]["sha"]
local_parent = next(l.split()[1] for l in lines if l.startswith("parent "))
print("remote head:", head)
print("local parent:", local_parent)

diff = git(["git", "diff-tree", "-r", "--name-status", local_parent, COMMIT])
entries = []
for ln in diff.splitlines():
    if not ln.strip():
        continue
    status, path = ln.split("\t")
    blob = git(["git", "rev-parse", COMMIT + ":" + path])
    content = git_blob(blob)
    assert git_sha(content) == blob
    r = call("POST", API + "/git/blobs", {"content": base64.b64encode(content).decode(), "encoding": "base64"})
    assert r["sha"] == blob, (path, r["sha"])
    entries.append({"path": path, "mode": "100644", "type": "blob", "sha": blob})
    print("blob ok:", path)

parent_tree = git(["git", "rev-parse", local_parent + "^{tree}"])
tr = call("POST", API + "/git/trees", {"base_tree": parent_tree, "tree": entries})
assert tr["sha"] == tree, ("tree mismatch", tr["sha"], tree)
print("tree ok:", tr["sha"])

cm = call("POST", API + "/git/commits", {
    "message": msg, "tree": tr["sha"], "parents": [head],
    "author": author, "committer": committer})
ref = call("PATCH", API + "/git/refs/heads/main", {"sha": cm["sha"], "force": False})
print("pushed:", ref["object"]["sha"])
