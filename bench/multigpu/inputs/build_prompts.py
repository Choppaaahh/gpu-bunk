#!/usr/bin/env python3
"""Frozen rule (prereg section 4, ITEM12): source databricks/databricks-dolly-15k @ bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a,
file databricks-dolly-15k.jsonl. text = instruction, plus "\\n\\n" + context when the
context is non-empty. Eligible: 400 <= len(text) <= 4000 characters (>= 64 DeepSeek-V2 tokens with margin). Take the eligible
rows in file order per category; round-robin over the categories in sorted order until N = 72 prompts.
Usage: build_prompts.py <dolly jsonl> <out jsonl>      build_prompts.py --self-test
Output contract: exit 0 + 'WROTE n=72 sha256=<hex>' (out overwritten); exit 1 if fewer than 72 are eligible; 2 = usage."""
import hashlib
import json
import sys

N, LO, HI = 72, 400, 4000


def text_of(r):
    c = (r.get("context") or "").strip()
    return r["instruction"].strip() + ("\n\n" + c if c else "")


def select(rows, n=N):
    per = {}
    for i, r in enumerate(rows):
        t = text_of(r)
        if LO <= len(t) <= HI:
            per.setdefault(r["category"], []).append({"text": t, "category": r["category"], "src_line": i})
    out, cats, k = [], sorted(per), 0
    while len(out) < n and any(k < len(per[c]) for c in cats):
        for c in cats:
            if k < len(per[c]) and len(out) < n:
                out.append(per[c][k])
        k += 1
    return out


def self_test():
    rows = [{"instruction": "a" * 450, "context": "", "category": "x"},
            {"instruction": "b" * 10, "context": "c" * 500, "category": "y"},
            {"instruction": "short", "context": "", "category": "x"},
            {"instruction": "d" * 5000, "context": "", "category": "y"},
            {"instruction": "e" * 420, "context": "", "category": "x"}]
    got = select(rows, 3)
    want = [0, 1, 4]            # hand: x[0], y[1] (instruction + \n\n + context = 512 chars), then x's second eligible row
    ok = [g["src_line"] for g in got] == want and len(got[1]["text"]) == 10 + 2 + 500
    print("SELF-TEST " + ("PASS" if ok else f"FAIL {[g['src_line'] for g in got]}"))
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    if len(sys.argv) != 3:
        print(__doc__.split("Usage: ")[1].split("\n")[0], file=sys.stderr)
        sys.exit(2)
    rows = [json.loads(ln) for ln in open(sys.argv[1])]
    sel = select(rows)
    if len(sel) < N:
        print(f"ONLY {len(sel)} eligible")
        sys.exit(1)
    blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sel)
    open(sys.argv[2], "w").write(blob)
    print(f"WROTE n={len(sel)} sha256={hashlib.sha256(blob.encode()).hexdigest()}")
