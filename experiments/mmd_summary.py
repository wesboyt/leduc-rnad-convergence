"""Summarise reference jsonl curves: python experiments/mmd_summary.py "results/mmd/*.jsonl" """
import glob
import json
import os
import sys

for f in sorted(glob.glob(sys.argv[1])):
    rs = [json.loads(line) for line in open(f) if '"t"' in line]
    if not rs:
        print(os.path.basename(f), "(empty)")
        continue
    k = max(1, len(rs) // 8)
    pts = " ".join("%d:%.4f" % (r["t"], r["nashconv"]) for r in rs[::k])
    print("%-40s %s | last %.4f min %.4f" % (os.path.basename(f)[:40], pts, rs[-1]["nashconv"],
                                             min(r["nashconv"] for r in rs)))
