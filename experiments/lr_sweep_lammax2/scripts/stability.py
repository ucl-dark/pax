import sys, glob, os, numpy as np
sys.path.insert(0, '/tmp/claude-0/runs')
from summarize import parse
def coop(r): return r["s20"] >= -14 and r["o20"] >= -14
def sucker(r): return r["s20"] <= -22 and r["o20"] >= -8
print(f"{'variant':16} {'seed':4} {'coop%last2k':>11} {'sucker%last2k':>13} {'best500 (S,O)':>18} {'mean last1k (S,O)':>19} {'first coop it':>13}")
for tag in sorted(glob.glob("/tmp/claude-0/runs/[CUV]*")):
    for sd in sorted(glob.glob(tag + "/s*")):
        if not os.path.exists(sd + "/DONE"): continue
        rows = parse(sd + "/log.txt")
        if len(rows) < 100: continue
        last2k = [r for r in rows if r["it"] >= rows[-1]["it"] - 2000]
        last1k = [r for r in rows if r["it"] >= rows[-1]["it"] - 1000]
        c = np.mean([coop(r) for r in last2k]); s = np.mean([sucker(r) for r in last2k])
        # best 500-iteration window by welfare of final episode
        w = 100  # 100 logs = 500 iters
        best = max(range(len(rows) - w), key=lambda i: np.mean([rows[j]["s20"] + rows[j]["o20"] for j in range(i, i + w)]))
        bS = np.mean([rows[j]["s20"] for j in range(best, best + w)]); bO = np.mean([rows[j]["o20"] for j in range(best, best + w)])
        mS = np.mean([r["s20"] for r in last1k]); mO = np.mean([r["o20"] for r in last1k])
        first = next((r["it"] for r in rows if coop(r) and all(coop(x) for x in rows[rows.index(r):rows.index(r)+20])), None)
        print(f"{os.path.basename(tag):16} {os.path.basename(sd):4} {100*c:10.0f}% {100*s:12.0f}% {bS:8.1f},{bO:6.1f}    {mS:8.1f},{mO:6.1f}    {str(first):>13}")
