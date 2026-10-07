import sys, glob, os, csv
sys.path.insert(0, os.path.dirname(__file__))
from summarize import parse
out = sys.argv[1]
for sd in sorted(glob.glob("/tmp/claude-0/runs/[CUV]_*/s*")):
    rows = parse(sd + "/log.txt")
    if not rows: continue
    tag, seed = sd.split("/")[-2:]
    os.makedirs(f"{out}/{tag}", exist_ok=True)
    keys = ["it","s","o","w","s20","o20","lam_s","lam_c","pCC","pCD","pDC","pDD","pSTART","vCC","vCD","vDC","vDD"]
    with open(f"{out}/{tag}/{seed}.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(keys)
        for r in rows: w.writerow([r.get(k,"") for k in keys])
