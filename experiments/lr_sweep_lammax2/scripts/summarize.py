import re, glob, sys, os
import numpy as np
pat_it = re.compile(r"^Iteration (\d+)")
pat_mm = re.compile(r"meta-mean\s+: shaper (\S+) \| co-player (\S+) \| welfare (\S+)")
pat_last = re.compile(r"episode 20\s+: shaper (\S+) \| co-player (\S+)")
pat_lam = re.compile(r"\[(shaper|co-player)\s*\] R .*lam (\S+) ->")
pat_cp = re.compile(r"cooperation_probability/(CC|CD|DC|DD|START): (\S+)")
pat_sp = re.compile(r"state_probability/(CC|CD|DC|DD): (\S+)")
def parse(path):
    rows=[]; cur=None
    for line in open(path):
        line=line.strip()
        m=pat_it.match(line)
        if m:
            if cur: rows.append(cur)
            cur={"it":int(m.group(1))}; continue
        if cur is None: continue
        if (m:=pat_mm.search(line)): cur.update(s=float(m[1]),o=float(m[2]),w=float(m[3]))
        elif (m:=pat_last.search(line)): cur.update(s20=float(m[1]),o20=float(m[2]))
        elif (m:=pat_lam.search(line)): cur["lam_"+m[1][0]]=float(m[2])
        elif (m:=pat_cp.search(line)): cur["p"+m[1]]=float(m[2])
        elif (m:=pat_sp.search(line)): cur["v"+m[1]]=float(m[2])
    if cur and "pDD" in cur: rows.append(cur)
    return rows
def classify(r):
    s,o=r["s20"],r["o20"]
    if s>=-14 and o>=-14: return "COOP(~-12,-12)"
    if s<=-22 and o>=-8: return "SUCKER(~-27,-2)"
    if s>=-8 and o<=-22: return "EXPLOIT"
    if s<=-17 and o<=-17: return "mutual-D"
    return "mixed"
def classify_old(r):
    # "good": both IR over meta-episode, and co-player ends cooperating mutually
    ok_ir = r["s"]>=-20 and r["o"]>=-20
    if ok_ir and r["vCC"]>0.5: return "GOOD(coop)"
    if r["vCD"]>0.4: return "sucker"
    if r["vDC"]>0.3: return "exploit"
    if r["vDD"]>0.5: return "mutual-D"
    return "mixed"
if __name__ == "__main__":
    tail = int(sys.argv[2]) if len(sys.argv)>2 else 50
    for tag in sorted(glob.glob(sys.argv[1] if len(sys.argv)>1 else "/tmp/claude-0/runs/[CUV]_*")):
        print("==",os.path.basename(tag))
        for sd in sorted(glob.glob(tag+"/s*")):
            rows=parse(sd+"/log.txt")
            if len(rows)<2: print("  ",os.path.basename(sd),"no data"); continue
            t=rows[-tail:]
            avg={k:np.mean([r[k] for r in t if k in r]) for k in t[-1]}
            std_w=np.std([r["w"] for r in t])
            print(f"  {os.path.basename(sd):3} it={rows[-1]['it']:5d} final-ep=({avg['s20']:6.2f},{avg['o20']:6.2f}) meta S={avg['s']:6.2f} O={avg['o']:6.2f} W={avg['w']:6.2f}(sd {std_w:4.2f}) "
                  f"lam_s={avg['lam_s']:.2f} lam_o={avg['lam_c']:.2f} pC[CC,CD,DC,DD]=[{avg['pCC']:.2f},{avg['pCD']:.2f},{avg['pDC']:.2f},{avg['pDD']:.2f}] "
                  f"vis[CC,CD,DC,DD]=[{avg['vCC']:.2f},{avg['vCD']:.2f},{avg['vDC']:.2f},{avg['vDD']:.2f}] {classify(avg)}"
                  + ("" if os.path.exists(sd+"/DONE") else " (running)"))
