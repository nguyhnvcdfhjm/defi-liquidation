"""Tab 2: quét hệ số nhân λ của cú sốc đa tài sản, 30 seed (1-30). Chạy: python run_tab2_multiseed.py [file_ra.csv]"""
import sys, os, math, statistics as st, csv
from scenarios import SCENARIOS, run_scenario, amplification
seeds=list(range(1,31)); lams=[0.7,0.85,0.9,0.95,1.0,1.25]
def ci(v):
    n=len(v); m=st.mean(v); return m, (1.96*st.stdev(v)/math.sqrt(n) if n>1 else 0)
rows=[]
for lam in lams:
    sh={k:min(0.95,v*lam) for k,v in SCENARIOS["multi"].items()}
    liq=[];bad=[];trig=0;amp=[]
    for s in seeds:
        m,d=run_scenario(sh,15,seed=s)
        ld=sum(e["debt_covered"] for e in m.liquidation_events)
        liq.append(ld);bad.append(m.bad_debt);trig+= ld>0
        a=amplification(sh,15,seed=s); amp.append(a)
    fin=[a for a in amp if math.isfinite(a)]
    lm,lc=ci(liq);bm,bc=ci(bad)
    am,ac=ci(fin) if len(fin)>1 else (float('nan'),0)
    rows.append([lam,"/".join(f"{100*v:.1f}" for v in sh.values()),trig,len(seeds),round(lm),round(lc),round(bm),round(bc),round(am,2),round(ac,2),len(fin)])
    print(rows[-1],flush=True)
OUT = sys.argv[1] if len(sys.argv) > 1 else "tab2_multiseed.csv"
with open(OUT,"w",newline="",encoding="utf-8-sig") as f:
    w=csv.writer(f);w.writerow(["lambda","WETH/WBTC/LINK_%","seeds_triggered","n_seeds","liq_debt_mean","liq_debt_ci95","bad_debt_mean","bad_debt_ci95","amp_mean","amp_ci95","n_finite_amp"]);w.writerows(rows)
