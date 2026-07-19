import csv, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load(p):
    t=[];g=[];pd=[]
    for r in csv.DictReader(open(p)):
        t.append(float(r['elapsed_s'])); g.append(float(r['gdop'])); pd.append(float(r['pdop']))
    return t,g,pd

for label,title in [('GPS_GAL','GPS+GAL'),('GPS_GAL_BDS','GPS+GAL+BDS')]:
    ts,gs,ps=load(f'/tmp/dop_{label}.csv')       # skydel truth
    tr,gr,pr=load(f'/tmp/dop_rx_{label}.csv')    # orion tracked
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(ts,ps,label='PDOP (Skydel, all visible SV)',lw=1.3,color='tab:blue')
    ax.plot(tr,pr,label='PDOP (Orion, tracked SV)',lw=1.1,color='tab:orange',alpha=0.9)
    ax.plot(ts,gs,label='GDOP (Skydel)',lw=1.0,color='tab:blue',ls='--',alpha=0.6)
    ax.plot(tr,gr,label='GDOP (Orion)',lw=1.0,color='tab:orange',ls='--',alpha=0.6)
    ax.set_xlabel('Elapsed time [s]'); ax.set_ylabel('DOP')
    ax.set_title(f'Geometric dilution of precision — {title}')
    ax.legend(fontsize=8,ncol=2); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(f'/mnt/user-data/outputs/dop_{label}.png',dpi=130)
    print(f'saved dop_{label}.png')