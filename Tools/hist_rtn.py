import csv, matplotlib, math
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statistics import mean, pstdev

def load(p):
    R=[];T=[];N=[]
    for r in csv.DictReader(open(p)):
        R.append(float(r['dR_m'])); T.append(float(r['dT_m'])); N.append(float(r['dN_m']))
    return R,T,N

for label,title in [('gps_gal','GPS+GAL'),('gps_gal_bds','GPS+GAL+BDS')]:
    R,T,N=load(f'RTN/{label}/comparisonA_residuals.csv')
    fig,axs=plt.subplots(1,3,figsize=(12,3.6))
    for ax,(data,name,color) in zip(axs,[(R,'Radial (R)','tab:blue'),(T,'Along-track (T)','tab:orange'),(N,'Cross-track (N)','tab:green')]):
        m,s=mean(data),pstdev(data)
        ax.hist(data,bins=40,density=True,color=color,alpha=0.7,edgecolor='white',linewidth=0.3)
        # gaussian overlay
        xs=[m-4*s+i*(8*s/100) for i in range(101)]
        ys=[1/(s*math.sqrt(2*math.pi))*math.exp(-0.5*((x-m)/s)**2) for x in xs]
        ax.plot(xs,ys,'k--',lw=1,label='Gaussian fit')
        ax.axvline(m,color='red',lw=1,ls=':')
        ax.set_title(f'{name}\n$\\mu$={m:+.2f}, $\\sigma$={s:.2f} m',fontsize=10)
        ax.set_xlabel('Residual [m]'); ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axs[0].set_ylabel('Density')
    fig.suptitle(f'Comparison A residual distribution — {title}',fontsize=11)
    fig.tight_layout()
    fig.savefig(f'/mnt/user-data/outputs/rtn_hist_{label}.png',dpi=130)
    print(f'saved rtn_hist_{label}.png : R mu={mean(R):+.2f} T mu={mean(T):+.2f} N mu={mean(N):+.2f}')