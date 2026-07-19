import csv, math, glob, os, sys

def load_truth(path):
    T={}
    for r in csv.DictReader(open(path)):
        tow=round(float(r['GPS TOW']),0)
        T[tow]=(float(r['ECEF X (m)']),float(r['ECEF Y (m)']),float(r['ECEF Z (m)']))
    return T

def load_sat(path):
    S={}
    for r in csv.DictReader(open(path)):
        tow=round(float(r['GPS TOW']),0)
        S[tow]=(float(r['ECEF X (m)']),float(r['ECEF Y (m)']),float(r['ECEF Z (m)']),
                float(r['Body Elevation (rad)']))
    return S

def dop_at(rx, sats):
    # sats: list of (sx,sy,sz). Build geometry matrix H (n x 4), rows [-ex,-ey,-ez,1]
    import itertools
    H=[]
    for (sx,sy,sz) in sats:
        dx,dy,dz=sx-rx[0],sy-rx[1],sz-rx[2]
        r=math.sqrt(dx*dx+dy*dy+dz*dz)
        H.append([dx/r,dy/r,dz/r,1.0])
    n=len(H)
    if n<4: return None
    # Q = (H^T H)^-1
    HtH=[[sum(H[k][i]*H[k][j] for k in range(n)) for j in range(4)] for i in range(4)]
    # invert 4x4 (Gauss-Jordan)
    import copy
    A=[row[:]+[1.0 if i==j else 0.0 for j in range(4)] for i,row in enumerate(HtH)]
    for col in range(4):
        piv=max(range(col,4),key=lambda r:abs(A[r][col]))
        if abs(A[piv][col])<1e-12: return None
        A[col],A[piv]=A[piv],A[col]
        d=A[col][col]
        A[col]=[x/d for x in A[col]]
        for r in range(4):
            if r!=col:
                f=A[r][col]
                A[r]=[A[r][k]-f*A[col][k] for k in range(8)]
    Q=[A[i][4:] for i in range(4)]
    gdop=math.sqrt(Q[0][0]+Q[1][1]+Q[2][2]+Q[3][3])
    pdop=math.sqrt(Q[0][0]+Q[1][1]+Q[2][2])
    return gdop,pdop

def run(scendir, label):
    truth=load_truth(os.path.join(scendir,'receiver_antenna.csv'))
    satfiles=[f for f in glob.glob(os.path.join(scendir,'*.csv')) if 'receiver_antenna' not in f]
    sats={}
    for f in satfiles:
        sats[os.path.basename(f)]=load_sat(f)
    rows=[]
    for tow in sorted(truth):
        rx=truth[tow]
        vis=[]
        for name,S in sats.items():
            if tow in S:
                sx,sy,sz,el=S[tow]
                if el>math.radians(10):  # elevation mask 10 deg (matches -m 10)
                    vis.append((sx,sy,sz))
        d=dop_at(rx,vis)
        if d:
            rows.append((tow-min(truth), len(vis), d[0], d[1]))
    return rows

for scendir,label in [('Skydel_logs/Orbit_s6a','GPS_GAL'),('Skydel_logs/Orbit_s6a_withBDS','GPS_GAL_BDS')]:
    rows=run(scendir,label)
    valid=[r for r in rows if r[0]<=553]
    gd=[r[2] for r in valid]; pd=[r[3] for r in valid]; ns=[r[1] for r in valid]
    print(f"=== {label} (Skydel truth, {len(valid)} epochs) ===")
    print(f"  nsat: mean {sum(ns)/len(ns):.1f} (min {min(ns)}, max {max(ns)})")
    print(f"  GDOP: mean {sum(gd)/len(gd):.2f} (min {min(gd):.2f}, max {max(gd):.2f})")
    print(f"  PDOP: mean {sum(pd)/len(pd):.2f} (min {min(pd):.2f}, max {max(pd):.2f})")
    # save csv
    out=f'/tmp/dop_{label}.csv'
    with open(out,'w') as f:
        f.write('elapsed_s,nsat,gdop,pdop\n')
        for r in valid: f.write(f'{r[0]:.0f},{r[1]},{r[2]:.4f},{r[3]:.4f}\n')
    print(f"  -> {out}")