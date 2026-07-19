import csv, math, glob, os

def load_truth(path):
    T={}
    for r in csv.DictReader(open(path)):
        T[round(float(r['GPS TOW']),0)]=(float(r['ECEF X (m)']),float(r['ECEF Y (m)']),float(r['ECEF Z (m)']))
    return T

def load_sat_pos(scendir):
    # map (prn -> {tow: (x,y,z,el)}); prn from filename like 'L1CA 02.csv'->G02, 'E1 12'->E12, 'B1I 11'->C11
    sats={}
    for f in glob.glob(os.path.join(scendir,'*.csv')):
        b=os.path.basename(f)
        if 'receiver_antenna' in b: continue
        sig,num=b.replace('.csv','').split()
        sysc={'L1CA':'G','E1':'E','B1I':'C'}.get(sig)
        if not sysc: continue
        prn=f"{sysc}{int(num):02d}"
        d={}
        for r in csv.DictReader(open(f)):
            d[round(float(r['GPS TOW']),0)]=(float(r['ECEF X (m)']),float(r['ECEF Y (m)']),float(r['ECEF Z (m)']),float(r['Body Elevation (rad)']))
        sats[prn]=d
    return sats

def rx_tracked(obspath):
    # returns {tow_sod: set(prn)} ; tow via sod; approximate tow from receiver_antenna mapping externally
    ep={}
    with open(obspath) as fh:
        started=False; cur=None
        for line in fh:
            if not started:
                if 'END OF HEADER' in line: started=True
                continue
            if line.startswith('>'):
                p=line.split()
                sod=int(p[4])*3600+int(p[5])*60+int(float(p[6]))
                cur=sod; ep[cur]=set()
            elif line[:1] in 'GEC':
                ep[cur].add(line[:3])
    return ep

def dop(rx,sats_xyz):
    H=[]
    for (sx,sy,sz) in sats_xyz:
        dx,dy,dz=sx-rx[0],sy-rx[1],sz-rx[2]; r=math.sqrt(dx*dx+dy*dy+dz*dz)
        H.append([dx/r,dy/r,dz/r,1.0])
    n=len(H)
    if n<4: return None
    HtH=[[sum(H[k][i]*H[k][j] for k in range(n)) for j in range(4)] for i in range(4)]
    A=[row[:]+[1.0 if i==j else 0.0 for j in range(4)] for i,row in enumerate(HtH)]
    for col in range(4):
        piv=max(range(col,4),key=lambda r:abs(A[r][col]))
        if abs(A[piv][col])<1e-12: return None
        A[col],A[piv]=A[piv],A[col]; d=A[col][col]; A[col]=[x/d for x in A[col]]
        for r in range(4):
            if r!=col:
                ff=A[r][col]; A[r]=[A[r][k]-ff*A[col][k] for k in range(8)]
    Q=[A[i][4:] for i in range(4)]
    return math.sqrt(Q[0][0]+Q[1][1]+Q[2][2]+Q[3][3]), math.sqrt(Q[0][0]+Q[1][1]+Q[2][2])

for scendir,obs,label in [('Skydel_logs/Orbit_s6a','rnx_out/GPS_GAL/orion_gps_gal.obs','GPS_GAL'),
                          ('Skydel_logs/Orbit_s6a_withBDS','rnx_out/GPS_GAL_BDS/orion_gps_gal_bds.obs','GPS_GAL_BDS')]:
    truth=load_truth(scendir+'/receiver_antenna.csv')
    sats=load_sat_pos(scendir)
    tracked=rx_tracked(obs)
    tow0=min(truth); sod0=min(tracked)
    rows=[]
    for sod in sorted(tracked):
        tow=tow0+(sod-sod0)
        if tow not in truth: continue
        if (tow-tow0)>553: continue
        rx=truth[tow]
        xyz=[]
        for prn in tracked[sod]:
            if prn in sats and tow in sats[prn]:
                sx,sy,sz,el=sats[prn][tow]; xyz.append((sx,sy,sz))
        d=dop(rx,xyz)
        if d: rows.append((tow-tow0,len(xyz),d[0],d[1]))
    gd=[r[2] for r in rows]; pd=[r[3] for r in rows]
    print(f"=== {label} (Orion Rx-tracked, {len(rows)} ep) ===  GDOP mean {sum(gd)/len(gd):.2f}  PDOP mean {sum(pd)/len(pd):.2f}")
    with open(f'/tmp/dop_rx_{label}.csv','w') as f:
        f.write('elapsed_s,nsat,gdop,pdop\n')
        for r in rows: f.write(f'{r[0]:.0f},{r[1]},{r[2]:.4f},{r[3]:.4f}\n')