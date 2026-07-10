#!/usr/bin/env python3
"""
Forensic pseudorange residual analysis for the u-blox outliers.

For each epoch in a RINEX obs excerpt:
  residual(sat) = P_obs - |r_sat(t_tx) - ARP| + c*dt_sat_clock
Then subtract the per-epoch, per-constellation median (receiver clock +
inter-system bias). A healthy satellite stays within tens of metres
(iono/tropo/eph errors); a millisecond-level fault shows up as ~300 km.
"""
import math, sys
from datetime import datetime, timedelta

C = 299792458.0
MU_GPS = 3.986005e14
MU_GAL = 3.986004418e14
MU_BDS = 3.986004418e14
OMEGA_E = 7.2921151467e-5
F_REL = -4.442807633e-10

# ---------- ARP (ECEF) ----------
lat = math.radians(40 + 35/60 + 28.86718/3600)
lon = math.radians(-(3 + 42/60 + 24.03787/3600))
h = 805.246
a_e = 6378137.0; f_e = 1/298.257223563; e2 = f_e*(2-f_e)
Nrad = a_e/math.sqrt(1-e2*math.sin(lat)**2)
ARP = ((Nrad+h)*math.cos(lat)*math.cos(lon), (Nrad+h)*math.cos(lat)*math.sin(lon), (Nrad*(1-e2)+h)*math.sin(lat))

# ---------- parse mixed nav (GPS/GAL/BDS Keplerian) ----------
def fnum(s):
    s = s.replace('D','E').replace('d','e').strip()
    return float(s) if s else 0.0

def parse_nav(path):
    eph = {}
    with open(path) as f:
        for line in f:
            if "END OF HEADER" in line:
                break
        lines = f.readlines()
    i = 0
    while i < len(lines):
        l = lines[i]
        if len(l) > 3 and l[0] in 'GEC' and l[1:3].strip().isdigit():
            sat = l[:3].replace(' ','0')
            try:
                toc = datetime(int(l[4:8]), int(l[9:11]), int(l[12:14]), int(l[15:17]), int(l[18:20]), int(l[21:23]))
                af0, af1, af2 = fnum(l[23:42]), fnum(l[42:61]), fnum(l[61:80])
                b = [ [fnum(lines[i+k][4+19*j:4+19*(j+1)]) for j in range(4)] for k in range(1,8) ]
                rec = dict(toc=toc, af0=af0, af1=af1, af2=af2, crs=b[0][1], dn=b[0][2], m0=b[0][3], cuc=b[1][0], ecc=b[1][1], cus=b[1][2], sqrta=b[1][3], toe=b[2][0], cic=b[2][1], omega0=b[2][2], cis=b[2][3], i0=b[3][0], crc=b[3][1], w=b[3][2], omegadot=b[3][3], idot=b[4][0])
                eph.setdefault(sat, []).append(rec)
            except (ValueError, IndexError):
                pass
            i += 8
        else:
            i += 1
    return eph

def sat_pos_clock(sat, ephlist, t_gpst, pr):
    """Satellite ECEF position & clock at transmission time (broadcast)."""
    sys_c = sat[0]
    t_sys = t_gpst - timedelta(seconds=14) if sys_c == 'C' else t_gpst
    t_tx = t_sys - timedelta(seconds=pr/C)
    # pick ephemeris with closest toc
    rec = min(ephlist, key=lambda r: abs((t_tx - r['toc']).total_seconds()))
    dt_toc = (t_tx - rec['toc']).total_seconds()
    if abs(dt_toc) > 7200:
        return None
    mu = {'G': MU_GPS, 'E': MU_GAL, 'C': MU_BDS}[sys_c]
    A = rec['sqrta']**2
    n = math.sqrt(mu/A**3) + rec['dn']
    tk = dt_toc  # toc ~ toe for these products; km-level accuracy suffices
    M = rec['m0'] + n*tk
    E = M
    for _ in range(15):
        E = M + rec['ecc']*math.sin(E)
    nu = math.atan2(math.sqrt(1-rec['ecc']**2)*math.sin(E), math.cos(E)-rec['ecc'])
    phi = nu + rec['w']
    du = rec['cus']*math.sin(2*phi) + rec['cuc']*math.cos(2*phi)
    dr = rec['crs']*math.sin(2*phi) + rec['crc']*math.cos(2*phi)
    di = rec['cis']*math.sin(2*phi) + rec['cic']*math.cos(2*phi)
    u = phi + du
    r = A*(1-rec['ecc']*math.cos(E)) + dr
    inc = rec['i0'] + di + rec['idot']*tk
    x, y = r*math.cos(u), r*math.sin(u)
    # BDS GEO would need special handling; skip (not visible from Madrid)
    Om = rec['omega0'] + (rec['omegadot']-OMEGA_E)*tk - OMEGA_E*rec['toe']
    X = x*math.cos(Om) - y*math.cos(inc)*math.sin(Om)
    Y = x*math.sin(Om) + y*math.cos(inc)*math.cos(Om)
    Z = y*math.sin(inc)
    # Earth rotation during signal flight
    w_tau = OMEGA_E * pr/C
    Xr = X*math.cos(w_tau) + Y*math.sin(w_tau)
    Yr = -X*math.sin(w_tau) + Y*math.cos(w_tau)
    dts = rec['af0'] + rec['af1']*dt_toc + rec['af2']*dt_toc**2 + F_REL*rec['ecc']*rec['sqrta']*math.sin(E)
    return (Xr, Yr, Z), dts

# ---------- parse obs excerpt ----------
def parse_obs(path):
    types = {}
    with open(path) as f:
        for line in f:
            if "SYS / # / OBS TYPES" in line:
                sysc = line[0]
                tps = line[7:60].split()
                types[sysc] = tps
            if "END OF HEADER" in line:
                break
        epochs = []
        cur = None
        for line in f:
            if line.startswith('>'):
                p = line.split()
                t = datetime(int(p[1]),int(p[2]),int(p[3]),int(p[4]),int(p[5])) + timedelta(seconds=float(p[6]))
                cur = {'t': t, 'obs': {}}
                epochs.append(cur)
            elif cur is not None and len(line) > 3 and line[0] in 'GEC':
                sat = line[:3].replace(' ','0')
                tps = types[line[0]]
                vals = {}
                for j, tp in enumerate(tps):
                    fld = line[3+16*j:3+16*j+14]
                    try:
                        vals[tp] = float(fld)
                    except ValueError:
                        vals[tp] = None
                cur['obs'][sat] = vals
    return epochs

CODE = {'G':'C1C', 'E':'C1X', 'C':'C2I'}
SNR  = {'G':'S1C', 'E':'S1X', 'C':'S2I'}

def residuals(epochs, eph):
    out = []
    for ep in epochs:
        res = {}
        for sat, vals in ep['obs'].items():
            pr = vals.get(CODE[sat[0]])
            if pr is None or sat not in eph:
                continue
            r = sat_pos_clock(sat, eph[sat], ep['t'], pr)
            if r is None: continue
            (X,Y,Z), dts = r
            rho = math.dist((X,Y,Z), ARP)
            res[sat] = pr - rho + C*dts
        # per-constellation median = receiver clock + ISB
        clk = {}
        for s in 'GEC':
            v = sorted(x for k, x in res.items() if k[0] == s)
            if v: clk[s] = v[len(v)//2]
        out.append({'t': ep['t'], 'res': {k: v - clk[k[0]] for k, v in res.items() if k[0] in clk}, 'snr': {k: ep['obs'][k].get(SNR[k[0]]) for k in res}})
    return out

if __name__ == "__main__":
    eph = parse_nav('brdc175.rnx')
    print(f"Efemérides cargadas: {sum(len(v) for v in eph.values())} registros, {len(eph)} satélites")
    for name in sys.argv[1:]:
        epochs = parse_obs(name)
        print(f"\n{'='*70}\n{name}: {len(epochs)} épocas, {epochs[0]['t']} -> {epochs[-1]['t']}")
        rr = residuals(epochs, eph)
        # find anomalous residuals
        events = []
        for e in rr:
            for sat, v in e['res'].items():
                if abs(v) > 10000:
                    events.append((e['t'], sat, v, e['snr'].get(sat)))
        if not events:
            print("  Sin residuos anómalos > 10 km")
            continue
        print(f"  Residuos anómalos (>10 km): {len(events)}")
        # group per satellite
        sats = sorted(set(x[1] for x in events))
        for s in sats:
            ev = [x for x in events if x[1] == s]
            ms_vals = sorted(set(round(x[2]/C*1000, 2) for x in ev))
            print(f"\n  >>> {s}: {len(ev)} épocas anómalas, residuo en ms: {ms_vals}")
            # tracking arc: first epoch this sat appears in excerpt at all
            first = next(e['t'] for e in rr if s in e['res'])
            print(f"      primera época del arco en el extracto: {first}")
            for t, _, v, snr in ev[:10]:
                print(f"      {t}  residuo {v/1000:12.1f} km ({v/C*1000:+.3f} ms)  S1 {snr}")
    import pickle
    # keep last file's rr for plotting later
    with open('last_rr.pkl','wb') as f:
        pickle.dump(rr, f)