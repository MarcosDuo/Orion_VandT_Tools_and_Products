#!/usr/bin/env python3
"""Cross-receiver C/N0 and tracking analysis for the zero-baseline test.

Both receivers share the same antenna (splitter), so per-satellite C/N0
differences between them reflect receiver front-end/reporting behaviour,
not antenna or environment. Uses header-driven RINEX parsing (works with
gfzrnx-reordered observable lists) and broadcast ephemerides for
elevation computation.
"""
import math, bisect, pickle
from datetime import datetime, timedelta
from collections import defaultdict
from outlier_forensic import parse_nav, sat_pos_clock, ARP

# first-frequency and second-frequency SNR codes per system, per receiver
SNR1 = {'orion': {'G':'S1C','E':'S1C','C':'S2I'},
        'ublox': {'G':'S1C','E':'S1X','C':'S2I'}}
SNR2 = {'ublox': {'G':'S2X','E':'S7X','C':'S7I'}}

GEO_BDS = {f'C{n:02d}' for n in list(range(1,6))+list(range(59,64))}

def parse_obs_snr(path, rx):
    """epoch(rounded s since first) -> sat -> (snr1, snr2)"""
    types = {}
    data = {}
    with open(path, encoding='latin1') as f:
        for line in f:
            if "SYS / # / OBS TYPES" in line and line[0] in 'GEC':
                types[line[0]] = line[7:60].split()
            if "END OF HEADER" in line:
                break
        cur = None
        for line in f:
            if line.startswith('>'):
                p = line.split()
                t = datetime(int(p[1]),int(p[2]),int(p[3]),int(p[4]),int(p[5])) + timedelta(seconds=float(p[6]))
                # round to 30 s grid
                sec = round((t - datetime(2026,6,24)).total_seconds() / 30) * 30
                cur = {}
                data[sec] = (t, cur)
            elif cur is not None and line[0] in types:
                sysc = line[0]
                sat = line[:3].replace(' ','0')
                tps = types[sysc]
                def get(code):
                    if code not in tps: return None
                    j = tps.index(code)
                    fld = line[3+16*j:3+16*j+14].strip()
                    try: return float(fld)
                    except ValueError: return None
                s1 = get(SNR1[rx][sysc])
                s2 = get(SNR2.get(rx, {}).get(sysc)) if rx in SNR2 else None
                if s1 is not None or s2 is not None:
                    cur[sat] = (s1, s2)
    return data

def build_eph_index(paths):
    eph = {}
    for p in paths:
        for sat, recs in parse_nav(p).items():
            eph.setdefault(sat, []).extend(recs)
    idx = {}
    for sat, recs in eph.items():
        recs.sort(key=lambda r: r['toc'])
        idx[sat] = ([r['toc'] for r in recs], recs)
    return idx

latr = math.radians(40 + 35/60 + 28.86718/3600)
lonr = math.radians(-(3 + 42/60 + 24.03787/3600))

def elevation(sat, t, eph_idx):
    if sat in GEO_BDS or sat not in eph_idx:
        return None
    tocs, recs = eph_idx[sat]
    t_sys = t - timedelta(seconds=14) if sat[0]=='C' else t
    i = bisect.bisect_left(tocs, t_sys)
    cands = [recs[j] for j in (i-1, i) if 0 <= j < len(recs)]
    if not cands: return None
    rec = min(cands, key=lambda r: abs((t_sys - r['toc']).total_seconds()))
    r = sat_pos_clock(sat, [rec], t, 21e6)
    if r is None: return None
    X = r[0]
    dx = [X[i]-ARP[i] for i in range(3)]
    e = -math.sin(lonr)*dx[0] + math.cos(lonr)*dx[1]
    n = -math.sin(latr)*math.cos(lonr)*dx[0] - math.sin(latr)*math.sin(lonr)*dx[1] + math.cos(latr)*dx[2]
    u =  math.cos(latr)*math.cos(lonr)*dx[0] + math.cos(latr)*math.sin(lonr)*dx[1] + math.sin(latr)*dx[2]
    return math.degrees(math.asin(u/math.sqrt(e*e+n*n+u*u)))

if __name__ == "__main__":
    print("Parsing obs...")
    orion = parse_obs_snr('/mnt/user-data/uploads/orion_s30.obs', 'orion')
    ublox = parse_obs_snr('/mnt/user-data/uploads/ublox_s30.obs', 'ublox')
    print(f"  orion: {len(orion)} épocas | ublox: {len(ublox)} épocas")
    common = sorted(set(orion) & set(ublox))
    print(f"  épocas comunes (grid 30 s): {len(common)}")
    print("Loading ephemerides...")
    eph_idx = build_eph_index(['brdc175.rnx','brdc176.rnx','brdc177.rnx'])
    with open('cross_data.pkl','wb') as f:
        pickle.dump((orion, ublox, common), f)
    print("OK — data cached")