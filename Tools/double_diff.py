#!/usr/bin/env python3
"""
zb_double_diff.py v4 — Between-receiver single differences (zero-baseline).

Phase noise is characterised inside continuous arcs; phase discontinuities
(cycle slips, half-cycle slips, re-locks) split arcs at |dSD| > JUMP and are
LOGGED to a CSV report (satellite, epoch, jump size in metres and cycles)
rather than silently removed. Inter-receiver phase clock is estimated by
integrating the median epoch-to-epoch increment (ambiguity-free); code clock
uses the direct per-epoch median.
"""
import statistics as st
from collections import defaultdict
from datetime import datetime, timezone
from cycle_slips import parse_obs, LAMBDA

JUMP = 0.05         # m, phase arc-split threshold
REPORT = 'C:/GNSS/TESTS_3CANTOS/plots/phase_discontinuities.csv'
FIGURE = 'C:/GNSS/TESTS_3CANTOS/plots/single_diff_noise_3.png'

def key(t):
    return round(t.timestamp())

def load(path):
    d = parse_obs(path)
    out = {}
    for sat, rows in d.items():
        for t, L, D, P, lli in rows:
            out[(sat, key(t))] = (L*LAMBDA[sat[0]], P)
    return out

def moving_median_residual(series, win=61):
    vals = [v for _, v in series]
    n = len(vals); h = win//2
    return [vals[i] - st.median(vals[max(0,i-h):min(n,i+h+1)]) for i in range(n)]

def split_arcs(sat, ser, jump, events):
    """Split on data gaps or jumps > threshold; log jump events."""
    ser.sort()
    lam = LAMBDA[sat[0]]
    arcs, arc = [], [ser[0]]
    for a, b in zip(ser, ser[1:]):
        gap = b[0] - a[0] > 1
        jmp = abs(b[1] - a[1]) > jump
        if gap or jmp:
            arcs.append(arc); arc = []
            if jmp:
                events.append((b[0], sat, b[1]-a[1], (b[1]-a[1])/lam, b[0]-a[0]))
        arc.append(b)
    arcs.append(arc)
    return arcs

if __name__ == "__main__":
    o = load('C:/GNSS/TESTS_3CANTOS/ORION_LONG_20260624_065633.obs')
    u = load('C:/GNSS/TESTS_3CANTOS/UBLOX_LONG_20260624_065649.obs')
    common = sorted(set(o) & set(u))
    print(f"Common measurements (sat, epoch): {len(common)}")
    sdL = {k: u[k][0]-o[k][0] for k in common}
    sdP = {k: (u[k][1]-o[k][1]) if (u[k][1] and o[k][1]) else None for k in common}
    epochs = sorted({e for _, e in common})

    # phase clock: integrated median epoch-to-epoch increment
    by_sat = defaultdict(dict)
    for (sat, e) in common:
        by_sat[sat][e] = sdL[(sat, e)]
    rate = {}
    for e0, e1 in zip(epochs, epochs[1:]):
        if e1 - e0 != 1: continue
        inc = [s[e1]-s[e0] for s in by_sat.values() if e0 in s and e1 in s]
        if inc: rate[e1] = st.median(inc)
    clk = {epochs[0]: 0.0}
    for e0, e1 in zip(epochs, epochs[1:]):
        clk[e1] = clk[e0] + rate.get(e1, 0.0)

    # code clock: direct per-epoch median (no ambiguity in code)
    by_epP = defaultdict(list)
    for k, v in sdP.items():
        if v is not None: by_epP[k[1]].append(v)
    clkP = {e: st.median(v) for e, v in by_epP.items()}

    per_sat_L = defaultdict(list); per_sat_P = defaultdict(list)
    for (sat, e) in common:
        per_sat_L[sat].append((e, sdL[(sat,e)]-clk[e]))
        if sdP[(sat,e)] is not None:
            per_sat_P[sat].append((e, sdP[(sat,e)]-clkP[e]))

    noiseL = defaultdict(list); noiseP = defaultdict(list)
    isb = defaultdict(list); events = []
    arcs_by_sat = {}
    for sat, ser in per_sat_L.items():
        arcs = [a for a in split_arcs(sat, ser, JUMP, events) if len(a) >= 120]
        arcs_by_sat[sat] = arcs
        for arc in arcs:
            noiseL[sat[0]] += moving_median_residual(arc)
    for sat, ser in per_sat_P.items():
        ser.sort()
        vals = [v for _, v in ser]
        if len(vals) > 120:
            noiseP[sat[0]] += moving_median_residual(ser)
            isb[sat[0]].append(st.median(vals))

    # ---- discontinuity report ----
    events.sort()
    with open(REPORT, 'w') as f:
        f.write("utc_time,sat,constellation,jump_m,jump_cycles,dt_s\n")
        for e, sat, jm, jc, dt in events:
            ts = datetime.fromtimestamp(e, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{ts},{sat},{sat[0]},{jm:.3f},{jc:.2f},{dt}\n")
    span_h = (epochs[-1]-epochs[0])/3600
    print(f"\nPhase discontinuities logged: {len(events)} in {span_h:.1f} h "
          f"-> {REPORT}")
    nby = defaultdict(int)
    half = sum(1 for _, s, _, jc, _ in events if abs(abs(jc) % 1 - 0.5) < 0.15)
    for _, sat, *_ in events: nby[sat[0]] += 1
    print(f"  per constellation: GPS {nby['G']}, Galileo {nby['E']}, BeiDou {nby['C']}"
          f" | near-half-cycle events: {half}")

    print(f"\nCombined PHASE noise (SD, split threshold {JUMP*100:.0f} cm):")
    for s, n in [('G','GPS'),('E','Galileo'),('C','BeiDou')]:
        v = [abs(x) for x in noiseL[s]]
        mad = st.median(v)*1.4826
        tail = 100*sum(1 for x in v if x > 0.02)/len(v)
        print(f"  {n:8s}: sigma {mad*1000:5.2f} mm | samples >20mm: {tail:.4f}% | max {max(v)*1000:.0f} mm")
    print("\nCombined CODE noise (SD):")
    for s, n in [('G','GPS'),('E','Galileo'),('C','BeiDou')]:
        mad = st.median([abs(x) for x in noiseP[s]])*1.4826
        print(f"  {n:8s}: {mad*100:6.1f} cm  (n={len(noiseP[s])})")
    g = st.median(isb['G'])
    print("\nInter-constellation code bias (w.r.t. GPS):")
    for s, n in [('E','Galileo'),('C','BeiDou')]:
        med = st.median(isb[s])
        disp = st.median([abs(x-med) for x in isb[s]])
        print(f"  {n} - GPS: {med-g:+.2f} m  (between-satellite spread {disp:.2f} m)")

    # ---- figure (English) ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(16, 4.3))
    t00 = epochs[0]
    colors = {'G': 'tab:blue', 'E': 'tab:orange', 'C': 'tab:green'}
    names = {'G': 'GPS', 'E': 'Galileo', 'C': 'BeiDou'}
    seen = set()
    for sat in sorted(arcs_by_sat):
        c = colors[sat[0]]
        lab = names[sat[0]] if sat[0] not in seen else None
        seen.add(sat[0])
        for arc in arcs_by_sat[sat]:
            r = moving_median_residual(arc)
            tt = [(e-t00)/60 for e, _ in arc]
            rr = [x*1000 for x in r]
            a0.plot(tt, rr, lw=0.3, alpha=0.5, color=c, label=lab)
            a1.plot(tt, rr, lw=0.3, alpha=0.5, color=c)
            lab = None
    a0.set_xlabel('Minutes'); a0.set_ylabel('Detrended phase SD [mm]')
    a0.set_title(f'Full range — {len(arcs_by_sat)} satellites')
    a0.legend(fontsize=8); a0.grid(alpha=0.3)
    a1.set_xlabel('Minutes'); a1.set_ylim(-22, 22)
    a1.set_title('Zoom: noise floor'); a1.grid(alpha=0.3)
    for s in ('G', 'E', 'C'):
        a2.hist([x for x in noiseP[s] if abs(x) < 4], bins=100, alpha=0.5,
                label=names[s], color=colors[s], density=True)
    a2.set_xlabel('Detrended code SD [m]'); a2.set_ylabel('Density')
    a2.set_title('Combined code noise'); a2.legend(); a2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE, dpi=150)
    print("\nfigure saved")