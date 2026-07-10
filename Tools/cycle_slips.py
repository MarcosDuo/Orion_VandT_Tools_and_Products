#!/usr/bin/env python3
"""
zb_cycle_slips.py — Carrier-phase continuity and cycle-slip analysis
(zero-baseline test, step 4 of the pipeline).

Method (single-frequency receivers, so no geometry-free combination):

  1. Arc segmentation: for each satellite, contiguous runs of L1 phase
     observations (gap > 1.5 * interval starts a new arc).

  2. Phase-Doppler consistency test: between consecutive epochs,
         r(t) = L(t) - L(t-1) + 0.5 * (D(t) + D(t-1)) * dt   [cycles]
     The Doppler predicts the phase change (RINEX sign convention:
     dL/dt = -D). Geometry, satellite clock and ionosphere are common
     to both terms and cancel to first order; a cycle slip appears as a
     jump of |r| >= 1 cycle.

  3. Common-mode (receiver clock) removal: the receiver clock/oscillator
     dynamics affect all satellites identically, so the per-epoch median
     of r over all satellites is subtracted (same median-based approach
     as the Doppler validation of the orbital test).

  4. Detection threshold: |r_clean| > max(0.8 cycles, 6 * MAD_sat),
     with MAD the robust noise estimate of each satellite's residual.

  5. Cross-checks: receiver LLI flags (RINEX loss-of-lock bit 0) and a
     code-minus-carrier (CMC) jump test (|d(P - lambda*L)| > 5 m)
     that confirms large slips independently of the Doppler.

Outputs: per-receiver report (arcs, slips by method, slips/sat-hour)
and residual figures.

Usage:
    python zb_cycle_slips.py --obs FILE.obs --label "Orion" --out-dir results/slips_orion
"""
import argparse, math
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

FIRST_BAND = {'G':'1', 'E':'1', 'C':'2'}
C_LIGHT = 299792458.0
FREQ = {'G': 1575.42e6, 'E': 1575.42e6, 'C': 1561.098e6}   # L1 / E1 / B1I
LAMBDA = {s: C_LIGHT/f for s, f in FREQ.items()}


def parse_obs(path):
    """sat -> list of (t, L[cycles], D[Hz], P[m], LLI)"""
    types = {}
    data = defaultdict(list)
    with open(path, encoding='latin1') as f:
        for line in f:
            if "SYS / # / OBS TYPES" in line and line[0] in FIRST_BAND:
                types.setdefault(line[0], []).extend(line[7:60].split())
            if "END OF HEADER" in line:
                break
        idx = {}
        for sysc, tps in types.items():
            b = FIRST_BAND[sysc]
            def find(pref):
                for i, t in enumerate(tps):
                    if t.startswith(pref + b):
                        return i
                return None
            idx[sysc] = (find('C'), find('L'), find('D'))
        cur_t = None
        for line in f:
            if line.startswith('>'):
                p = line.split()
                cur_t = datetime(int(p[1]),int(p[2]),int(p[3]),int(p[4]),int(p[5])) \
                        + timedelta(seconds=float(p[6]))
            elif cur_t is not None and line[0] in idx:
                sysc = line[0]
                sat = line[:3].replace(' ','0')
                ic, il, idd = idx[sysc]
                def fld(j):
                    return line[3+16*j:3+16*j+14] if j is not None else ''
                def num(s):
                    s = s.strip()
                    try: return float(s)
                    except ValueError: return None
                L = num(fld(il)); D = num(fld(idd)); P = num(fld(ic))
                lli = 0
                if il is not None and len(line) > 3+16*il+14:
                    ch = line[3+16*il+14]
                    if ch.strip().isdigit():
                        lli = int(ch)
                if L is not None:
                    data[sat].append((cur_t, L, D, P, lli))
    return data


def analyze(data, interval, label, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = min(v[0][0] for v in data.values())

    # ---- pass 1: raw phase-Doppler residuals per satellite, in METRES ----
    # The residual is scaled by the wavelength so that receiver clock
    # events (e.g. millisecond clock jumps) are identical across
    # constellations of different carrier frequency and cancel exactly
    # in the common-mode removal below.
    raw = defaultdict(list)   # sat -> list of (t, r_metres)
    arcs = defaultdict(int)
    for sat, rows in data.items():
        rows.sort(key=lambda r: r[0])
        arcs[sat] += 1
        lam = LAMBDA[sat[0]]
        for a, b in zip(rows, rows[1:]):
            dt = (b[0]-a[0]).total_seconds()
            if dt > 1.5*interval:
                arcs[sat] += 1
                continue
            if a[2] is None or b[2] is None:
                continue
            r = (b[1] - a[1] + 0.5*(a[2]+b[2])*dt) * lam
            raw[sat].append((b[0], r))

    # ---- pass 2: per-epoch common-mode (receiver clock) removal ----
    by_epoch = defaultdict(list)
    for sat, lst in raw.items():
        for t, r in lst:
            by_epoch[t].append(r)
    med = {t: sorted(v)[len(v)//2] for t, v in by_epoch.items()}
    # receiver clock events: epochs whose common mode exceeds 1000 m
    clock_events = sorted(t for t, m in med.items() if abs(m) > 1000)
    # back to cycles per satellite after clock removal
    clean = {sat: [(t, (r - med[t])/LAMBDA[sat[0]]) for t, r in lst]
             for sat, lst in raw.items()}

    # ---- pass 3: detection ----
    def mad(v):
        m = sorted(v)[len(v)//2]
        return sorted(abs(x-m) for x in v)[len(v)//2]
    # Epochs at receiver clock events are excluded from slip detection:
    # the epoch re-labelling makes the nominal dt differ from the true
    # elapsed time, which leaves per-satellite residuals of up to
    # range-rate * step that are NOT cycle slips.
    skip = set(clock_events)
    slips_dopp = []
    for sat, lst in clean.items():
        if len(lst) < 20: continue
        rs = [r for t, r in lst if t not in skip]
        if len(rs) < 20: continue
        bias = sorted(rs)[len(rs)//2]     # per-satellite Doppler bias
        thr = max(0.8, 6*1.4826*mad(rs))
        for t, r in lst:
            if t in skip: continue
            if abs(r - bias) > thr:
                slips_dopp.append((sat, t, r - bias))

    # LLI flags (bit 0 = loss of lock)
    slips_lli = [(sat, r[0]) for sat, rows in data.items() for r in rows if r[4] & 1]

    # CMC jump confirmation
    slips_cmc = []
    for sat, rows in data.items():
        lam = LAMBDA[sat[0]]
        dcmc = []
        prev = None
        for t, L, D, P, lli in rows:
            if P is None: prev = None; continue
            cmc = P - lam*L
            if prev is not None and (t-prev[0]).total_seconds() <= 1.5*interval:
                dcmc.append((t, cmc - prev[1]))
            prev = (t, cmc)
        if len(dcmc) < 20: continue
        v = [x for _, x in dcmc]
        m = sorted(v)[len(v)//2]
        thr = max(5.0, 6*1.4826*mad(v))
        for t, x in dcmc:
            if abs(x - m) > thr:
                slips_cmc.append((sat, t, x))

    # ---- report ----
    n_arcs = sum(arcs.values())
    sat_hours = sum(len(v) for v in data.values()) * interval / 3600
    rep = []
    rep.append(f"CYCLE SLIP / PHASE CONTINUITY — {label}")
    rep.append("="*60)
    rep.append(f"Satellites with L1 phase: {len(data)}  |  tracking arcs: {n_arcs}")
    rep.append(f"Observed satellite-hours: {sat_hours:.1f}")
    rep.append(f"Receiver clock events (common-mode jump > 1 km): {len(clock_events)}")
    for t in clock_events[:8]:
        rep.append(f"    {t}   ({med[t]/299792458*1000:+.3f} ms)")
    all_r = [abs(r) for lst in clean.values() for _, r in lst]
    all_r.sort()
    rep.append(f"Phase-Doppler residual (clock-corrected): median |r| {all_r[len(all_r)//2]:.3f} cy, "
               f"P95 {all_r[int(0.95*len(all_r))]:.3f} cy, P99.9 {all_r[int(0.999*len(all_r))]:.3f} cy")
    rep.append("")
    rep.append(f"Slips detected (phase-Doppler test): {len(slips_dopp)}"
               f"  -> {len(slips_dopp)/sat_hours:.2f} per satellite-hour")
    rep.append(f"Receiver LLI loss-of-lock flags:     {len(slips_lli)}")
    rep.append(f"CMC jumps (adaptive 6*MAD, >=5 m):   {len(slips_cmc)}")
    rep.append("")
    if slips_dopp:
        rep.append("Detected slip events (satellite, epoch, residual [cycles]):")
        for sat, t, r in sorted(slips_dopp, key=lambda x: x[1])[:40]:
            has_lli = any(s==sat and abs((tt-t).total_seconds())<1.1 for s, tt in slips_lli)
            rep.append(f"  {sat}  {t}  {r:+9.2f} cy  {'[LLI]' if has_lli else ''}")
    text = "\n".join(rep)
    (out_dir/"cycle_slip_report.txt").write_text(text, encoding="utf-8")
    print(text)

    # ---- figure: binned envelope + slip rate (readable at any duration) ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    span_h = max((max(by_epoch) - t0).total_seconds()/3600, 0.1)
    bin_min = 1 if span_h <= 3 else 15          # bin width in minutes
    bins = defaultdict(list)
    for t, v in by_epoch.items():
        b = int((t - t0).total_seconds() // (60*bin_min))
        for sat_r in v:
            pass
    # per-epoch cleaned residuals pooled into time bins
    pool = defaultdict(list)
    for sat, lst in clean.items():
        for t, r in lst:
            pool[int((t - t0).total_seconds() // (60*bin_min))].append(r)
    bx, p05, p50, p95 = [], [], [], []
    for b in sorted(pool):
        v = sorted(pool[b])
        bx.append(b*bin_min/60)
        p05.append(v[int(0.05*len(v))]); p50.append(v[len(v)//2]); p95.append(v[int(0.95*len(v))])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                 gridspec_kw={'height_ratios': [2, 1]})
    a1.fill_between(bx, p05, p95, color='tab:blue', alpha=0.3, label='residual P5-P95')
    a1.plot(bx, p50, color='tab:blue', lw=1, label='median')
    for sat, t, r in slips_dopp:
        a1.plot((t-t0).total_seconds()/3600, max(min(r, 3), -3), 'x', ms=5, color='red', mew=1.2)
    for t in clock_events:
        a1.axvline((t-t0).total_seconds()/3600, color='purple', ls=':', lw=1)
    a1.plot([], [], 'x', color='red', label='slip candidate (clipped ±3 cy)')
    a1.plot([], [], ':', color='purple', label='receiver clock event')
    a1.set_ylim(-3.2, 3.2); a1.set_ylabel('Residual [cycles]')
    a1.set_title(f'Phase-Doppler consistency — {label}')
    a1.legend(loc='upper right', fontsize=8); a1.grid(alpha=0.3)
    counts = defaultdict(int)
    for sat, t, r in slips_dopp:
        counts[int((t-t0).total_seconds() // 3600)] += 1
    hrs = sorted(counts)
    a2.bar([h+0.5 for h in hrs], [counts[h] for h in hrs], width=0.9, color='tab:red', alpha=0.7)
    a2.set_xlabel('Hours since window start'); a2.set_ylabel('Slip candidates / h')
    a2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir/'phase_doppler_residual.png', dpi=150)
    return len(slips_dopp), sat_hours


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--label", default="receiver")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    data = parse_obs(a.obs)
    analyze(data, a.interval, a.label, a.out_dir)