#!/usr/bin/env python3
"""
spp_vs_arp_enu.py

Zero-baseline analysis: compare RTKLIB SPP solutions (.pos) against a
precise ARP reference position (from CSRS-PPP) in local ENU coordinates.

- Parses one or more rnx2rtkp .pos files (lat/lon/height format).
- Converts each epoch to East/North/Up errors w.r.t. the ARP.
- Prints bias, STD, RMS (per component, horizontal and 3D).
- Plots the ENU error time series of all receivers together.

Usage:
    python spp_vs_arp_enu.py orion_spp_1h.pos ublox_spp_1h.pos
    python spp_vs_arp_enu.py orion_spp.pos ublox_spp.pos -o zb_full_enu.png --labels "Orion B16-C1" "u-blox ZED-F9P"

The ARP reference is set below (CSRS-PPP, ITRF2020/IGS20 epoch 2024.9).
Note: plate motion between the ARP epoch (2024.9) and the observation
date (2026.5) is ~4 cm, negligible compared to metre-level SPP errors.
"""

import argparse
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# ARP reference position (CSRS-PPP report SEPT3280_30s, ITRF2020/IGS20 2024.9)
# 40 35' 28.86718" N, 3 42' 24.03787" W, h = 805.246 m
# ---------------------------------------------------------------------------
LAT_REF_DEG = 40 + 35 / 60 + 28.86718 / 3600     # +North
LON_REF_DEG = -(3 + 42 / 60 + 24.03787 / 3600)   # West -> negative
H_REF = 805.246                                  # ellipsoidal height [m]

# WGS84 ellipsoid
A = 6378137.0
F = 1 / 298.257223563
E2 = F * (2 - F)


GPS_EPOCH = datetime(1980, 1, 6)


def parse_pos(path):
    """Parse an rnx2rtkp .pos file (lat/lon/height solution format).

    Handles both rnx2rtkp time formats transparently:
      - GPS week/tow:   "2424 288000.000  40.59 ..."
      - date and time:  "2026/06/24 07:00:00.000  40.59 ..."
    Returns a list of tuples (t_seconds, lat_deg, lon_deg, h, Q, ns),
    where t_seconds is an absolute timestamp (seconds since GPS epoch)
    so that files with different formats share the same time axis.
    """
    epochs = []
    with open(path) as f:
        for line in f:
            if line.startswith('%'):
                continue
            p = line.split()
            if len(p) < 8:
                continue
            try:
                if '/' in p[0]:
                    # date/time format
                    d = datetime.strptime(p[0], "%Y/%m/%d")
                    hh, mm, ss = p[1].split(':')
                    t = (d - GPS_EPOCH).total_seconds() \
                        + int(hh)*3600 + int(mm)*60 + float(ss)
                else:
                    # GPS week / tow format
                    t = int(p[0]) * 604800 + float(p[1])
                lat, lon, h = float(p[2]), float(p[3]), float(p[4])
                Q, ns = int(p[5]), int(p[6])
            except ValueError:
                continue
            epochs.append((t, lat, lon, h, Q, ns))
    return epochs


def to_enu(epochs):
    """Convert geodetic epochs to ENU errors w.r.t. the ARP reference.

    Uses the small-offset approximation with the meridian (M) and prime
    vertical (N) radii of curvature evaluated at the reference latitude.
    Valid for offsets far below 1 km, which is always the case here.
    """
    lat0 = math.radians(LAT_REF_DEG)
    lon0 = math.radians(LON_REF_DEG)
    n0 = A / math.sqrt(1 - E2 * math.sin(lat0) ** 2)
    m0 = A * (1 - E2) / (1 - E2 * math.sin(lat0) ** 2) ** 1.5

    t, e, n, u = [], [], [], []
    for tow, lat, lon, h, _q, _ns in epochs:
        t.append(tow)
        n.append((math.radians(lat) - lat0) * (m0 + H_REF))
        e.append((math.radians(lon) - lon0) * (n0 + H_REF) * math.cos(lat0))
        u.append(h - H_REF)
    return t, e, n, u


def print_stats(name, e, n, u):
    mean = lambda v: sum(v) / len(v)
    rms = lambda v: math.sqrt(sum(x * x for x in v) / len(v))
    std = lambda v: math.sqrt(sum((x - mean(v)) ** 2 for x in v) / len(v))
    h2d = [math.hypot(a, b) for a, b in zip(e, n)]
    e3d = [math.sqrt(a * a + b * b + c * c) for a, b, c in zip(e, n, u)]

    print(f"--- {name} ({len(e)} epochs) ---")
    print(f"  Bias (mean):  E {mean(e):+7.3f}  N {mean(n):+7.3f}  U {mean(u):+7.3f} m")
    print(f"  STD:          E {std(e):7.3f}  N {std(n):7.3f}  U {std(u):7.3f} m")
    print(f"  RMS:          E {rms(e):7.3f}  N {rms(n):7.3f}  U {rms(u):7.3f} m")
    print(f"  RMS horizontal: {rms(h2d):.3f} m | RMS 3D: {rms(e3d):.3f} m")
    print(f"  Max horizontal: {max(h2d):.3f} m | Max 3D: {max(e3d):.3f} m")
    print()


def main():
    ap = argparse.ArgumentParser(description="SPP vs ARP reference in ENU")
    ap.add_argument("pos_files", nargs="+", help=".pos files from rnx2rtkp")
    ap.add_argument("-o", "--output", default="spp_vs_arp_enu.png",
                    help="output figure filename (PNG)")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="legend labels (one per .pos file)")
    ap.add_argument("--max-3d", type=float, default=None,
                    help="exclude epochs with 3D error above this threshold [m] "
                         "(outlier filter; excluded count is reported)")
    args = ap.parse_args()

    labels = args.labels
    if labels is None:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in args.pos_files]

    colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple']

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    comp_labels = ['East error [m]', 'North error [m]', 'Up error [m]']

    t0_global = None
    for i, (path, label) in enumerate(zip(args.pos_files, labels)):
        epochs = parse_pos(path)
        if not epochs:
            print(f"WARNING: no solutions found in {path}")
            continue
        t, e, n, u = to_enu(epochs)
        if args.max_3d is not None:
            keep = [i for i in range(len(t))
                    if (e[i]**2 + n[i]**2 + u[i]**2) <= args.max_3d**2]
            n_excl = len(t) - len(keep)
            if n_excl:
                print(f"{label}: {n_excl} epochs excluded "
                      f"(3D error > {args.max_3d:g} m, {100*n_excl/len(t):.3f}%)")
            t = [t[i] for i in keep]; e = [e[i] for i in keep]
            n = [n[i] for i in keep]; u = [u[i] for i in keep]
        if t0_global is None:
            t0_global = t[0]
        trel = [(x - t0_global) / 3600.0 for x in t]   # hours since first epoch

        print_stats(label, e, n, u)

        for ax, y in zip(axes, (e, n, u)):
            ax.plot(trel, y, lw=0.5, color=colors[i % len(colors)], label=label)

    for ax, lab in zip(axes, comp_labels):
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0].legend(loc='upper right', ncol=2)
    axes[0].set_title('SPP error vs. ARP reference (zero-baseline test)')
    axes[2].set_xlabel('Time since first epoch [h]')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Figure saved to: {args.output}")


if __name__ == "__main__":
    main()