#!/usr/bin/env python3
"""
availability_cn0.py  (v2 — zero-baseline ready)

Availability and signal-quality (C/N0) analysis. Works for both the
orbital-simulation and the zero-baseline datasets:

  - Obs types are read from the RINEX header (handles convbin and
    gfzrnx-reordered files; first-frequency SNR is selected per system).
  - Epochs are full datetimes (multi-day recordings do not wrap).
  - Fix availability matches .pos epochs to obs epochs by nearest second,
    so a decimated obs can be combined with a full-rate .pos (only obs
    epochs are evaluated).
  - .pos time formats week/tow and yyyy/mm/dd hh:mm:ss both supported.

RINEX mode:
    python availability_cn0.py --obs orion.obs --pos orion_spp.pos \
        --label "Orion" --out-dir results/avail_orion

NMEA mode (PX1125S): unchanged
    python availability_cn0.py --nmea PX_NMEA.out --label "PX1125S" --out-dir results/avail_px
"""

import argparse
import math
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

SYS_NAMES = {"G": "GPS", "E": "Galileo", "C": "BeiDou", "R": "GLONASS", "J": "QZSS", "S": "SBAS"}
# first-frequency band per system (RINEX 3 band digit)
FIRST_BAND = {"G": "1", "E": "1", "R": "1", "J": "1", "S": "1", "C": "2"}
GPS_EPOCH = datetime(1980, 1, 6)


def stats(values):
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan")}
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return {"n": n, "mean": mean, "std": math.sqrt(var), "min": min(values), "max": max(values)}

# ---------------------------------------------------------------------------
# RINEX obs parsing (header-driven)
# ---------------------------------------------------------------------------

def parse_rinex_obs(path):
    """Returns:
       epochs: sorted list of datetimes
       tracked: dict datetime -> set of sat_ids
       cn0: dict (datetime, sat_id) -> C/N0 [dB-Hz] (first frequency)
    """
    obs_types = {}
    epochs = []
    tracked = defaultdict(set)
    cn0 = {}
    snr_idx = {}   # sysc -> column index of first-frequency SNR
    with open(path, encoding="latin1") as f:
        for line in f:
            if "SYS / # / OBS TYPES" in line and line[0].strip():
                sysc = line[0]
                obs_types.setdefault(sysc, []).extend(line[7:60].split())
            if "END OF HEADER" in line:
                break
        for sysc, types in obs_types.items():
            band = FIRST_BAND.get(sysc, "1")
            cands = [i for i, t in enumerate(types) if t.startswith("S" + band)]
            if not cands:  # fall back to any SNR observable
                cands = [i for i, t in enumerate(types) if t.startswith("S")]
            if cands:
                snr_idx[sysc] = cands[0]
        cur = None
        for line in f:
            if line.startswith(">"):
                p = line.split()
                cur = datetime(int(p[1]), int(p[2]), int(p[3]),
                               int(p[4]), int(p[5])) + timedelta(seconds=float(p[6]))
                epochs.append(cur)
                continue
            sysc = line[0] if line else ""
            if sysc in snr_idx and len(line) > 4 and cur is not None:
                sat3 = line[0:3]
                try:
                    prn = int(sat3[1:3])
                except ValueError:
                    continue
                sat_id = f"{sysc}{prn:02d}"
                tracked[cur].add(sat_id)
                j = snr_idx[sysc]
                field = line[3 + 16 * j:3 + 16 * j + 14].strip()
                if field:
                    try:
                        cn0[(cur, sat_id)] = float(field)
                    except ValueError:
                        pass
    return sorted(set(epochs)), tracked, cn0

# ---------------------------------------------------------------------------
# .pos parsing (both time formats)
# ---------------------------------------------------------------------------

def parse_pos_fixes(path):
    """Returns list of (datetime, q, ns)."""
    rows = []
    with open(path, "r", encoding="latin1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            p = line.split()
            if len(p) < 7:
                continue
            try:
                if "/" in p[0]:
                    d = datetime.strptime(p[0], "%Y/%m/%d")
                    hh, mm, ss = p[1].split(":")
                    t = d + timedelta(hours=int(hh), minutes=int(mm), seconds=float(ss))
                    q, ns = int(p[5]), int(p[6])
                else:
                    t = GPS_EPOCH + timedelta(weeks=int(p[0]), seconds=float(p[1]))
                    q, ns = int(p[5]), int(p[6])
            except ValueError:
                continue
            rows.append((t, q, ns))
    return rows

# ---------------------------------------------------------------------------
# NMEA parsing (unchanged from v1)
# ---------------------------------------------------------------------------

def parse_nmea(path):
    n_gga = 0
    n_fix = 0
    sats_used = []
    sats_in_view = []
    first_time = None
    last_time = None
    first_fix_time = None
    with open(path, encoding="latin1", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("$GPGGA") or line.startswith("$GNGGA"):
                fields = line.split(",")
                if len(fields) < 8:
                    continue
                n_gga += 1
                t = fields[1]
                if t:
                    if first_time is None:
                        first_time = t
                    last_time = t
                try:
                    quality = int(fields[6]) if fields[6] else 0
                except ValueError:
                    quality = 0
                if quality > 0:
                    n_fix += 1
                    if first_fix_time is None:
                        first_fix_time = t
                try:
                    sats_used.append(int(fields[7]))
                except ValueError:
                    pass
            elif line[1:6].endswith("GSV") and line.startswith("$"):
                fields = line.split(",")
                if len(fields) > 3 and fields[2] == "1":
                    try:
                        sats_in_view.append(int(fields[3]))
                    except ValueError:
                        pass
    return {"n_gga": n_gga, "n_fix": n_fix,
            "availability_pct": 100.0 * n_fix / n_gga if n_gga else float("nan"),
            "first_time": first_time, "last_time": last_time,
            "first_fix_time": first_fix_time,
            "sats_used": stats(sats_used), "sats_in_view": stats(sats_in_view)}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--obs", default=None)
    parser.add_argument("--pos", default=None)
    parser.add_argument("--nmea", default=None)
    parser.add_argument("--valid-q", default="1,2,5,6")
    parser.add_argument("--label", default="receiver")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tmax-sod", type=float, default=None,
                        help="Last valid epoch as seconds-of-day (single-day data). "
                             "Epochs after this instant are excluded from all statistics "
                             "and plots; the exclusion is documented in the report. "
                             "Use when the recording extended beyond the simulated "
                             "scenario end (recording overrun).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    valid_q = tuple(int(x) for x in args.valid_q.split(","))

    report = []
    report.append(f"AVAILABILITY & SIGNAL QUALITY - {args.label}")
    report.append("=" * 60)

    if args.nmea:
        r = parse_nmea(args.nmea)
        report.append(f"Input NMEA file: {args.nmea}")
        report.append("")
        report.append(f"GGA sentences: {r['n_gga']}")
        report.append(f"GGA with valid fix (quality > 0): {r['n_fix']}")
        report.append(f"FIX AVAILABILITY: {r['availability_pct']:.2f} %")
        report.append(f"Time span (GGA hhmmss.sss): {r['first_time']} -> {r['last_time']}")
        ttff = r['first_fix_time'] if r['first_fix_time'] else "never (no fix in entire recording)"
        report.append(f"First fix at: {ttff}")
        report.append("")
        su = r["sats_used"]; sv = r["sats_in_view"]
        report.append(f"Satellites USED in solution (GGA field 8): mean {su['mean']:.1f}, "
                      f"min {su['min']}, max {su['max']}")
        report.append(f"Satellites IN VIEW (GSV): mean {sv['mean']:.1f}, min {sv['min']}, max {sv['max']}")
        report.append("")
        report.append("Note: no raw observables are available from this receiver, so no C/N0 or")
        report.append("per-satellite tracking analysis is possible - this limitation is itself a")
        report.append("finding for receiver-selection purposes.")

    elif args.obs:
        epochs, tracked, cn0 = parse_rinex_obs(args.obs)

        # ---- optional time cutoff (recording overrun after scenario end) ----
        n_excluded = 0
        if args.tmax_sod is not None:
            def sod(e):
                return e.hour * 3600 + e.minute * 60 + e.second + e.microsecond / 1e6
            n_before = len(epochs)
            keep = [e for e in epochs if sod(e) <= args.tmax_sod]
            excluded = [e for e in epochs if sod(e) > args.tmax_sod]
            n_excluded = len(excluded)
            epochs = keep
            tracked = {e: tracked[e] for e in keep if e in tracked}
            cn0 = {(e, s): v for (e, s), v in cn0.items() if e in set(keep)}
            report.append(f"TIME CUTOFF APPLIED: epochs after SoD {args.tmax_sod:.0f} s excluded")
            report.append(f"  ({n_excluded} of {n_before} epochs removed: recording continued after")
            report.append(f"  the simulated scenario ended; those epochs show loss of all simulated")
            report.append(f"  signals and residual noise-floor tracking, and do not describe")
            report.append(f"  receiver behaviour under test conditions)")
            report.append("")

        n_epochs = len(epochs)
        span = (epochs[-1] - epochs[0]).total_seconds() if n_epochs > 1 else 0.0
        report.append(f"Input RINEX obs: {args.obs}")
        if args.pos:
            report.append(f"Input RTKLIB pos: {args.pos}")
        report.append("")
        report.append(f"Observation epochs: {n_epochs}  (span {span:.0f} s, "
                      f"{epochs[0]} -> {epochs[-1]})")

        all_sats = set()
        for e in epochs:
            all_sats |= tracked.get(e, set())
        all_systems = sorted(set(s[0] for s in all_sats))
        per_sys_counts = {sysc: [] for sysc in all_systems}
        for e in epochs:
            sats = tracked.get(e, set())
            by_sys = defaultdict(int)
            for s in sats:
                by_sys[s[0]] += 1
            for sysc in all_systems:
                per_sys_counts[sysc].append(by_sys.get(sysc, 0))
        total_counts = [len(tracked.get(e, set())) for e in epochs]

        report.append("")
        report.append("SATELLITES TRACKED (per epoch):")
        st = stats(total_counts)
        report.append(f"  TOTAL: mean {st['mean']:.1f}, min {st['min']}, max {st['max']}")
        for sysc in sorted(per_sys_counts.keys()):
            s = stats(per_sys_counts[sysc])
            n_unique = len([x for x in all_sats if x[0] == sysc])
            report.append(f"  {SYS_NAMES.get(sysc, sysc)}: mean {s['mean']:.1f}, min {s['min']}, "
                          f"max {s['max']}  ({n_unique} unique satellites)")

        if args.pos:
            pos_rows = parse_pos_fixes(args.pos)
            # match pos epochs to obs epochs by nearest second
            pos_by_sec = {}
            for t, q, ns in pos_rows:
                key = round((t - epochs[0]).total_seconds())
                pos_by_sec[key] = (q, ns)
            n_valid = 0
            first_fix_dt = None
            for e in epochs:
                key = round((e - epochs[0]).total_seconds())
                row = pos_by_sec.get(key)
                if row and row[0] in valid_q:
                    n_valid += 1
                    if first_fix_dt is None:
                        first_fix_dt = e
            availability = 100.0 * n_valid / n_epochs if n_epochs else float("nan")
            report.append("")
            report.append("FIX AVAILABILITY (RTKLIB SPP, evaluated on obs epochs):")
            report.append(f"  Obs epochs with a valid fix (Q in {valid_q}): {n_valid} of {n_epochs}")
            report.append(f"  FIX AVAILABILITY: {availability:.2f} %")
            if first_fix_dt is not None:
                ttff = (first_fix_dt - epochs[0]).total_seconds()
                report.append(f"  Time to first fix (from first obs epoch): {ttff:.1f} s")

        cn0_by_sat = defaultdict(list)
        cn0_by_sys = defaultdict(list)
        for (e, sat), v in cn0.items():
            cn0_by_sat[sat].append(v)
            cn0_by_sys[sat[0]].append(v)
        report.append("")
        report.append("C/N0 [dB-Hz] per constellation (first frequency):")
        for sysc in sorted(cn0_by_sys.keys()):
            s = stats(cn0_by_sys[sysc])
            report.append(f"  {SYS_NAMES.get(sysc, sysc)}: mean {s['mean']:.1f}, "
                          f"min {s['min']:.0f}, max {s['max']:.0f}  (n={s['n']})")
        report.append("")
        report.append("C/N0 [dB-Hz] per satellite (mean / min / max, n):")
        for sat in sorted(cn0_by_sat.keys()):
            s = stats(cn0_by_sat[sat])
            report.append(f"  {sat}: {s['mean']:.1f} / {s['min']:.0f} / {s['max']:.0f}  (n={s['n']})")

        # ------------- plots -------------
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t0 = epochs[0]
        hours = span > 7200
        t_axis = [(e - t0).total_seconds() / (3600 if hours else 1) for e in epochs]
        xlabel = "Elapsed time [h]" if hours else "Elapsed time [s]"

        fig, ax = plt.subplots(figsize=(9, 5))
        for sysc in sorted(per_sys_counts.keys()):
            ax.plot(t_axis, per_sys_counts[sysc], label=SYS_NAMES.get(sysc, sysc), linewidth=1.2)
        ax.plot(t_axis, total_counts, label="Total", color="black", linewidth=1.6, linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Satellites tracked")
        ax.set_title(f"Satellites tracked per epoch - {args.label}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(out_dir / "sats_per_epoch.png", dpi=150)
        plt.close(fig)

        sat_series = defaultdict(list)
        for (e, sat), v in cn0.items():
            sat_series[sat].append(((e - t0).total_seconds() / (3600 if hours else 1), v))
        fig, ax = plt.subplots(figsize=(10, 6))
        for sat in sorted(sat_series.keys()):
            pts = sorted(sat_series[sat])
            ax.plot([x for x, _ in pts], [v for _, v in pts], linewidth=0.9, alpha=0.8, label=sat)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("C/N0 [dB-Hz]")
        ax.set_title(f"C/N0 per satellite - {args.label}")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "cn0_per_satellite.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 5))
        sats_sorted = sorted(cn0_by_sat.keys())
        means = [stats(cn0_by_sat[s])["mean"] for s in sats_sorted]
        sys_colors = {"G": "tab:blue", "E": "tab:orange", "C": "tab:green",
                      "R": "tab:red", "J": "tab:purple", "S": "tab:gray"}
        colors = [sys_colors.get(s[0], "tab:gray") for s in sats_sorted]
        ax.bar(range(len(sats_sorted)), means, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(sats_sorted)))
        ax.set_xticklabels(sats_sorted, rotation=70, fontsize=7)
        ax.set_ylabel("Mean C/N0 [dB-Hz]")
        ax.set_title(f"Mean C/N0 per satellite - {args.label}")
        handles = [plt.Rectangle((0, 0), 1, 1, color=sys_colors[s])
                   for s in sorted(set(x[0] for x in sats_sorted)) if s in sys_colors]
        labels = [SYS_NAMES.get(s, s) for s in sorted(set(x[0] for x in sats_sorted))]
        ax.legend(handles, labels, loc="lower right")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "cn0_mean_bar.png", dpi=150)
        plt.close(fig)

    else:
        raise RuntimeError("Provide either --obs (RINEX mode) or --nmea (NMEA mode).")

    report_path = out_dir / "availability_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()