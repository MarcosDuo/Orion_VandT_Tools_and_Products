#!/usr/bin/env python3
"""
doppler_validation.py

Validates the receiver's measured Doppler (RINEX D1C/D1I observable) against
the Doppler that Skydel actually simulated for each satellite (from the
per-satellite truth CSV logs in Logs_Skydel/<scenario>/<SIGNAL> <PRN>.csv).

Why a per-epoch common-mode correction is applied
--------------------------------------------------
A raw Doppler measurement contains the receiver clock DRIFT as a common term
added to every satellite at a given epoch (the Doppler-domain equivalent of
the receiver clock BIAS in the position domain). Comparing raw measured
Doppler directly against the true geometric Doppler therefore mixes two
different things: how well the receiver tracks each satellite, and the
(unknown, time-varying) clock drift. This script estimates that common term
at each epoch as the across-satellite mean residual, subtracts it, and
reports both the raw and the bias-removed residuals so the distinction stays
visible and defensible.

Usage:
  python doppler_validation.py \
      --obs orion_gps_gal.obs \
      --truth-dir Logs_Skydel/Orbit_s6a \
      --label "Orion GPS+GAL" \
      --out-dir results/doppler_gps_gal

  python doppler_validation.py \
      --obs orion_gps_gal_bds.obs \
      --truth-dir Logs_Skydel/Orbit_s6a_withBDS \
      --label "Orion GPS+GAL+BDS elevmask off" \
      --out-dir results/doppler_gps_gal_bds

--truth-dir must contain files named "L1CA PP.csv" (GPS), "E1 PP.csv"
(Galileo) and/or "B1 PP.csv" (BeiDou), PP being the 2-digit PRN - i.e. exactly
the Skydel Raw Logging folder structure used in this project's repo.
"""

import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict

OBS_TYPES = {"G": ["C1C", "L1C", "D1C", "S1C"], "E": ["C1C", "L1C", "D1C", "S1C"], "J": ["C1C", "L1C", "D1C", "S1C"], "R": ["C1C", "L1C", "D1C", "S1C"], "C": ["C1I", "L1I", "D1I", "S1I"]}
SYS_TO_SIGNAL = {"G": "L1CA", "E": "E1", "C": "B1"}

def stats(values):
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "rms": float("nan")}
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)
    rms = math.sqrt(sum(v * v for v in values) / n)
    return {"n": n, "mean": mean, "std": std, "rms": rms}

def parse_rinex_doppler(path):
    """Returns dict (epoch_sow_like, sat_id) -> Doppler [Hz], and the sorted
    list of distinct epoch tags in file order (using GPST hh:mm:ss.sss as the
    epoch key, consistent within a single day)."""
    result = {}
    epochs = []
    header_done = False
    cur_epoch = None
    with open(path, encoding="latin1") as f:
        for line in f:
            if not header_done:
                if "END OF HEADER" in line:
                    header_done = True
                continue
            if line.startswith(">"):
                p = line.split()
                h, m = int(p[4]), int(p[5])
                s = float(p[6])
                cur_epoch = h * 3600 + m * 60 + s
                epochs.append(cur_epoch)
                continue
            sysc = line[0]
            if sysc in OBS_TYPES and len(line) > 4:
                sat = line[0:3]
                types = OBS_TYPES[sysc]
                if "D1C" in types:
                    idx = types.index("D1C")
                elif "D1I" in types:
                    idx = types.index("D1I")
                else:
                    continue
                start = 3 + 16 * idx
                field = line[start:start + 14].strip()
                if field:
                    try:
                        result[(cur_epoch, sat)] = float(field)
                    except ValueError:
                        pass
    return result, epochs

def load_truth_doppler(truth_dir):
    """Loads every 'SIGNAL PP.csv' file in truth_dir into
    dict sat_id ('G02','E04','C07',...) -> sorted list of (GPS_TOW, Doppler_Hz)."""
    truth_dir = Path(truth_dir)
    truth = {}
    for sysc, signal in SYS_TO_SIGNAL.items():
        for path in truth_dir.glob(f"{signal} *.csv"):
            prn = path.stem.split(" ")[-1]
            sat_id = f"{sysc}{int(prn):02d}"
            rows = []
            with open(path, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    rows.append((float(row["GPS TOW"]), float(row["Doppler Frequency (Hz)"])))
            rows.sort()
            truth[sat_id] = rows
    return truth

def interp_truth(rows, target_tow):
    lo = None
    for i in range(len(rows) - 1):
        if rows[i][0] <= target_tow <= rows[i + 1][0]:
            lo = i
            break
    if lo is None:
        return None
    (t0, v0), (t1, v1) = rows[lo], rows[lo + 1]
    span = t1 - t0
    f = (target_tow - t0) / span if span > 0 else 0.0
    return v0 + f * (v1 - v0)

def rinex_sat_to_id(sat3):
    """'G 2' or 'G02' -> 'G02', 'E12' -> 'E12', 'C 4' -> 'C04'."""
    sysc = sat3[0]
    prn = int(sat3[1:3])
    return f"{sysc}{prn:02d}"

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--obs", required=True, help="Receiver RINEX obs file")
    parser.add_argument("--truth-dir", required=True, help="Folder with Skydel per-satellite truth CSVs (Logs_Skydel/<scenario>)")
    parser.add_argument("--obs-start-sow", type=float, required=True, help="GPS SOW corresponding to hh:mm:ss=00:00:00 of the obs file's day "
                              "(printed in the .obs header as 'obs start'; use the SOW at 00:00:00, "
                              "i.e. obs_start_sow_from_header - hh*3600 - mm*60 - ss)")
    parser.add_argument("--label", default="scenario")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--edge-trim", type=int, default=3, help="Number of epochs to exclude at the END of each satellite tracking arc "
                              "(loss-of-lock transition; the receiver's Doppler tends to 'freeze' for "
                              "the last few epochs before dropping a satellite). Set 0 to disable.")
    parser.add_argument("--arc-gap", type=float, default=5.0, help="Minimum gap [s] between consecutive observations of a satellite for "
                              "them to be considered separate tracking arcs.")
    parser.add_argument("--max-example-plots", type=int, default=4, help="(kept for compatibility; the per-satellite plot now shows all satellites)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rinex_dop, epochs = parse_rinex_doppler(args.obs)
    truth = load_truth_doppler(args.truth_dir)

    # Build raw residuals: (epoch, sat) -> (tow, sat_id, d_rx, d_truth, raw_resid)
    per_epoch = defaultdict(list)  # epoch -> list of (sat_id, d_rx, d_truth, raw_resid)
    for (epoch, sat3), d_rx in rinex_dop.items():
        sat_id = rinex_sat_to_id(sat3)
        if sat_id not in truth:
            continue
        tow = args.obs_start_sow + epoch
        d_truth = interp_truth(truth[sat_id], tow)
        if d_truth is None:
            continue
        raw_resid = d_rx - d_truth
        per_epoch[epoch].append((sat_id, tow, d_rx, d_truth, raw_resid))

    if not per_epoch:
        raise RuntimeError("No matching (epoch, satellite) pairs between the RINEX obs and the "
                            "truth CSVs. Check --obs-start-sow and that --truth-dir matches the "
                            "scenario used to generate --obs.")

    # Per-epoch common-mode (clock drift) removal.
    # We use the MEDIAN across satellites, not the mean: a single satellite
    # with a wrong-sign or mismatched truth value (e.g. a PRN/ID mapping
    # issue between the RINEX satellite number and the Skydel truth log)
    # would otherwise drag the mean off and corrupt the correction for
    # every other, perfectly healthy, satellite at that epoch.
    def median(vals):
        s = sorted(vals)
        n = len(s)
        mid = n // 2
        return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2

    rows = []  # flat list: epoch, sat_id, tow, d_rx, d_truth, raw_resid, common_mode, corrected_resid
    for epoch, entries in per_epoch.items():
        common_mode = median([e[4] for e in entries])
        for sat_id, tow, d_rx, d_truth, raw_resid in entries:
            rows.append({"epoch": epoch, "sat_id": sat_id, "tow": tow, "d_rx": d_rx, "d_truth": d_truth, "raw_resid": raw_resid, "common_mode": common_mode, "corrected_resid": raw_resid - common_mode})

    all_rows = rows  # keep the full, unfiltered set for the CSV

    # --- Edge-of-arc trimming ---
    # Rationale: right before losing lock on a setting satellite, the receiver's
    # tracking loop can 'freeze' its Doppler output for the last few epochs
    # (observed in this project's data: measured Doppler changing at ~-2 Hz/s
    # while the true Doppler keeps evolving at ~-21 Hz/s). Those transition
    # epochs describe loss-of-lock behaviour, not steady-state Doppler quality,
    # so they are excluded from the statistics (but kept, flagged, in the CSV).
    # Exception: if an arc ends because the RECORDING ends (not because the
    # satellite was lost), its final epochs are normal data and are NOT trimmed.
    edge_trimmed_keys = set()  # (epoch, sat_id)
    if args.edge_trim > 0:
        global_last_epoch = max(r["epoch"] for r in all_rows)
        by_sat_epochs = defaultdict(list)
        for r in all_rows:
            by_sat_epochs[r["sat_id"]].append(r["epoch"])
        for sat_id, eps in by_sat_epochs.items():
            eps = sorted(set(eps))
            # split into arcs at gaps larger than --arc-gap
            arcs = [[eps[0]]]
            for e in eps[1:]:
                if e - arcs[-1][-1] > args.arc_gap:
                    arcs.append([e])
                else:
                    arcs[-1].append(e)
            for arc in arcs:
                # if the arc reaches the end of the recording, the satellite was
                # not lost -- do not trim
                if global_last_epoch - arc[-1] <= args.arc_gap:
                    continue
                for e in arc[-args.edge_trim:]:
                    edge_trimmed_keys.add((e, sat_id))

    ANOMALY_THRESHOLD_HZ = 50.0

    def classify(r):
        """0 = kept, 1 = outlier (>threshold), 2 = edge-of-arc trim."""
        if (r["epoch"], r["sat_id"]) in edge_trimmed_keys:
            return 2
        if abs(r["corrected_resid"]) > ANOMALY_THRESHOLD_HZ:
            return 1
        return 0

    good_rows = [r for r in all_rows if classify(r) == 0]
    bad_rows = [r for r in all_rows if classify(r) == 1]
    edge_rows = [r for r in all_rows if classify(r) == 2]
    n_excluded = len(bad_rows)
    n_edge = len(edge_rows)
    rows = good_rows

    # --- Write CSV (all points, with an exclusion flag, so raw data stays inspectable) ---
    csv_path = out_dir / "doppler_residuals.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch_s", "sat_id", "gps_tow", "d_rx_hz", "d_truth_hz", "raw_resid_hz", "common_mode_hz", "corrected_resid_hz", "excluded"])  # 0=kept, 1=outlier, 2=edge-of-arc trim
        for r in sorted(all_rows, key=lambda r: (r["epoch"], r["sat_id"])):
            w.writerow([f"{r['epoch']:.3f}", r["sat_id"], f"{r['tow']:.3f}", f"{r['d_rx']:.3f}", f"{r['d_truth']:.3f}", f"{r['raw_resid']:.3f}", f"{r['common_mode']:.3f}", f"{r['corrected_resid']:.3f}", str(classify(r))])

    # --- Stats ---
    raw_vals = [r["raw_resid"] for r in rows]
    corr_vals = [r["corrected_resid"] for r in rows]
    common_vals = sorted(set((r["epoch"], r["common_mode"]) for r in rows))
    common_only = [c for _, c in common_vals]

    s_raw = stats(raw_vals)
    s_corr = stats(corr_vals)
    s_common = stats(common_only)

    per_sat = defaultdict(list)
    for r in rows:
        per_sat[r["sat_id"]].append(r["corrected_resid"])
    per_sat_stats = {sat: stats(vals) for sat, vals in per_sat.items()}

    report_lines = []
    report_lines.append(f"DOPPLER VALIDATION - {args.label}")
    report_lines.append("=" * 60)
    report_lines.append(f"RINEX obs file: {args.obs}")
    report_lines.append(f"Truth directory: {args.truth_dir}")
    if n_edge:
        by_sat_edge = defaultdict(int)
        for r in edge_rows:
            by_sat_edge[r["sat_id"]] += 1
        report_lines.append("")
        report_lines.append(f"EDGE-OF-ARC TRIMMED: {n_edge} points (last {args.edge_trim} epochs of each "
                             f"tracking arc before loss of lock; arcs reaching the end of the recording "
                             f"are NOT trimmed):")
        for sat, n in sorted(by_sat_edge.items()):
            report_lines.append(f"  {sat}: {n} points trimmed")
        report_lines.append("  Rationale: the receiver's Doppler output tends to 'freeze' during the "
                             "last epochs before dropping a setting satellite (loss-of-lock transition). "
                             "Those points describe acquisition/loss behaviour, not steady-state Doppler "
                             "measurement quality. Flagged as excluded=2 in doppler_residuals.csv.")
    if n_excluded:
        by_sat = defaultdict(int)
        for r in bad_rows:
            by_sat[r["sat_id"]] += 1
        report_lines.append("")
        report_lines.append(f"EXCLUDED {n_excluded} individual (epoch, satellite) outlier points "
                             f"(|corrected residual| > {ANOMALY_THRESHOLD_HZ:.0f} Hz):")
        for sat, n in sorted(by_sat.items()):
            report_lines.append(f"  {sat}: {n} points excluded")
        report_lines.append("  These are isolated transient anomalies (e.g. a brief tracking glitch "
                             "or sign ambiguity), not representative of the satellite's overall "
                             "Doppler tracking quality - see doppler_residuals.csv for the raw, "
                             "unfiltered per-epoch values if you want to inspect them directly.")
    report_lines.append("")
    report_lines.append(f"Satellites matched (used in stats): {sorted(per_sat.keys())}")
    report_lines.append(f"Total (epoch, satellite) pairs: {len(rows)}")
    report_lines.append("")
    report_lines.append("Common-mode term (~receiver clock drift) [Hz] mean/std/rms:")
    report_lines.append(f"  {s_common['mean']:+.3f} / {s_common['std']:.3f} / {s_common['rms']:.3f}")
    report_lines.append("")
    report_lines.append("RAW residual (d_rx - d_truth), all satellites pooled [Hz] mean/std/rms:")
    report_lines.append(f"  {s_raw['mean']:+.3f} / {s_raw['std']:.3f} / {s_raw['rms']:.3f}")
    report_lines.append("")
    report_lines.append("CORRECTED residual (common-mode removed), all satellites pooled [Hz] mean/std/rms:")
    report_lines.append(f"  {s_corr['mean']:+.3f} / {s_corr['std']:.3f} / {s_corr['rms']:.3f}")
    report_lines.append("")
    report_lines.append("Per-satellite corrected residual stats [Hz] (mean/std/rms, n):")
    for sat in sorted(per_sat_stats.keys()):
        st = per_sat_stats[sat]
        report_lines.append(f"  {sat}: {st['mean']:+.3f} / {st['std']:.3f} / {st['rms']:.3f}  (n={st['n']})")

    report_path = out_dir / "doppler_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))

    # --- Plots ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = min(r["tow"] for r in rows)

    # Plot 1: common-mode (clock drift) term vs time
    fig, ax = plt.subplots(figsize=(9, 4))
    ep_sorted = sorted(per_epoch.keys())
    t_axis = [e - ep_sorted[0] for e in ep_sorted]
    cm_axis = [median([x[4] for x in per_epoch[e]]) for e in ep_sorted]
    ax.plot(t_axis, cm_axis, color="black", linewidth=1.2)
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("Common-mode Doppler [Hz]")
    ax.set_title(f"Estimated receiver clock-drift term - {args.label}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "doppler_common_mode.png", dpi=150)
    plt.close(fig)

    # Plot 2: corrected residuals for ALL satellites
    sats_to_plot = sorted(per_sat.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    for sat in sats_to_plot:
        sat_rows = sorted([r for r in rows if r["sat_id"] == sat], key=lambda r: r["tow"])
        t = [r["tow"] - t0 for r in sat_rows]
        v = [r["corrected_resid"] for r in sat_rows]
        ax.plot(t, v, label=sat, linewidth=0.9, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("Corrected Doppler residual [Hz]")
    ax.set_title(f"Doppler residual per satellite (clock-drift removed) - {args.label}")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, ncol=1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "doppler_per_satellite.png", dpi=150)
    plt.close(fig)

    # Plot 3: histogram of all corrected residuals pooled
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(corr_vals, bins=40, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Corrected Doppler residual [Hz]")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of corrected Doppler residuals - {args.label}\n"
                 f"RMS = {s_corr['rms']:.3f} Hz, n = {s_corr['n']}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "doppler_histogram.png", dpi=150)
    plt.close(fig)

    print(f"\nOutputs written to: {out_dir}")

if __name__ == "__main__":
    main()