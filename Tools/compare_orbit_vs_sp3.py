#!/usr/bin/env python3
"""
compare_orbit_vs_sp3.py 

Two separate, honest comparisons for the orbital GNSS simulation scenarios:

  Comparison A - RECEIVER PERFORMANCE
      RTKLIB SPP solution (.pos)  vs  Skydel ground-truth receiver trajectory (.csv)
      -> isolates how well the receiver + SPP processing recovers the trajectory
         that Skydel actually simulated. Expected: few metres.

  Comparison B - SIMULATION FIDELITY
      Skydel ground-truth trajectory (.csv)  vs  real precise orbit (.sp3)
      -> isolates how far the TLE/Keplerian-based scenario used to drive Skydel
         is from the true Sentinel-6A orbit. Expected: can be kilometres, this is
         a property of the input orbital elements, not of the receiver.

Both residuals are expressed in the RTN (Radial / Transverse / Normal) frame.

Usage:
  python compare_orbit_vs_sp3.py \
      --pos orion_gps_gal_spp.pos \
      --truth receiver_antenna.csv \
      --sp3 S6ACPOD22963.sp3 \
      --out-dir results/orion_gps_gal \
      --label "Orion GPS+GAL"

You can also run only one comparison by omitting --sp3 (A only) or --pos (B only).

Outputs (written to --out-dir):
  comparisonA_residuals.csv, comparisonA_report.txt, comparisonA_timeseries.png
  comparisonB_residuals.csv, comparisonB_report.txt, comparisonB_timeseries.png
  summary_rms_comparison.png   (only if both A and B were computed)
"""

import argparse
import csv
import math
from datetime import datetime, timedelta
from pathlib import Path

GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)

# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def norm(v):
    return math.sqrt(sum(x * x for x in v))

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

def sub(a, b):
    return [x - y for x, y in zip(a, b)]

def unit(v):
    n = norm(v)
    if n == 0:
        raise ValueError("Cannot normalize a zero-length vector")
    return [x / n for x in v]

def rtn_unit_vectors(r_ref, v_ref):
    r_hat = unit(r_ref)
    h = cross(r_ref, v_ref)
    n_hat = unit(h)
    t_hat = unit(cross(n_hat, r_hat))
    return r_hat, t_hat, n_hat

def project_rtn(delta, r_ref, v_ref):
    r_hat, t_hat, n_hat = rtn_unit_vectors(r_ref, v_ref)
    return dot(delta, r_hat), dot(delta, t_hat), dot(delta, n_hat)

# ---------------------------------------------------------------------------
# RTKLIB .pos parsing
# ---------------------------------------------------------------------------

def parse_rtkpos(path, min_ns=6, r_min=6.8e6, r_max=8.5e6, valid_q=(1, 2, 5, 6)):
    rows = []
    with open(path, "r", encoding="latin1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                week = int(parts[0])
                sow = float(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                q = int(parts[5])
                ns = int(parts[6])
            except ValueError:
                continue

            rnorm = norm([x, y, z])
            if q not in valid_q or ns < min_ns or not (r_min <= rnorm <= r_max):
                continue

            dt = GPS_EPOCH + timedelta(weeks=week, seconds=sow)
            rows.append({"dt": dt, "x": x, "y": y, "z": z, "q": q, "ns": ns})
    rows.sort(key=lambda r: r["dt"])
    return rows

# ---------------------------------------------------------------------------
# Skydel ground-truth receiver_antenna.csv parsing
# ---------------------------------------------------------------------------

def parse_truth_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            week = int(float(row["GPS Week Number"]))
            tow = float(row["GPS TOW"])
            dt = GPS_EPOCH + timedelta(weeks=week, seconds=tow)
            rows.append({"dt": dt, "x": float(row["ECEF X (m)"]), "y": float(row["ECEF Y (m)"]), "z": float(row["ECEF Z (m)"]), "vx": float(row["Velocity X (m/s)"]), "vy": float(row["Velocity Y (m/s)"]), "vz": float(row["Velocity Z (m/s)"])})
    rows.sort(key=lambda r: r["dt"])
    return rows

def interp_linear_truth(rows, target_dt):
    """Simple linear interpolation between the two bracketing truth samples.
    Truth is logged at 10 Hz, dense enough that linear interpolation is
    accurate to well under a millimetre for this purpose."""
    lo_idx = None
    for i in range(len(rows) - 1):
        if rows[i]["dt"] <= target_dt <= rows[i + 1]["dt"]:
            lo_idx = i
            break
    if lo_idx is None:
        return None
    lo, hi = rows[lo_idx], rows[lo_idx + 1]
    span = (hi["dt"] - lo["dt"]).total_seconds()
    f = (target_dt - lo["dt"]).total_seconds() / span if span > 0 else 0.0
    x = lo["x"] + f * (hi["x"] - lo["x"])
    y = lo["y"] + f * (hi["y"] - lo["y"])
    z = lo["z"] + f * (hi["z"] - lo["z"])
    vx = lo["vx"] + f * (hi["vx"] - lo["vx"])
    vy = lo["vy"] + f * (hi["vy"] - lo["vy"])
    vz = lo["vz"] + f * (hi["vz"] - lo["vz"])
    return [x, y, z], [vx, vy, vz]

# ---------------------------------------------------------------------------
# SP3 parsing (subset of SP3-c/d sufficient for position records)
# ---------------------------------------------------------------------------

def parse_sp3(path, sat_id=None, leap_seconds=18.0):
    time_system = "GPS"
    sats_data = {}
    current_epoch = None

    with open(path, "r", encoding="latin1") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("%c"):
                tokens = line.split()
                if len(tokens) > 4 and tokens[3] in ("GPS", "UTC", "TAI"):
                    time_system = tokens[3]
                continue
            if line.startswith("*"):
                parts = line[1:].split()
                y, mo, d, h, mi = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
                sec = float(parts[5])
                whole_sec = int(sec)
                micro = int(round((sec - whole_sec) * 1e6))
                current_epoch = datetime(y, mo, d, h, mi, whole_sec, micro)
                if time_system == "UTC":
                    current_epoch = current_epoch + timedelta(seconds=leap_seconds)
                continue
            if line.startswith("P"):
                sid = line[1:4].strip()
                if sat_id is not None and sid != sat_id:
                    continue
                try:
                    x_km = float(line[4:18])
                    y_km = float(line[18:32])
                    z_km = float(line[32:46])
                except ValueError:
                    continue
                if current_epoch is None:
                    continue
                sats_data.setdefault(sid, []).append({
                    "dt": current_epoch,
                    "x": x_km * 1000.0, "y": y_km * 1000.0, "z": z_km * 1000.0,
                })
                continue

    for sid in sats_data:
        sats_data[sid].sort(key=lambda r: r["dt"])
    return sats_data, time_system

def pick_satellite(sats_data, sat_id):
    if sat_id is not None:
        if sat_id not in sats_data:
            raise ValueError(f"Satellite ID '{sat_id}' not found in SP3. "
                              f"Available IDs: {sorted(sats_data.keys())}")
        return sats_data[sat_id]
    if len(sats_data) == 1:
        return next(iter(sats_data.values()))
    raise ValueError(f"SP3 file contains multiple satellite IDs {sorted(sats_data.keys())}. "
        f"Re-run with --sat <ID> to pick the right one.")

def lagrange_interpolate(sp3_rows, target_dt, order=9):
    n_pts = order + 1
    times = [r["dt"] for r in sp3_rows]
    idx = 0
    while idx < len(times) and times[idx] < target_dt:
        idx += 1
    half = n_pts // 2
    start = max(0, idx - half)
    end = min(len(sp3_rows), start + n_pts)
    start = max(0, end - n_pts)
    window = sp3_rows[start:end]
    if len(window) < 4:
        raise ValueError("Not enough SP3 samples around target epoch for interpolation.")

    t0 = window[0]["dt"]
    t = [(r["dt"] - t0).total_seconds() for r in window]
    tt = (target_dt - t0).total_seconds()

    def lagrange_at(values, tt, t):
        total = 0.0
        m = len(t)
        for i in range(m):
            term = values[i]
            for j in range(m):
                if j == i:
                    continue
                term *= (tt - t[j]) / (t[i] - t[j])
            total += term
        return total

    xs = [r["x"] for r in window]
    ys = [r["y"] for r in window]
    zs = [r["z"] for r in window]
    x = lagrange_at(xs, tt, t)
    y = lagrange_at(ys, tt, t)
    z = lagrange_at(zs, tt, t)

    eps = 0.5
    xm, ym, zm = (lagrange_at(xs, tt - eps, t), lagrange_at(ys, tt - eps, t), lagrange_at(zs, tt - eps, t))
    xp, yp, zp = (lagrange_at(xs, tt + eps, t), lagrange_at(ys, tt + eps, t), lagrange_at(zs, tt + eps, t))
    v = [(xp - xm) / (2 * eps), (yp - ym) / (2 * eps), (zp - zm) / (2 * eps)]
    return [x, y, z], v

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def stats(values):
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "rms": float("nan")}
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)
    rms = math.sqrt(sum(v * v for v in values) / n)
    return {"n": n, "mean": mean, "std": std, "rms": rms}

def write_residuals_csv(path, residuals):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["datetime_gps", "dR_m", "dT_m", "dN_m", "d3_m"])
        for r in residuals:
            writer.writerow([r["dt"].isoformat(), f"{r['dR']:.5f}", f"{r['dT']:.5f}", f"{r['dN']:.5f}", f"{r['d3']:.5f}"])

def write_report(path, title, residuals, extra_lines=None):
    r_vals = [r["dR"] for r in residuals]
    t_vals = [r["dT"] for r in residuals]
    n_vals = [r["dN"] for r in residuals]
    d3_vals = [r["d3"] for r in residuals]
    sr, st, sn, s3 = stats(r_vals), stats(t_vals), stats(n_vals), stats(d3_vals)

    lines = [title, "=" * len(title), ""]
    if extra_lines:
        lines += extra_lines + [""]
    lines.append(f"Epochs compared: {len(residuals)}")
    lines.append("")
    lines.append("Residual statistics [m]  (mean / std / RMS):")
    lines.append(f"  Radial      (R): {sr['mean']:+.4f} / {sr['std']:.4f} / {sr['rms']:.4f}")
    lines.append(f"  Along-track (T): {st['mean']:+.4f} / {st['std']:.4f} / {st['rms']:.4f}")
    lines.append(f"  Cross-track (N): {sn['mean']:+.4f} / {sn['std']:.4f} / {sn['rms']:.4f}")
    lines.append(f"  3D norm        : {s3['mean']:+.4f} / {s3['std']:.4f} / {s3['rms']:.4f}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print()
    return {"R": sr, "T": st, "N": sn, "d3": s3}

# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

def comparison_A_receiver_vs_truth(pos_rows, truth_rows):
    """SPP vs Skydel ground-truth trajectory. RTN frame built from truth."""
    residuals = []
    for row in pos_rows:
        interp = interp_linear_truth(truth_rows, row["dt"])
        if interp is None:
            continue
        r_ref, v_ref = interp
        delta = sub([row["x"], row["y"], row["z"]], r_ref)
        dR, dT, dN = project_rtn(delta, r_ref, v_ref)
        residuals.append({"dt": row["dt"], "dR": dR, "dT": dT, "dN": dN, "d3": norm(delta)})
    return residuals


def comparison_B_truth_vs_sp3(truth_rows, sp3_rows, order=9, step=1):
    """Skydel ground-truth trajectory vs real precise SP3 orbit. RTN frame from SP3."""
    residuals = []
    margin = order // 2 + 2
    t_lo = sp3_rows[margin]["dt"]
    t_hi = sp3_rows[-margin - 1]["dt"]
    usable = [r for r in truth_rows if t_lo <= r["dt"] <= t_hi]
    for row in usable[::step]:
        try:
            r_ref, v_ref = lagrange_interpolate(sp3_rows, row["dt"], order=order)
        except ValueError:
            continue
        delta = sub([row["x"], row["y"], row["z"]], r_ref)
        dR, dT, dN = project_rtn(delta, r_ref, v_ref)
        residuals.append({"dt": row["dt"], "dR": dR, "dT": dT, "dN": dN, "d3": norm(delta)})
    return residuals

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_timeseries(residuals, out_path, title, unit_label="m", unit_scale=1.0, show_d3=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = residuals[0]["dt"]
    t = [(r["dt"] - t0).total_seconds() for r in residuals]
    dR = [r["dR"] * unit_scale for r in residuals]
    dT = [r["dT"] * unit_scale for r in residuals]
    dN = [r["dN"] * unit_scale for r in residuals]
    d3 = [r["d3"] * unit_scale for r in residuals]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, dR, label="Radial (R)", linewidth=1.2)
    ax.plot(t, dT, label="Along-track (T)", linewidth=1.2)
    ax.plot(t, dN, label="Cross-track (N)", linewidth=1.2)
    if show_d3:
        ax.plot(t, d3, label="3D norm (always \u2265 0)", linewidth=1.6, color="black", linestyle="--")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel(f"Residual [{unit_label}]")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_timeseries_abs(residuals, out_path, title, unit_label="m", unit_scale=1.0):
    """ABSOLUTE VALUE variant: |R|, |T|, |N| vs time. Visually cleaner (no
    sign flips), but it discards the sign of each component, so it cannot
    show systematic bias direction - only magnitude. Kept as a clearly
    labelled extra, not a replacement for the signed plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    t0 = residuals[0]["dt"]
    t = [(r["dt"] - t0).total_seconds() for r in residuals]
    dR = [abs(r["dR"]) * unit_scale for r in residuals]
    dT = [abs(r["dT"]) * unit_scale for r in residuals]
    dN = [abs(r["dN"]) * unit_scale for r in residuals]
    d3 = [r["d3"] * unit_scale for r in residuals]
 
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, dR, label="|Radial (R)|", linewidth=1.0, alpha=0.8)
    ax.plot(t, dT, label="|Along-track (T)|", linewidth=1.0, alpha=0.8)
    ax.plot(t, dN, label="|Cross-track (N)|", linewidth=1.0, alpha=0.8)
    ax.plot(t, d3, label="3D norm", linewidth=1.6, color="black", linestyle="--")
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel(f"|Residual| [{unit_label}]")
    ax.set_title(title + "\n(absolute value - sign/bias direction not shown)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_rms_summary(statsA, statsB, out_path, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes_names = ["R", "T", "N", "d3"]
    display_names = ["Radial", "Along-track", "Cross-track", "3D norm"]
    a_vals = [statsA[k]["rms"] for k in axes_names]
    b_vals = [statsB[k]["rms"] for k in axes_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(axes_names))
    width = 0.35
    ax.bar([i - width / 2 for i in x], a_vals, width, label="A: Receiver vs Skydel truth")
    ax.bar([i + width / 2 for i in x], b_vals, width, label="B: Skydel truth vs real SP3")
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(display_names)
    ax.set_ylabel("RMS residual [m] (log scale)")
    ax.set_title(f"Receiver error vs simulation-fidelity error — {label}")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pos", default=None, help="RTKLIB SPP .pos file (ECEF). Needed for Comparison A.")
    parser.add_argument("--truth", default=None, help="Skydel ground-truth receiver_antenna.csv. Needed for Comparisons A and B.")
    parser.add_argument("--sp3", default=None, help="Reference precise orbit .sp3 file. Needed for Comparison B.")
    parser.add_argument("--sat", default=None, help="Satellite ID to extract from the SP3 if it has several")
    parser.add_argument("--leap-seconds", type=float, default=18.0)
    parser.add_argument("--order", type=int, default=9, help="Lagrange interpolation order for the SP3")
    parser.add_argument("--min-ns", type=int, default=6)
    parser.add_argument("--truth-step", type=int, default=10, help="Subsample truth rows by this factor for Comparison B (10 Hz truth -> step=10 gives ~1 Hz)")
    parser.add_argument("--label", default="scenario", help="Short label used in plot titles")
    parser.add_argument("--out-dir", required=True, help="Directory to write CSVs, reports and plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth_rows = parse_truth_csv(args.truth) if args.truth else None

    statsA = statsB = None

    # --- Comparison A: receiver vs truth ---
    if args.pos and truth_rows:
        pos_rows = parse_rtkpos(args.pos, min_ns=args.min_ns)
        if len(pos_rows) == 0:
            raise RuntimeError("No valid epochs found in the .pos file after quality filtering.")
        resA = comparison_A_receiver_vs_truth(pos_rows, truth_rows)
        if len(resA) == 0:
            raise RuntimeError("No overlapping epochs between .pos and truth CSV.")
        write_residuals_csv(out_dir / "comparisonA_residuals.csv", resA)
        statsA = write_report(out_dir / "comparisonA_report.txt", f"COMPARISON A - RECEIVER PERFORMANCE ({args.label})\n"
            f"SPP ({Path(args.pos).name}) vs Skydel ground truth ({Path(args.truth).name})", resA)
        plot_timeseries(resA, out_dir / "comparisonA_timeseries.png", f"Comparison A - Receiver vs Skydel truth ({args.label})", unit_label="m", unit_scale=1.0)
        plot_timeseries_abs(resA, out_dir / "comparisonA_timeseries_abs.png", f"Comparison A - Receiver vs Skydel truth ({args.label})", unit_label="m", unit_scale=1.0)

    # --- Comparison B: truth vs SP3 ---
    if truth_rows and args.sp3:
        sats_data, time_system = parse_sp3(args.sp3, sat_id=args.sat, leap_seconds=args.leap_seconds)
        sp3_rows = pick_satellite(sats_data, args.sat)
        if len(sp3_rows) < args.order + 1:
            raise RuntimeError("SP3 file too short for the requested interpolation order.")
        resB = comparison_B_truth_vs_sp3(truth_rows, sp3_rows, order=args.order, step=args.truth_step)
        if len(resB) == 0:
            raise RuntimeError("No overlapping epochs between truth CSV and SP3.")
        write_residuals_csv(out_dir / "comparisonB_residuals.csv", resB)
        statsB = write_report(out_dir / "comparisonB_report.txt", f"COMPARISON B - SIMULATION FIDELITY ({args.label})\n"
            f"Skydel ground truth ({Path(args.truth).name}) vs real SP3 ({Path(args.sp3).name}, "
            f"time system: {time_system}, sat: {args.sat or list(sats_data.keys())[0]})", resB)
        plot_timeseries(resB, out_dir / "comparisonB_timeseries.png", f"Comparison B - Skydel truth vs real SP3 ({args.label})", unit_label="km", unit_scale=1e-3)
        plot_timeseries_abs(resB, out_dir / "comparisonB_timeseries_abs.png", f"Comparison B - Skydel truth vs real SP3 ({args.label})", unit_label="km", unit_scale=1e-3)

    # --- Summary plot if both were computed ---
    if statsA and statsB:
        plot_rms_summary(statsA, statsB, out_dir / "summary_rms_comparison.png", args.label)
        print(f"Summary plot written to: {out_dir / 'summary_rms_comparison.png'}")

    if not statsA and not statsB:
        raise RuntimeError("Nothing to compute: provide --pos + --truth for Comparison A, "
                            "and/or --truth + --sp3 for Comparison B.")

if __name__ == "__main__":
    main()