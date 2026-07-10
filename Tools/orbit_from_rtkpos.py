#!/usr/bin/env python3
import argparse
import csv
import math
from datetime import datetime, timedelta
from pathlib import Path

MU = 3.98600442e14          # m^3/s^2, value used in Sentinel-6 POD context
OMEGA_E = 7.2921150e-5      # rad/s
GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)

def norm(v):
    return math.sqrt(sum(x*x for x in v))

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def cross(a, b):
    return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]

def sub(a, b):
    return [x-y for x, y in zip(a, b)]

def mul(s, v):
    return [s*x for x in v]

def angle_0_360(rad):
    deg = math.degrees(rad) % 360.0
    return deg

def jd_from_datetime(dt):
    # Gregorian calendar JD
    y = dt.year
    m = dt.month
    d = dt.day + (dt.hour + (dt.minute + dt.second / 60.0) / 60.0) / 24.0

    if m <= 2:
        y -= 1
        m += 12

    A = int(y / 100)
    B = 2 - A + int(A / 4)

    JD = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5
    return JD

def gmst_rad(dt_utc):
    JD = jd_from_datetime(dt_utc)
    T = (JD - 2451545.0) / 36525.0
    gmst_deg = (280.46061837 + 360.98564736629 * (JD - 2451545.0) + 0.000387933 * T*T - T*T*T / 38710000.0)
    return math.radians(gmst_deg % 360.0)

def rz(theta, v):
    c = math.cos(theta)
    s = math.sin(theta)
    x, y, z = v
    return [c*x - s*y, s*x + c*y, z]

def ecef_to_eci(r_ecef, v_ecef, dt_gps, gps_utc_offset=18.0):
    # GMST needs UTC/UT1 approximately. We use UTC = GPST - 18 s for Jan 2024.
    dt_utc = dt_gps - timedelta(seconds=gps_utc_offset)
    theta = gmst_rad(dt_utc)

    omega_cross_r = [-OMEGA_E * r_ecef[1], OMEGA_E * r_ecef[0], 0.0]

    v_aux = [v_ecef[0] + omega_cross_r[0], v_ecef[1] + omega_cross_r[1], v_ecef[2] + omega_cross_r[2]]

    r_eci = rz(theta, r_ecef)
    v_eci = rz(theta, v_aux)

    return r_eci, v_eci, theta

def orbital_elements(r, v):
    rnorm = norm(r)
    vnorm = norm(v)

    h = cross(r, v)
    hnorm = norm(h)

    k = [0.0, 0.0, 1.0]
    n = cross(k, h)
    nnorm = norm(n)

    e_vec = sub(mul(1.0 / MU, cross(v, h)), mul(1.0 / rnorm, r))
    ecc = norm(e_vec)

    energy = 0.5 * vnorm*vnorm - MU / rnorm
    a = -MU / (2.0 * energy)

    inc = math.acos(h[2] / hnorm)

    if nnorm > 1e-12:
        raan = math.atan2(n[1], n[0])
    else:
        raan = 0.0

    if nnorm > 1e-12 and ecc > 1e-10:
        argp = math.atan2(dot(cross(n, e_vec), h) / (nnorm * ecc * hnorm), dot(n, e_vec) / (nnorm * ecc))
    else:
        argp = 0.0

    if ecc > 1e-10:
        nu = math.atan2(dot(cross(e_vec, r), h) / (ecc * rnorm * hnorm), dot(e_vec, r) / (ecc * rnorm))
    else:
        nu = 0.0

    # Argument of latitude: useful for nearly circular orbits
    if nnorm > 1e-12:
        arglat = math.atan2(dot(cross(n, r), h) / (nnorm * rnorm * hnorm), dot(n, r) / (nnorm * rnorm))
    else:
        arglat = 0.0

    # Mean anomaly for elliptical orbit
    if ecc < 1.0:
        E = math.atan2(math.sqrt(max(0.0, 1.0 - ecc*ecc)) * math.sin(nu), ecc + math.cos(nu))
        M = E - ecc * math.sin(E)
    else:
        M = float("nan")

    return {"a_m": a, "a_km": a / 1000.0, "ecc": ecc, "inc_deg": angle_0_360(inc), "raan_deg": angle_0_360(raan), "argp_deg": angle_0_360(argp), "true_anomaly_deg": angle_0_360(nu), "mean_anomaly_deg": angle_0_360(M) if not math.isnan(M) else float("nan"), "argument_of_latitude_deg": angle_0_360(arglat), "r_norm_m": rnorm, "v_norm_m_s": vnorm}

def parse_rtkpos(path):
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

            dt = GPS_EPOCH + timedelta(weeks=week, seconds=sow)

            rows.append({"week": week, "sow": sow, "dt_gps": dt, "x": x, "y": y, "z": z, "q": q, "ns": ns, "rnorm": norm([x, y, z])})

    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pos_file")
    parser.add_argument("--out", default="orbit\\s6a_orbit_solution.txt")
    parser.add_argument("--csv", default="orbit\\s6a_ecef_trajectory.csv")
    parser.add_argument("--gps-utc", type=float, default=18.0)
    args = parser.parse_args()

    pos_file = Path(args.pos_file)
    out_file = Path(args.out)
    csv_file = Path(args.csv)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    rows = parse_rtkpos(pos_file)

    # Keep only single/valid solutions with enough satellites and plausible LEO radius
    clean = [r for r in rows if r["q"] in (1, 2, 5, 6) and r["ns"] >= 6 and 6.8e6 <= r["rnorm"] <= 8.5e6]

    if len(clean) < 3:
        raise RuntimeError("Not enough valid points to estimate velocity/orbit.")

    # Export trajectory with ECEF finite-difference velocity
    traj = []
    for i, row in enumerate(clean):
        if i == 0:
            prev_r = clean[i]
            next_r = clean[i+1]
        elif i == len(clean) - 1:
            prev_r = clean[i-1]
            next_r = clean[i]
        else:
            prev_r = clean[i-1]
            next_r = clean[i+1]

        dt = (next_r["dt_gps"] - prev_r["dt_gps"]).total_seconds()
        vx = (next_r["x"] - prev_r["x"]) / dt
        vy = (next_r["y"] - prev_r["y"]) / dt
        vz = (next_r["z"] - prev_r["z"]) / dt

        traj.append({**row, "vx_ecef": vx, "vy_ecef": vy, "vz_ecef": vz, "speed_ecef": norm([vx, vy, vz])})

    # Choose a central point to reduce finite-difference edge effects
    i0 = len(traj) // 2
    ref = traj[i0]

    r_ecef = [ref["x"], ref["y"], ref["z"]]
    v_ecef = [ref["vx_ecef"], ref["vy_ecef"], ref["vz_ecef"]]

    r_eci, v_eci, theta = ecef_to_eci(r_ecef, v_ecef, ref["dt_gps"], args.gps_utc)
    elems = orbital_elements(r_eci, v_eci)

    # Also compute simple stats
    rnorms = [r["rnorm"] for r in clean]
    ns_values = [r["ns"] for r in clean]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gps_week", "gps_sow", "gps_datetime", "x_ecef_m", "y_ecef_m", "z_ecef_m", "vx_ecef_mps", "vy_ecef_mps", "vz_ecef_mps", "q", "ns", "rnorm_m"])
        for r in traj:
            writer.writerow([r["week"], f"{r['sow']:.3f}", r["dt_gps"].isoformat(), f"{r['x']:.4f}", f"{r['y']:.4f}", f"{r['z']:.4f}", f"{r['vx_ecef']:.6f}", f"{r['vy_ecef']:.6f}", f"{r['vz_ecef']:.6f}", r["q"], r["ns"], f"{r['rnorm']:.4f}"])

    report = []
    report.append("SENTINEL-6A APPROXIMATE INITIAL ORBIT FROM RTKLIB SPP")
    report.append("=" * 70)
    report.append(f"Input POS file: {pos_file}")
    report.append(f"Total RTKLIB solution rows: {len(rows)}")
    report.append(f"Clean solution rows used: {len(clean)}")
    report.append("")
    report.append("Clean solution statistics:")
    report.append(f"  Radius min/max/mean [km]: {min(rnorms)/1000:.3f} / {max(rnorms)/1000:.3f} / {sum(rnorms)/len(rnorms)/1000:.3f}")
    report.append(f"  Satellites ns min/max: {min(ns_values)} / {max(ns_values)}")
    report.append("")
    report.append("Reference state selected:")
    report.append(f"  GPS epoch: {ref['dt_gps'].isoformat()}")
    report.append(f"  GPS week / SOW: {ref['week']} / {ref['sow']:.3f}")
    report.append(f"  ECEF position [m]: {r_ecef[0]:.4f}, {r_ecef[1]:.4f}, {r_ecef[2]:.4f}")
    report.append(f"  ECEF velocity [m/s]: {v_ecef[0]:.6f}, {v_ecef[1]:.6f}, {v_ecef[2]:.6f}")
    report.append(f"  ECI position [m]: {r_eci[0]:.4f}, {r_eci[1]:.4f}, {r_eci[2]:.4f}")
    report.append(f"  ECI velocity [m/s]: {v_eci[0]:.6f}, {v_eci[1]:.6f}, {v_eci[2]:.6f}")
    report.append("")
    report.append("Approximate osculating Keplerian elements:")
    report.append(f"  semi-major axis a [km]: {elems['a_km']:.6f}")
    report.append(f"  eccentricity e [-]: {elems['ecc']:.9f}")
    report.append(f"  inclination i [deg]: {elems['inc_deg']:.9f}")
    report.append(f"  RAAN Ω [deg]: {elems['raan_deg']:.9f}")
    report.append(f"  argument of perigee ω [deg]: {elems['argp_deg']:.9f}")
    report.append(f"  true anomaly ν [deg]: {elems['true_anomaly_deg']:.9f}")
    report.append(f"  mean anomaly M [deg]: {elems['mean_anomaly_deg']:.9f}")
    report.append(f"  argument of latitude u [deg]: {elems['argument_of_latitude_deg']:.9f}")
    report.append("")
    report.append("Skydel note:")
    report.append("  For near-circular orbits, argument of perigee and true anomaly can be unstable.")
    report.append("  If e is very small, argument of latitude u is often more meaningful than ω and ν separately.")
    report.append("  Prefer importing the ECEF trajectory CSV if Skydel allows trajectory import.")
    report.append("")
    report.append(f"Trajectory CSV written to: {csv_file}")

    out_file.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))

if __name__ == "__main__":
    main()