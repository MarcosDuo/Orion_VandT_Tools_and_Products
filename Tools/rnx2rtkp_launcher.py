#!/usr/bin/env python3
r"""
rnx2rtkp_chunked.py

Launcher to run RTKLIB's rnx2rtkp over a long observation file in time
chunks, then merge the partial .pos outputs into a single solution file.

Why: rnx2rtkp loads the whole observation file into memory. With very
long files (e.g. 52 h at 1 Hz, multi-GNSS) the Windows build may fail to
allocate memory and exit immediately. Processing in chunks with -ts/-te
avoids this, and for SPP (-p 0) it is rigorous: the solution is computed
epoch by epoch with no filter state, so chunked results are identical to
a single full run.

Chunk boundaries are made INCLUSIVE on both ends (consecutive chunks
share the boundary second) and duplicated epochs are removed at merge
time. This protects receivers whose epochs fall on fractional seconds
(e.g. u-blox at hh:mm:ss.996).

Usage example (Windows PowerShell / cmd):
    python rnx2rtkp_chunked.py ^
        --exe .\rnx2rtkp.exe ^
        --obs .\ORION_LONG_20260624_065633.obs ^
        --nav .\BRDC00IGS_R_20261750000_01D_MN.rnx .\BRDC00IGS_R_20261760000_01D_MN.rnx .\BRDC00IGS_R_20261770000_01D_MN.rnx ^
        --start "2026/06/24 06:58:42" --end "2026/06/26 11:29:02" ^
        --chunk-hours 6 ^
        --rtk-args "-p 0 -sys G,E,C -m 10" ^
        --out orion_spp_full.pos

Partial files are kept in a subfolder (chunks_<outname>/) so you can
inspect or re-run a failed chunk without repeating the rest.
"""

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime, timedelta

TIME_FMT = "%Y/%m/%d %H:%M:%S"


def parse_time(s):
    return datetime.strptime(s.strip(), TIME_FMT)


def build_chunks(t_start, t_end, chunk_hours):
    """Build inclusive [ts, te] windows covering [t_start, t_end].

    Consecutive chunks share their boundary second on purpose; duplicate
    epochs are removed when merging.
    """
    chunks = []
    t = t_start
    step = timedelta(hours=chunk_hours)
    while t < t_end:
        te = min(t + step, t_end)
        chunks.append((t, te))
        t = te
    return chunks


def run_chunk(exe, obs, navs, ts, te, rtk_args, out_path):
    cmd = [exe] + shlex.split(rtk_args)
    cmd += ["-ts", ts.strftime("%Y/%m/%d"), ts.strftime("%H:%M:%S")]
    cmd += ["-te", te.strftime("%Y/%m/%d"), te.strftime("%H:%M:%S")]
    cmd += [obs] + navs + ["-o", out_path]
    print(f"  -> {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"     WARNING: rnx2rtkp returned {result.returncode}")
        if result.stderr:
            print("     stderr:", result.stderr.strip()[:500])
    return os.path.isfile(out_path) and os.path.getsize(out_path) > 0


def merge_pos(chunk_files, out_file):
    """Concatenate .pos chunk files, keeping one header and removing
    duplicated epochs at chunk boundaries.

    The epoch key is the first two whitespace-separated fields as raw
    strings, so it works with any rnx2rtkp time format (week/tow,
    yyyy/mm/dd hh:mm:ss, hms...) without parsing numbers."""
    n_data = 0
    n_dup = 0
    seen = set()
    header_written = False
    with open(out_file, "w") as fo:
        for path in chunk_files:
            n_chunk = 0
            with open(path) as fi:
                for line in fi:
                    if line.startswith("%"):
                        if not header_written:
                            fo.write(line)
                        continue
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    key = (parts[0], parts[1])
                    if key in seen:
                        n_dup += 1
                        continue
                    seen.add(key)
                    fo.write(line)
                    n_data += 1
                    n_chunk += 1
            print(f"    {os.path.basename(path)}: {n_chunk} epochs added")
            header_written = True
    return n_data, n_dup


def main():
    ap = argparse.ArgumentParser(description="Chunked rnx2rtkp launcher")
    ap.add_argument("--exe", default="rnx2rtkp", help="path to rnx2rtkp executable")
    ap.add_argument("--obs", required=True, help="observation RINEX file")
    ap.add_argument("--nav", nargs="+", required=True, help="navigation file(s)")
    ap.add_argument("--start", required=True, help='start time "YYYY/MM/DD hh:mm:ss"')
    ap.add_argument("--end", required=True, help='end time "YYYY/MM/DD hh:mm:ss"')
    ap.add_argument("--chunk-hours", type=float, default=6.0, help="chunk length in hours (default 6)")
    ap.add_argument("--rtk-args", default="-p 0 -sys G,E,C -m 10",
                    help='processing options passed to rnx2rtkp (default "-p 0 -sys G,E,C -m 10")')
    ap.add_argument("--out", required=True, help="merged output .pos file")
    ap.add_argument("--keep-chunks", action="store_true",
                    help="keep partial chunk files (default: keep them anyway in subfolder)")
    args = ap.parse_args()

    t_start = parse_time(args.start)
    t_end = parse_time(args.end)
    if t_end <= t_start:
        sys.exit("ERROR: end time must be after start time")

    chunks = build_chunks(t_start, t_end, args.chunk_hours)
    base = os.path.splitext(os.path.basename(args.out))[0]
    chunk_dir = f"chunks_{base}"
    os.makedirs(chunk_dir, exist_ok=True)

    print(f"Processing {len(chunks)} chunks of {args.chunk_hours} h "
          f"({t_start} -> {t_end})\n")

    chunk_files = []
    for i, (ts, te) in enumerate(chunks, 1):
        out_path = os.path.join(chunk_dir, f"{base}_chunk{i:02d}.pos")
        print(f"[{i}/{len(chunks)}] {ts} -> {te}")
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            print("     already exists, skipping (delete it to re-run)")
            chunk_files.append(out_path)
            continue
        ok = run_chunk(args.exe, args.obs, args.nav, ts, te, args.rtk_args, out_path)
        if ok:
            chunk_files.append(out_path)
        else:
            print(f"     ERROR: chunk {i} produced no output — continuing, "
                  f"re-run later after checking")

    if not chunk_files:
        sys.exit("ERROR: no chunk produced any output")

    print("\nMerging...")
    n_data, n_dup = merge_pos(chunk_files, args.out)
    print(f"Merged {len(chunk_files)} chunks -> {args.out}")
    print(f"  {n_data} epochs written, {n_dup} boundary duplicates removed")


if __name__ == "__main__":
    main()