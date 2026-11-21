#!/usr/bin/env python3
"""Extract lateral control data from rlogs to CSV/Parquet.

Replaces the complex 3-file pipeline (lat.py -> lat_to_csv.py / lat_to_csv_torquennd.py)
with a single script that reads rlogs and emits one row per controlsState message.

Usage:
  python -m nnlc_tools.extract_lateral_data /path/to/rlogs/ -o output.csv
  python -m nnlc_tools.extract_lateral_data /path/to/rlogs/ -o output.parquet --format parquet
  python -m nnlc_tools.extract_lateral_data /path/to/rlogs/ -o output.csv --temporal
  python -m nnlc_tools.extract_lateral_data /path/to/rlogs/ -o output.csv --temporal \\
      --filter-overrides --override-margin 1.0 --stride 10   # 100+ hours of logs
"""

import argparse
import concurrent.futures
import functools
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# Temporal offsets matching nnlc.py's past_times and future_times
PAST_TIMES = [-0.3, -0.2, -0.1]
FUTURE_TIMES = [0.3, 0.6, 1.0, 1.5]

COLUMNS = [
    "timestamp",
    "v_ego",
    "a_ego",
    "steering_angle_deg",
    "steering_rate_deg",
    "steering_torque",
    "steering_pressed",
    "standstill",
    "desired_curvature",
    "curvature",
    "active",
    "lateral_control_type",
    "actual_lateral_accel",
    "desired_lateral_accel",
    "torque_output",
    "saturated",
    "roll",
    "lane_change_state",
    "torque_applied",  # carOutput torque: what the car got after the safety limits (falls back to torque_output)
    "lat_active",      # lateral control actually steering
    "segment",         # route--segment, keeps temporal columns from spilling across segments
]

# openpilot route--segment names: new style 00000016--6b14df1c14--29, old style 2024-01-15--12-30-45--5
SEGMENT_RE = re.compile(r"(?:[0-9a-f]{8}--[0-9a-f]{10}|\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})--\d+")


def segment_name(path):
    """route--segment from a device path (.../realdata/<route>--<seg>/rlog.zst) or a comma connect / flat file name
    (<dongle>_<route>--<seg>--rlog.zst). Falls back to the path itself."""
    found = SEGMENT_RE.findall(path.replace("\\", "/"))
    return found[-1] if found else path


def find_rlogs(input_dir):
    """Find all rlog files in the input directory."""
    patterns = ["**/rlog.zst", "**/rlog.bz2", "**/rlog",
                "**/*--rlog.zst", "**/*--rlog.bz2", "**/*--rlog"]  # flat names: <dongle>_<route>--<seg>--rlog.zst
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(input_dir, pattern), recursive=True))
    return sorted(f for f in set(files) if "qlog" not in os.path.basename(f))


def extract_segment(rlog_path):
    """Extract lateral data rows from a single rlog file.

    Follows the message iteration pattern from openpilot's
    measure_steering_accuracy.py — accumulate messages by type in a state dict,
    then emit a row when controlsState arrives.
    """
    from nnlc_tools.logreader import LogReader

    rows = []
    sm = {}
    segment = segment_name(rlog_path)

    try:
        lr = LogReader(rlog_path, sort_by_time=True)
    except Exception as e:
        print(f"  WARNING: Could not open {rlog_path}: {e}")
        return rows

    try:
        for msg in lr:
            try:
                msg_type = msg.which()
            except Exception:  # message type this schema doesn't know (fork-specific): not needed
                continue

            if msg_type == "carState":
                sm["carState"] = msg.carState
            elif msg_type == "controlsState":
                sm["controlsState"] = msg.controlsState
            elif msg_type == "selfdriveState":
                sm["selfdriveState"] = msg.selfdriveState
            elif msg_type == "liveParameters":
                sm["liveParameters"] = msg.liveParameters
            elif msg_type == "modelV2":
                sm["modelV2"] = msg.modelV2
            elif msg_type == "carOutput":
                sm["carOutput"] = msg.carOutput

            # Emit a row on each controlsState when we have carState too
            if msg_type == "controlsState" and "carState" in sm:
                cs = sm["carState"]
                ctrl = sm["controlsState"]

                timestamp = msg.logMonoTime / 1e9

                # Determine lateral control type and extract type-specific fields
                lat_state = ctrl.lateralControlState
                lat_type = lat_state.which()

                actual_lat_accel = float("nan")
                desired_lat_accel = float("nan")
                torque_output = float("nan")
                saturated = False
                lat_active = False

                if lat_type == "torqueState":
                    ts = lat_state.torqueState
                    actual_lat_accel = ts.actualLateralAccel
                    desired_lat_accel = ts.desiredLateralAccel
                    torque_output = ts.output
                    saturated = ts.saturated
                    lat_active = ts.active
                elif lat_type == "pidState":
                    ps = lat_state.pidState
                    torque_output = ps.output
                    saturated = ps.saturated
                    lat_active = ps.active

                if "carOutput" in sm:
                    torque_applied = sm["carOutput"].actuatorsOutput.torque
                else:  # older logs without carOutput
                    torque_applied = torque_output

                # active moved from controlsState to selfdriveState in newer openpilot
                if "selfdriveState" in sm:
                    active = sm["selfdriveState"].active
                else:
                    active = getattr(ctrl, "active", getattr(ctrl, "activeDEPRECATED", False))

                roll = sm["liveParameters"].roll if "liveParameters" in sm else float("nan")

                lane_change_state = 0
                if "modelV2" in sm:
                    try:
                        lane_change_state = int(sm["modelV2"].meta.laneChangeState)
                    except (AttributeError, ValueError, TypeError):
                        pass

                row = [
                    timestamp,
                    cs.vEgo,
                    cs.aEgo,
                    cs.steeringAngleDeg,
                    cs.steeringRateDeg,
                    cs.steeringTorque,
                    cs.steeringPressed,
                    cs.standstill,
                    ctrl.desiredCurvature,
                    ctrl.curvature,
                    active,
                    lat_type,
                    actual_lat_accel,
                    desired_lat_accel,
                    torque_output,
                    saturated,
                    roll,
                    lane_change_state,
                    torque_applied,
                    lat_active,
                    segment,
                ]
                rows.append(row)
    except Exception as e:
        print(f"  WARNING: Error processing {rlog_path}: {e}")

    return rows


def add_temporal_columns(df):
    """Add lagged/lead columns for temporal model training.

    Adds columns at offsets matching nnlc.py's past_times [-0.3, -0.2, -0.1]
    and future_times [0.3, 0.6, 1.0, 1.5].
    """
    dt = 0.01  # controlsState runs at 100Hz
    temporal_cols = ["actual_lateral_accel", "desired_lateral_accel", "roll"]

    for offset in PAST_TIMES + FUTURE_TIMES:
        shift_frames = int(round(offset / dt))
        suffix = f"_t{offset:+.1f}".replace(".", "").replace("+", "p").replace("-", "m")

        for col in temporal_cols:
            if col in df.columns:
                if "segment" in df.columns:  # never pull values from the next segment / route
                    df[f"{col}{suffix}"] = df.groupby("segment", sort=False)[col].shift(-shift_frames)
                else:
                    df[f"{col}{suffix}"] = df[col].shift(-shift_frames)

    return df


def process_file(rlog_path, temporal=False, filter_overrides=False, override_margin=0.0, stride=1):
    """Extract one segment into a DataFrame (temporal columns, override filter and stride applied at 100 Hz)."""
    rows = extract_segment(rlog_path)
    if not rows:
        return None, 0
    df = pd.DataFrame(rows, columns=COLUMNS)
    # Lateral jerk the way the runtime measures it: from the wheel's motion (openpilot feeds the NN the steering-rate
    # jerk). Centred difference over 0.1 s of the steering-angle lateral accel, at 100 Hz and inside this segment.
    a = df["actual_lateral_accel"]
    df["actual_lateral_jerk"] = (a.shift(-5) - a.shift(5)) / 0.1
    if temporal:
        df = add_temporal_columns(df)
    dropped = 0
    if filter_overrides and "steering_pressed" in df.columns:
        pressed = df["steering_pressed"].astype(bool).to_numpy().astype(np.int32)
        w = int(round(override_margin / 0.01))
        if w > 0:  # also drop the frames around the override (e.g. the Toyota EPS backs off around hand torque)
            # "full" + centre slice keeps the segment's length even when it is shorter than the window
            pressed = np.convolve(pressed, np.ones(2 * w + 1, dtype=np.int32), mode="full")[w:w + len(pressed)]
        near = pressed > 0
        dropped = int(near.sum())
        df = df[~near]
    if stride > 1:
        df = df.iloc[::stride]
    return df, dropped


def main():
    parser = argparse.ArgumentParser(
        description="Extract lateral control data from rlogs to CSV/Parquet.",
    )
    parser.add_argument("input", help="Directory containing rlog files")
    parser.add_argument("-o", "--output", default="lateral_data.csv",
                        help="Output file path (default: lateral_data.csv)")
    parser.add_argument("--format", choices=["csv", "parquet"], default=None,
                        help="Output format (default: inferred from extension)")
    parser.add_argument("--temporal", action="store_true",
                        help="Add temporal lag/lead columns for NNLC training")
    parser.add_argument("--filter-overrides", action="store_true",
                        help="Drop rows where driver overrides (steering_pressed=True)")
    parser.add_argument("--override-margin", type=float, default=0.0,
                        help="With --filter-overrides, also drop rows within this many seconds of an override")
    parser.add_argument("--stride", type=int, default=1,
                        help="Keep every Nth row (controlsState is 100 Hz), after the temporal columns and the "
                             "override filter. Use 5-10 for large data sets (100+ hours). nnlc-interventions needs "
                             "100 Hz rows, so with --stride use --filter-overrides here instead.")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1,
                        help="Parallel worker processes (default: all CPU cores)")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"ERROR: Input directory not found: {args.input}")
        sys.exit(1)

    rlog_files = find_rlogs(args.input)
    if not rlog_files:
        print(f"ERROR: No rlog files found in {args.input}")
        sys.exit(1)

    print(f"Found {len(rlog_files)} rlog files")
    seen, unique = set(), []
    for f in rlog_files:  # the same segment in two places (device copy + comma connect download) counts once
        seg = segment_name(f)
        if seg not in seen:
            seen.add(seg)
            unique.append(f)
    if len(unique) < len(rlog_files):
        print(f"Skipping {len(rlog_files) - len(unique)} duplicate segments")
    rlog_files = unique

    # Determine output format
    fmt = args.format
    if fmt is None:
        if args.output.endswith(".parquet"):
            fmt = "parquet"
        else:
            fmt = "csv"

    # One segment at a time: temporal columns never cross segments, and CSV output is streamed so memory stays flat
    # no matter how many hours of logs there are (parquet output is still collected in memory).
    work = functools.partial(process_file, temporal=args.temporal, filter_overrides=args.filter_overrides,
                             override_margin=args.override_margin, stride=args.stride)
    total, dropped_overrides, first = 0, 0, True
    parts = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for df, dropped in tqdm(pool.map(work, rlog_files, chunksize=4), total=len(rlog_files),
                                desc="Processing rlogs"):
            if df is None:
                continue
            dropped_overrides += dropped
            total += len(df)
            if fmt == "parquet":
                parts.append(df)
            else:
                df.to_csv(args.output, mode="w" if first else "a", header=first, index=False)
            first = False

    if total == 0:
        print("ERROR: No data extracted from any rlog files")
        sys.exit(1)
    if fmt == "parquet":
        pd.concat(parts, ignore_index=True).to_parquet(args.output, index=False)
    if args.filter_overrides:
        print(f"Filtered {dropped_overrides} override rows")
    print(f"Saved {total} rows to {args.output} ({fmt})")


if __name__ == "__main__":
    main()
