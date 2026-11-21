#!/usr/bin/env python3
"""Training target from the torque that actually turned the wheel, for openpilot-steered and hand-driven frames.

Training on the steering command alone (openpilot-steered frames only) teaches the model the controller that was
running when the data was recorded, including its friction setting. twilsonco's first NNLC models were trained on
hand-driven data, where the driver supplies the friction torque, using the measured EPS torque scaled to command
units. This step does the same and adds the driver's own torque:

    torque_target = k * (steering_torque_eps + c * steering_torque)

    k  EPS units -> command units, from openpilot-steered frames with hands off (EPS torque follows the command there)
    c  driver torque -> EPS units, from hand-driven steady curves: the torque they need matches what openpilot needed in
          the same speed / lateral accel bin. c = 0 is twilsonco's EPS-only target.

Frames kept:
    steered      lateral control active (includes lateral-only modes), not standstill
    hand-driven  lateral control off, no torque command, >= 5 m/s, |a_ego| < 2.5

Usage:
    nnlc-torque-target pruned.csv -o TOYOTA_PRIUS_V.csv
    nnlc-torque-target pruned.csv -o TOYOTA_PRIUS_V.csv --driver-scale 0    # EPS torque only (twilsonco)
    nnlc-torque-target pruned.csv -o TOYOTA_PRIUS_V.csv --steered-only      # same target, no hand-driven frames
"""

import argparse
import sys

import numpy as np
import pandas as pd

STEERED, HAND = 1, 0
G = 9.81


def to_bool(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def select_rows(df):
    standstill = to_bool(df["standstill"])
    lat = to_bool(df["lat_active"])
    finite = np.isfinite(df["actual_lateral_accel"]) & np.isfinite(df["steering_torque_eps"]) & np.isfinite(df["steering_torque"])
    steered = lat & ~standstill & finite
    hand = (~lat & ~standstill & finite & (df["torque_applied"].abs() < 1e-6) & (df["v_ego"] >= 5.0)
            & (df["a_ego"].abs() < 2.5))
    return steered.to_numpy(), hand.to_numpy()


def fit_eps_scale(df, steered):
    """Command units per EPS unit, from steered, hands-off, slow-changing frames (least squares through the origin)."""
    drv = df["steering_torque"].abs().to_numpy()
    jerk = df["actual_lateral_jerk"].abs().to_numpy() if "actual_lateral_jerk" in df else np.zeros(len(df))
    m = steered & (drv <= np.nanpercentile(drv[steered], 50)) & (jerk < 0.5) & (df["torque_applied"].abs().to_numpy() > 0.02)
    eps, cmd = df["steering_torque_eps"].to_numpy()[m], df["torque_applied"].to_numpy()[m]
    k = float(np.sum(cmd * eps) / np.sum(eps * eps))
    r = float(np.corrcoef(cmd, eps)[0, 1])
    return k, r, int(m.sum())


def steady_bins(df, rows):
    x = (df["actual_lateral_accel"] - G * df["roll"]).to_numpy()
    v = df["v_ego"].to_numpy()
    jerk = df["actual_lateral_jerk"].abs().to_numpy() if "actual_lateral_jerk" in df else np.zeros(len(df))
    m = rows & (jerk < 0.3) & (np.abs(x) >= 0.5) & (np.abs(x) <= 2.5) & (v >= 8) & (v <= 35)
    xs, vs = np.nan_to_num(x), np.nan_to_num(v)  # NaN-free for the integer bin key (same key for every row)
    key = np.floor(vs / 4.0).astype(int) * 1000 + np.floor(xs / 0.25).astype(int)
    return m, key


def fit_driver_scale(df, steered, hand, k):
    """Driver torque -> EPS units, so hand-driven steady curves need the torque openpilot needed in the same bin."""
    eps, drv = df["steering_torque_eps"].to_numpy(), df["steering_torque"].to_numpy()
    ms, key = steady_bins(df, steered & (np.abs(drv) <= np.nanpercentile(np.abs(drv[steered]), 50)))
    mh, _ = steady_bins(df, hand)
    need = pd.Series(k * eps[ms]).groupby(key[ms]).agg(["median", "size"])
    need = need[need["size"] >= 30]["median"]
    sel = mh & np.isin(key, need.index.to_numpy())
    n = int(sel.sum())
    if n < 500:
        return 0.0, {"n": n, "reason": f"only {n} hand-driven steady-curve frames (need 500)"}
    needed = need.reindex(key[sel]).to_numpy()
    r = needed - k * eps[sel]
    d = k * drv[sel]
    keep = np.ones(n, bool)
    for _ in range(3):  # least squares, then drop outliers (3 MAD) and refit
        c = float(np.sum(r[keep] * d[keep]) / np.sum(d[keep] ** 2))
        res = r - c * d
        mad = np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-9
        keep = np.abs(res - np.median(res[keep])) < 3 * 1.4826 * mad
    rms0 = float(np.sqrt(np.mean(r[keep] ** 2)))
    rms1 = float(np.sqrt(np.mean((r[keep] - c * d[keep]) ** 2)))
    share = float(np.median(np.abs(c * drv[sel]) / (np.abs(eps[sel] + c * drv[sel]) + 1e-9)))
    info = {"n": n, "rms_eps_only": rms0, "rms_with_driver": rms1, "driver_share": share}
    if not np.isfinite(c) or c <= 0 or rms1 >= rms0:
        info["reason"] = f"fit gave c = {c:.3g} (rms {rms0:.3f} -> {rms1:.3f}), not usable"
        return 0.0, info
    return c, info


def friction_report(df, steered, hand, out=print):
    """Near straight: how much torque the wheel needs per unit of lateral jerk, by frame source."""
    a = df["actual_lateral_accel"].to_numpy()
    roll, v = df["roll"].to_numpy(), df["v_ego"].to_numpy()
    if "actual_lateral_jerk" not in df:
        out("  (no actual_lateral_jerk column: re-extract for the friction comparison)")
        return
    j = df["actual_lateral_jerk"].to_numpy()
    y = -df["torque_target"].to_numpy()  # torque is left positive, lateral accel/jerk right positive (as the trainer)
    base = (np.abs(a) < 0.15) & (np.abs(roll) < 0.03) & (v >= 7) & (v <= 25) & np.isfinite(j)
    out("  near straight (|a| < 0.15, |roll| < 0.03, 7-25 m/s): torque for the wheel moving at a given lateral jerk")
    out(f"  {'jerk':>12s} " + "".join(f"{name:>18s}" for name in ("openpilot-steered", "hand-driven")))
    for lo, hi in ((0.03, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0)):
        cells = []
        for rows in (steered, hand):
            m = base & rows
            pos, neg = m & (j > lo) & (j <= hi), m & (j < -lo) & (j >= -hi)
            if pos.sum() >= 20 and neg.sum() >= 20:  # odd part: removes any constant offset (road crown, sensor offset)
                cells.append(f"{(y[pos].mean() - y[neg].mean()) / 2:+.3f} ({pos.sum() + neg.sum():>6d})")
            else:
                cells.append(f"{'-':>15s}")
        out(f"  {lo:5.2f}-{hi:<5.2f}  " + "".join(f"{c:>18s}" for c in cells))


def build(df, driver_scale=None, steered_only=False, out=print):
    for col in ("steering_torque_eps", "steering_torque", "lat_active", "torque_applied"):
        if col not in df:
            raise SystemExit(f"ERROR: column {col} missing - re-extract with this version of nnlc-extract")
    steered, hand = select_rows(df)
    if steered_only:
        hand = np.zeros_like(hand)
    lat_only = steered & ~to_bool(df["active"]).to_numpy() if "active" in df else np.zeros(len(df), bool)
    out(f"frames: openpilot-steered {steered.sum():,} (of which lateral-only, not engaged: {lat_only.sum():,}), "
        f"hand-driven {hand.sum():,}")
    k, r, n = fit_eps_scale(df, steered)
    out(f"EPS scale: 1.0 command = {1 / k:.0f} EPS units (steered hands-off frames: {n:,}, correlation {r:.3f})")
    if driver_scale is None:
        c, info = fit_driver_scale(df, steered, hand, k) if hand.any() else (0.0, {"reason": "no hand-driven frames"})
        if "reason" in info:
            out(f"driver torque scale: 0 (EPS torque only) - {info['reason']}")
        else:
            out(f"driver torque scale: 1 driver unit = {c:.2f} EPS units, from {info['n']:,} hand-driven steady-curve frames; "
                f"rms vs openpilot's torque {info['rms_eps_only']:.3f} -> {info['rms_with_driver']:.3f}; "
                f"the driver's hand is {info['driver_share'] * 100:.0f}% of the total there")
    else:
        c = float(driver_scale)
        out(f"driver torque scale: {c:.2f} (given)")
    keep = steered | hand
    res = df[keep].copy()
    res["torque_target"] = k * (res["steering_torque_eps"] + c * res["steering_torque"])
    res["source"] = np.where(steered[keep], STEERED, HAND).astype(np.int8)
    big = res["torque_target"].abs() > 1.5
    if big.any():
        out(f"dropped {int(big.sum()):,} frames needing more than 1.5x the maximum command")
        res = res[~big]
    s = res["source"].to_numpy()
    friction_report(res, s == STEERED, s == HAND, out=out)
    return res, k, c


def main():
    p = argparse.ArgumentParser(description="Torque target from EPS and driver torque, steered and hand-driven frames.")
    p.add_argument("input", help="CSV from nnlc-extract / nnlc-prune-routes (needs steering_torque_eps)")
    p.add_argument("-o", "--output", required=True, help="Output CSV for training")
    p.add_argument("--driver-scale", type=float, default=None,
                   help="Driver torque -> EPS units instead of fitting it (0 = EPS torque only, as twilsonco)")
    p.add_argument("--steered-only", action="store_true", help="Leave out hand-driven frames")
    args = p.parse_args()
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} rows")
    res, _, _ = build(df, args.driver_scale, args.steered_only)
    res.to_csv(args.output, index=False)
    print(f"Saved {len(res):,} rows to {args.output}")


if __name__ == "__main__":
    main()
