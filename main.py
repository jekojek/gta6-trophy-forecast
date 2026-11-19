"""Forecast GTA VI PS5 trophy earn rates on Exophase at a case's target time.

Model (counts decomposition, refresh aware):

  * Every trophy's earn rate is earners / players-tracked.  The two counts
    are modelled apart and divided again at the target, because Exophase
    refreshes its counts in jumps: when a refresh lands, the tracked
    player count leaps while earners barely move, so every rate is diluted
    by roughly the same factor.
  * Players tracked N(t) grows close to a power law in log time:
    log N against log(1 + hours since release) is nearly a straight line
    (launch night: 638 -> 3,868 -> ~13,300 players, slope about 1.05).
    The fit is taken from the case's own distinct snapshots, with a
    slower-growth and a recent-slope scenario bracketing it.
  * Earners E(t) per trophy follow the same log-time shape, with a slope
    that depends on how early / easy the trophy is: story-opening and
    arcade-style trophies climb fast, hard and platinum trophies barely
    move.  Each slope is shrunk toward its group's median slope, because
    a single trophy's counts are noisy (rates are printed to 2 decimals).
  * The spread is built from the scenario spread of N, the fit residual
    of E, and a term that widens with the horizon, so a 90-minute practice
    forecast is tight while the 4.5-day final forecast honestly admits how
    far N and E can wander.  Quantiles are the median's logit shifted by
    the normal quantiles times that spread.
  * Anything unusable falls back to the persistence forecast (last rate,
    random-walk spread), so a case is never left invalid.

Standard library only, deterministic, no network, no clock reads.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Power-law slope of players-tracked growth, calibrated on launch night
# (638 at release, 3,868 after the first refresh, ~13,300 by 17:00 UTC).
PRIOR_N_SLOPE = 1.05
# Earner slope used when a trophy has no usable history at all, by the
# band its last rate falls in.  Launch-night evidence says earners move
# far slower than the tracked count over a frozen page (the first refresh
# divided every rate by ~6 while earners barely moved), so these priors
# stay small; the runtime fit takes over whenever history varies.
PRIOR_E_SLOPE = {"hi": 0.05, "mid": 0.03, "low": 0.02, "rare": 0.01}
# Horizon-growth rate of the logit spread, per band (logit units per
# square-root day).  Rare trophies are steadier than early story ones.
# The floor is set by the final regime: over 4.5 days a power-law
# extrapolation can be off by ~0.5 logit, and the displayed answer is
# rounded to 0.01 points, which alone is ~0.4 logit at a 0.02% rate.
SPREAD_RATE = {"hi": 0.34, "mid": 0.26, "low": 0.22, "rare": 0.20}
PLATINUM_SPREAD_RATE = 0.10
# Overall spread calibration, tuned on synthetic refresh simulations
# (a pure power-law world and a growth-decay world) plus the practice
# proxy: 0.8 keeps the practice proxy under 0.1 while leaving the final
# spread honest about 4.5 days of extrapolation.
SIGMA_TUNE = 0.8


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    hours = (moment - RELEASE).total_seconds() / 3600.0
    return hours if hours > 0.0 else 0.0


def log_time(moment):
    return math.log1p(hours_since_release(moment))


def clamp_pct(value):
    if value < 0.01:
        return 0.01
    if value > 99.99:
        return 99.99
    return value


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(value):
    if value > 20.0:
        return 99.99
    if value < -20.0:
        return 0.01
    return clamp_pct(100.0 / (1.0 + math.exp(-value)))


def raw_logit(pct):
    """Logit without clamping, for CDF work near the 0/100 edges."""
    p = pct / 100.0
    return math.log(p / (1.0 - p))


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def quantile_row(centre_logit, sigma):
    """Five quantiles of the *displayed* rate, which Exophase prints to
    2 decimals (0.00 scores as 0.01).  The continuous forecast is
    logit-normal; each displayed quantum's probability is the mass of
    that 0.01-wide bin, so the row is the exact quantile of the rounded
    variable, found by binary search over the quantum index.  This is
    what puts the 5% quantile on the 0.01 floor for rare trophies whose
    continuous rate sits just above it.
    """
    row = []
    for tau in LEVELS:
        lo, hi = 1, 9999
        while lo < hi:
            mid = (lo + hi) // 2
            edge = (mid + 0.5) / 100.0
            prob = _phi((raw_logit(edge) - centre_logit) / sigma)
            if prob >= tau:
                hi = mid
            else:
                lo = mid + 1
        row.append(lo / 100.0)
    return row


def round4(value):
    """Round to 4 decimals, halves away from zero, as a plain number."""
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 10000.0
    return int(out) if float(out).is_integer() else out


def ols(xs, ys):
    """Least-squares slope and intercept of y on x."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0, my
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def rate_band(rate):
    if rate >= 1.0:
        return "hi"
    if rate >= 0.2:
        return "mid"
    if rate >= 0.05:
        return "low"
    return "rare"


def distinct_snapshots(case):
    """Collapse runs of identical reads (a frozen Exophase page) to one."""
    snaps = []
    prev_key = None
    for read in case.get("history") or []:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key == prev_key:
            continue
        prev_key = key
        snaps.append(read)
    return snaps


def last_known_rates(case):
    """Last non-null rate per trophy, scanning history backwards."""
    trophies = case["trophies"]
    known = [None] * len(trophies)
    for read in reversed(case.get("history") or []):
        rates = read.get("rates") or []
        for i in range(len(trophies)):
            if known[i] is None and i < len(rates) and rates[i] is not None:
                known[i] = rates[i]
        if all(v is not None for v in known):
            break
    return known


def project_players(snaps, target_moment):
    """Return (log N at target, sigma of log N) from the snapshots."""
    x_target = log_time(target_moment)
    points = []
    for read in snaps:
        n = read.get("players_tracked")
        if isinstance(n, (int, float)) and n > 0:
            points.append((log_time(parse_time(read["at"])), math.log(n)))
    if not points:
        return math.log(638.0), 0.35
    if len(points) == 1:
        x0, log_n0 = points[0]
        log_n = log_n0 + PRIOR_N_SLOPE * (x_target - x0)
        return log_n, 0.20
    slope, intercept = ols([p[0] for p in points], [p[1] for p in points])
    slope = min(max(slope, 0.0), 1.6)
    # Recent slope from the last two distinct snapshots.
    (x1, y1), (x2, y2) = points[-2], points[-1]
    recent = (y2 - y1) / (x2 - x1) if x2 > x1 else slope
    recent = min(max(recent, 0.0), 1.6)
    # Three scenarios: full-history fit, recent pace, growth decaying.
    cands = [
        intercept + slope * x_target,
        y2 + recent * (x_target - x2),
        intercept + 0.8 * slope * x_target,
    ]
    centre = statistics.median(cands)
    spread = statistics.pstdev(cands) if len(cands) > 1 else 0.0
    sigma = min(max(spread, 0.10), 0.45)
    # Never forecast fewer tracked players than the last read saw.
    centre = max(centre, points[-1][1])
    return centre, sigma


def earner_slopes(snaps, trophies, known_rates):
    """Per-trophy fitted log-time slope of earners, shrunk to its group."""
    n_t = len(trophies)
    fits = {}
    series = {}
    for i in range(n_t):
        xs, ys = [], []
        for read in snaps:
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            n = read.get("players_tracked")
            if not isinstance(n, (int, float)) or n <= 0:
                continue
            earners = rates[i] * n / 100.0
            xs.append(log_time(parse_time(read["at"])))
            ys.append(math.log(earners + 0.5))
        series[i] = (xs, ys)
        if len(xs) >= 3 and max(ys) - min(ys) > 1e-9:
            slope, _ = ols(xs, ys)
            fits[i] = min(max(slope, 0.0), 2.0)
    # Group median slope, keyed by tier and last-rate band.
    groups = {}
    for i, slope in fits.items():
        tier = trophies[i].get("tier", "bronze")
        band = rate_band(known_rates[i] if known_rates[i] is not None else 0.0)
        groups.setdefault((tier, band), []).append(slope)
    group_med = {k: statistics.median(v) for k, v in groups.items()}
    group_spread = {
        k: (statistics.pstdev(v) if len(v) > 1 else 0.05)
        for k, v in groups.items()
    }
    out = {}
    for i in range(n_t):
        tier = trophies[i].get("tier", "bronze")
        band = rate_band(known_rates[i] if known_rates[i] is not None else 0.0)
        key = (tier, band)
        prior = 0.0 if tier == "platinum" else PRIOR_E_SLOPE[band]
        target = group_med.get(key, prior)
        disp = group_spread.get(key, 0.05)
        if i in fits:
            xs, _ = series[i]
            weight = len(xs) / (len(xs) + 3.0)
            slope = weight * fits[i] + (1.0 - weight) * target
        else:
            slope = target
        out[i] = (slope, fits.get(i, slope), disp, series[i])
    return out


def forecast_trophy(i, trophy, known_rate, slope_info, snaps,
                    log_n_target, sigma_n, horizon_days, target_moment):
    slope, slope_fit, group_disp, (xs, ys) = slope_info
    tier = trophy.get("tier", "bronze")
    last_rate = known_rate if known_rate is not None else 0.0
    band = rate_band(last_rate)

    # A trophy everyone has stays (almost) at 100%: its earners track N.
    # Median at the scoring cap with a modest spread, so a small dip in
    # the rate (new players who have not done it yet) stays inside.
    if last_rate >= 99.5:
        return quantile_row(logit(99.99), 0.35)

    # Earners at the anchor: the fitted level at the last usable read
    # (fitted slope, so the anchor sits on the line), grown forward with
    # the shrunk slope; or the last read's count when there is no fit.
    anchor_x = None
    anchor_e = None
    if len(xs) >= 2:
        _, intercept = ols(xs, ys)
        anchor_x = xs[-1]
        anchor_e = math.exp(intercept + slope_fit * anchor_x)
    elif snaps:
        read = snaps[-1]
        rates = read.get("rates") or []
        n = read.get("players_tracked")
        if i < len(rates) and rates[i] is not None and n:
            anchor_x = log_time(parse_time(read["at"]))
            anchor_e = rates[i] * n / 100.0
    if anchor_e is None or anchor_x is None:
        anchor_x = log_time(parse_time(snaps[0]["at"])) if snaps else 0.0
        anchor_e = 0.5
    if tier == "platinum":
        slope = 0.0
    earners_target = anchor_e * math.exp(
        slope * (log_time(target_moment) - anchor_x))
    if anchor_e > 0.0:
        earners_target = max(earners_target, anchor_e * 0.999)
    n_target = math.exp(log_n_target)
    rate_med = earners_target / n_target * 100.0 if n_target > 0 else last_rate
    rate_med = clamp_pct(rate_med)

    # Spread: N scenarios, E fit noise, group slope spread over the
    # remaining log-time, and a term growing with the horizon.
    resid = 0.10
    if len(xs) >= 3:
        _, intercept = ols(xs, ys)
        errs = [y - (intercept + slope_fit * x) for x, y in zip(xs, ys)]
        resid = math.sqrt(sum(e * e for e in errs) / len(errs))
        resid = min(max(resid, 0.05), 0.60)
    remaining = max(log_time(target_moment) - anchor_x, 0.0)
    sigma_e = math.sqrt(resid ** 2 + (group_disp * remaining) ** 2)
    rate_g = PLATINUM_SPREAD_RATE if tier == "platinum" else SPREAD_RATE[band]
    sigma_h = rate_g * math.sqrt(horizon_days)
    sigma = math.sqrt(sigma_n ** 2 + sigma_e ** 2 + sigma_h ** 2)
    sigma = min(max(sigma * SIGMA_TUNE, 0.12), 1.10)

    centre = logit(rate_med)
    row = quantile_row(centre, sigma)
    # Keep the median continuous: the scored answer is quantised to
    # 0.01 points, and a continuous median hedges the coin flip between
    # two neighbouring quanta instead of betting on one of them.
    row[2] = round4(inv_logit(centre))
    for k in range(1, 5):
        if row[k] < row[k - 1]:
            row[k] = row[k - 1]
    return row


def persistence_rows(case, known_rates):
    """House-style fallback: last rate, random-walk spread."""
    horizon_days = case.get("horizon_minutes", 0) / 1440.0
    spread = 0.5 * math.sqrt(horizon_days) if horizon_days > 0 else 0.0
    # Tier prior when a trophy was never seen in any read.
    tier_prior = {"platinum": 0.05, "gold": 0.1, "silver": 0.2, "bronze": 1.0}
    rows = []
    for trophy, rate in zip(case["trophies"], known_rates):
        base = rate if rate is not None else tier_prior.get(
            trophy.get("tier", "bronze"), 1.0)
        centre = logit(base)
        row = [round4(inv_logit(centre + z * spread)) for z in Z]
        for k in range(1, 5):
            if row[k] < row[k - 1]:
                row[k] = row[k - 1]
        rows.append({"id": trophy["id"], "q": row})
    return {"trophies": rows}


def forecast_case(case):
    trophies = case["trophies"]
    if not trophies:
        return {"trophies": []}
    known = last_known_rates(case)
    snaps = distinct_snapshots(case)
    if not snaps:
        return persistence_rows(case, known)
    target_moment = parse_time(case["target"])
    horizon_days = case.get("horizon_minutes", 0) / 1440.0
    log_n_target, sigma_n = project_players(snaps, target_moment)
    slopes = earner_slopes(snaps, trophies, known)
    rows = []
    for i, trophy in enumerate(trophies):
        row = forecast_trophy(i, trophy, known[i], slopes[i], snaps,
                              log_n_target, sigma_n, horizon_days,
                              target_moment)
        rows.append({"id": trophy["id"], "q": row})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            # Never leave a case without an output: rebuild the case's
            # last rates from whatever could be read, else tier priors.
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                known = last_known_rates(case)
                forecast = persistence_rows(case, known)
            except Exception:
                forecast = {"trophies": []}
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
