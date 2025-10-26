#!/usr/bin/env python3
"""
Daily NHL predictor + season backtester using api-web.nhle.com
Predictions page (xG only, logos, responsive) + Standings page (Conference→Division),
advanced metrics + Wild Card visualization + Playoff Probability (PO%) via Monte Carlo.
"""

import csv
import json
import math
import logging
import random
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional, Set

from zoneinfo import ZoneInfo
import numpy as np
import requests
from scipy.stats import skellam
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =========================
# Config
# =========================
ELO_INIT = 1500.0
K_BASE = 18.0
HOME_ADV_ELO = 35.0
LEAGUE_AVG_GOALS = 6.2
ELO_TO_XG_SCALE = 0.0035
RECENCY_TAU_DAYS = 180.0

# Playoff simulation parameters (tweak if needed)
SIMS = 300                     # number of Monte Carlo seasons (full build)
OT_RATE = 0.23                 # chance a simulated game goes OT/SO (loser gets 1 point)
SCHEDULE_LOOKAHEAD_DAYS = 190  # scan forward for remaining schedule

LOCAL_TZ = ZoneInfo("America/Chicago")
CENTRAL_TZ = ZoneInfo("America/Chicago")

PREDICTIONS_CSV = "predictions_today.csv"
PREDICTIONS_HTML = "predictions_today.html"
STANDINGS_HTML = "standings_today.html"
ELO_DUMP_CSV = "elo_dump.csv"
BACKTEST_CSV = "backtest_results.csv"
BACKTEST_SUMMARY_JSON = "backtest_summary.json"

API_BASE = "https://api-web.nhle.com"
API_SCHEDULE_DATE = API_BASE + "/v1/schedule/{date}"   # YYYY-MM-DD (returns a week bucket)
API_SCORE_DATE    = API_BASE + "/v1/score/{date}"      # YYYY-MM-DD
API_STANDINGS_NOW = API_BASE + "/v1/standings/now"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# =========================
# Small disk cache (/tmp by default)
# =========================
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/tmp"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _cache_path(name: str) -> Path:
    return CACHE_DIR / name

def _read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _write_json(path: Path, obj: Any):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)
    except Exception:
        pass

# =========================
# Networking
# =========================
def session_with_retries():
    s = requests.Session()
    retry = Retry(total=5, connect=5, read=5, backoff_factor=0.4,
                  status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "nhl-predictor/3.5"})
    return s

SESSION = session_with_retries()

def safe_get_json(url: str) -> Optional[Dict[str, Any]]:
    try:
        r = SESSION.get(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.warning(f"Fetch failed for {url}: {e}")
        return None

# =========================
# Helpers: team keys, names, logos, records, time
# =========================
def canonical_team_key(team_obj: dict) -> str:
    abbr = (team_obj.get("abbrev") or team_obj.get("triCode") or "").strip().upper()
    if abbr:
        return abbr
    tid = team_obj.get("id") or team_obj.get("teamId")
    if tid is not None:
        return f"ID{int(tid)}"
    cn = (team_obj.get("commonName") or {}).get("default")
    if cn:
        return cn.strip().upper()
    return "UNKNOWN"

def full_team_name(team_obj: dict) -> str:
    common = (team_obj.get("commonName") or {}).get("default")
    if common:
        return common
    city = (team_obj.get("placeName") or {}).get("default") or ""
    nick = (team_obj.get("teamName") or {}).get("default") or ""
    name = f"{city} {nick}".strip()
    return name or team_obj.get("abbrev") or "Unknown"

def normalize_game_type(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        up = val.strip().upper()
        if up in {"R", "P", "PR"}:
            return up
        try:
            return normalize_game_type(int(up))
        except Exception:
            return ""
    if isinstance(val, (int, float)):
        ival = int(val)
        return "PR" if ival == 1 else "R" if ival == 2 else "P" if ival == 3 else ""
    return ""

def team_logo_candidates(team_key: str) -> List[str]:
    abbr = (team_key or "").upper()
    return [
        f"https://assets.nhle.com/logos/nhl/svg/{abbr}.svg",
        f"https://assets.nhle.com/logos/nhl/svg/{abbr}_light.svg",
        f"https://assets.nhle.com/logos/nhl/light/{abbr}.svg",
        f"https://assets.nhle.com/logos/nhl/svg/{abbr}_dark.svg",
    ]

def primary_team_logo(team_key: str) -> str:
    return team_logo_candidates(team_key)[0]

def get_team_records() -> Dict[str, str]:
    data = safe_get_json(API_STANDINGS_NOW) or {}
    items = data.get("standings") or data.get("records") or data.get("teams") or []
    out: Dict[str, str] = {}
    for it in items:
        try:
            abbr = (
                (it.get("teamAbbrev") if isinstance(it.get("teamAbbrev"), str) else None)
                or (it.get("teamAbbrev", {}) or {}).get("default")
                or (it.get("team", {}) or {}).get("abbrev")
                or (it.get("team", {}) or {}).get("triCode")
                or ""
            ).upper()
            if not abbr:
                continue
            w = it.get("wins") if isinstance(it.get("wins"), int) else (it.get("record", {}) or {}).get("wins") or it.get("w") or 0
            l = it.get("losses") if isinstance(it.get("losses"), int) else (it.get("record", {}) or {}).get("losses") or it.get("l") or 0
            ot = (it.get("otLosses") if isinstance(it.get("otLosses"), int) else None)
            if ot is None: ot = (it.get("overtimeLosses") if isinstance(it.get("overtimeLosses"), int) else None)
            if ot is None: ot = (it.get("otl") if isinstance(it.get("otl"), int) else None)
            if ot is None: ot = (it.get("record", {}) or {}).get("ot") or (it.get("record", {}) or {}).get("overtimeLosses") or 0
            out[abbr] = f"{int(w)}-{int(l)}-{int(ot)}"
        except Exception:
            continue
    return out

def _utc_to_local_dt(utc_str: str, tz: ZoneInfo) -> Optional[datetime]:
    if not utc_str:
        return None
    try:
        if utc_str.endswith("Z"):
            dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(utc_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(tz)
    except Exception:
        return None

def fmt_local_time(dt_local: Optional[datetime]) -> str:
    if not dt_local:
        return ""
    try:
        return dt_local.strftime("%-I:%M %p")
    except Exception:
        return dt_local.strftime("%I:%M %p").lstrip("0")

# =========================
# Odds helpers
# =========================
def american_moneyline(p: float) -> int:
    p = max(1e-6, min(1 - 1e-6, float(p)))
    if p >= 0.5:
        return int(round(- (p / (1.0 - p)) * 100))
    else:
        return int(round(((1.0 - p) / p) * 100))

# =========================
# Elo & probabilities
# =========================
def expected_win_prob_from_elo(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** (-(elo_a - elo_b) / 400.0))

def update_elo(state, home_key, away_key, home_goals, away_goals, game_date_dt, ref_today_dt):
    h = state.get("elo", {}).get(home_key, ELO_INIT)
    a = state.get("elo", {}).get(away_key, ELO_INIT)

    exp_home = expected_win_prob_from_elo(h + HOME_ADV_ELO, a)
    res_home = 1.0 if home_goals > away_goals else 0.0 if home_goals < away_goals else 0.5

    mov = abs(home_goals - away_goals)
    mov_mult = 1.0 + 0.05 * max(0, mov - 1)

    ref_d = ref_today_dt.date() if hasattr(ref_today_dt, "date") else ref_today_dt
    game_d = game_date_dt.date() if hasattr(game_date_dt, "date") else game_date_dt
    days_ago = (ref_d - game_d).days

    weight = math.exp(-max(0, days_ago) / RECENCY_TAU_DAYS)
    k = K_BASE * weight

    h_new = h + k * mov_mult * (res_home - exp_home)
    a_new = a - k * mov_mult * (res_home - exp_home)

    state.setdefault("elo", {})
    state["elo"][home_key] = h_new
    state["elo"][away_key] = a_new

def expected_goals(home_elo, away_elo) -> Tuple[float, float]:
    diff = (home_elo + HOME_ADV_ELO) - away_elo
    tilt = math.tanh(ELO_TO_XG_SCALE * diff)
    home_share = 0.5 + 0.25 * tilt

    avg_strength = (home_elo + away_elo) / 2.0
    strength_factor = 1.0 + (avg_strength - ELO_INIT) / 800.0
    total_goals = max(3.8, min(8.5, LEAGUE_AVG_GOALS * strength_factor))

    home_xg = max(0.2, total_goals * home_share)
    away_xg = max(0.2, total_goals - home_xg)
    return home_xg, away_xg

def win_probs_from_skellam(home_xg, away_xg) -> Tuple[float, float]:
    p_home = 1.0 - skellam.cdf(0, home_xg, away_xg)
    p_away = skellam.cdf(-1, home_xg, away_xg)
    s = p_home + p_away
    if s == 0:
        return 0.5, 0.5
    return p_home / s, p_away / s

# =========================
# NHL API helpers (schedule/score)
# =========================
def _iter_schedule_candidates(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not payload:
        return []
    if isinstance(payload.get("games"), list):
        return payload["games"]
    out = []
    for day in (payload.get("gameWeek") or []):
        g = day.get("games")
        if isinstance(g, list):
            out.extend(g)
    return out

@dataclass
class GameSched:
    game_id: Any
    home_key: str
    away_key: str
    game_type: str
    start_utc_str: str

def _parse_sched_item(g: dict) -> Optional[GameSched]:
    if not isinstance(g, dict):
        return None
    home, away = g.get("homeTeam", {}), g.get("awayTeam", {})
    start_utc = (g.get("startTimeUTC") or g.get("startTimeUTCISO") or "") or ""
    start_utc_clean = start_utc[:-1] if start_utc.endswith("Z") else start_utc
    return GameSched(
        game_id=g.get("id") or g.get("gamePk") or g.get("gameId"),
        home_key=canonical_team_key(home),
        away_key=canonical_team_key(away),
        game_type=normalize_game_type(g.get("gameType")),
        start_utc_str=start_utc_clean
    )

def get_schedule_for_local_date(local_date: datetime.date) -> List[GameSched]:
    ds = local_date.strftime("%Y-%m-%d")
    data = safe_get_json(API_SCHEDULE_DATE.format(date=ds))
    candidates = _iter_schedule_candidates(data)
    out: List[GameSched] = []
    for g in candidates:
        gs = _parse_sched_item(g)
        if not gs:
            continue
        dt_local = _utc_to_local_dt(gs.start_utc_str, LOCAL_TZ)
        if not dt_local or dt_local.date() != local_date:
            continue
        out.append(gs)
    return out

def get_remaining_regular_season(today_local: datetime.date) -> List[GameSched]:
    """Fetch remaining regular-season games (deduped) scanning week-by-week, cached for the day."""
    key = f"sched_{today_local.isoformat()}.json"
    p = _cache_path(key)
    cached = _read_json(p)
    if cached and isinstance(cached, list):
        out = []
        for g in cached:
            out.append(GameSched(**g))
        logging.info(f"Remaining regular-season games (from cache): {len(out)}")
        return out

    seen: Set[Any] = set()
    games: List[GameSched] = []
    for k in range(0, SCHEDULE_LOOKAHEAD_DAYS + 1, 7):
        d = today_local + timedelta(days=k)
        data = safe_get_json(API_SCHEDULE_DATE.format(date=d.strftime("%Y-%m-%d")))
        candidates = _iter_schedule_candidates(data)
        for g in candidates:
            gs = _parse_sched_item(g)
            if not gs or gs.game_type != "R":
                continue
            dt_local = _utc_to_local_dt(gs.start_utc_str, LOCAL_TZ)
            if not dt_local or dt_local.date() < today_local:
                continue
            if gs.game_id in seen:
                continue
            seen.add(gs.game_id)
            games.append(gs)

    _write_json(p, [g.__dict__ for g in games])
    logging.info(f"Remaining regular-season games collected: {len(games)} (cached)")
    return games

# =========================
# History build + cached Elo
# =========================
def log_elo_summary(state: dict):
    items = list(state.get("elo", {}).items())
    if not items:
        logging.warning("Elo is empty.")
        return
    vals = [v for _, v in items]
    logging.info(f"Elo summary: min={min(vals):.1f}, max={max(vals):.1f}, mean={np.mean(vals):.1f}, sd={np.std(vals):.1f}")
    with open(ELO_DUMP_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["team_key", "elo"])
        w.writerows(sorted(items, key=lambda kv: kv[1], reverse=True))

def build_elo_from_history(state, start_date, end_date, include_types=("R","P","PR")):
    ref_today_dt = datetime.now(tz=LOCAL_TZ)
    cur = start_date
    total = 0
    while cur <= end_date:
        ds = cur.strftime("%Y-%m-%d")
        data = safe_get_json(API_SCORE_DATE.format(date=ds))
        games = (data or {}).get("games") or []
        for g in games:
            state_str = (g.get("gameState") or "").upper()
            if state_str not in {"FINAL", "OFF", "COMPLETE", "COMPLETED", "GAME_OVER"}:
                continue
            home, away = g.get("homeTeam", {}), g.get("awayTeam", {})
            gt = normalize_game_type(g.get("gameType"))
            if include_types and gt and gt not in include_types:
                continue
            update_elo(state,
                       canonical_team_key(home),
                       canonical_team_key(away),
                       int(home.get("score", 0)),
                       int(away.get("score", 0)),
                       datetime.strptime(ds, "%Y-%m-%d"),
                       ref_today_dt)
        total += len(games)
        cur += timedelta(days=1)
    logging.info(
        f"Elo built from {start_date} to {end_date}. "
        f"{len(state.get('elo', {}))} teams rated. {total} final games ingested."
    )
    log_elo_summary(state)

def get_or_build_elo_cached(end_date: datetime.date):
    """
    Return {'elo': {...}} built from Oct 1 of previous season up to end_date.
    Cached in /tmp/elo_YYYY-MM-DD.json
    """
    key = f"elo_{end_date.isoformat()}.json"
    p = _cache_path(key)
    cached = _read_json(p)
    if cached and isinstance(cached, dict) and "elo" in cached:
        return cached

    start_date = datetime(end_date.year - 1, 10, 1, tzinfo=LOCAL_TZ).date()
    state = {"elo": {}}
    build_elo_from_history(state, start_date, end_date, include_types=("R","P"))
    _write_json(p, state)
    return state

# =========================
# Prediction + CSV/HTML
# =========================
def predict_day(state, local_date: datetime.date, records: Dict[str, str]) -> List[Dict[str, Any]]:
    games = get_schedule_for_local_date(local_date)
    preds = []
    for g in games:
        home_key, away_key = g.home_key, g.away_key
        helo = state.get("elo", {}).get(home_key, ELO_INIT)
        aelo = state.get("elo", {}).get(away_key, ELO_INIT)
        hxg, axg = expected_goals(helo, aelo)
        p_home, p_away = win_probs_from_skellam(hxg, axg)

        ml_home = american_moneyline(p_home)
        ml_away = american_moneyline(p_away)

        preds.append({
            "date": local_date.isoformat(),
            "gameId": g.game_id,
            "away_key": away_key,
            "home_key": home_key,
            "away_name": "",
            "home_name": "",
            "pred_xg_home": round(hxg, 2),
            "pred_xg_away": round(axg, 2),
            "p_home_win": round(p_home, 3),
            "p_away_win": round(p_away, 3),
            "ml_home": ml_home,
            "ml_away": ml_away,
            "home_logo": primary_team_logo(home_key),
            "away_logo": primary_team_logo(away_key),
            "home_logo_alts": json.dumps(team_logo_candidates(home_key)),
            "away_logo_alts": json.dumps(team_logo_candidates(away_key)),
            "home_record": records.get(home_key, ""),
            "away_record": records.get(away_key, ""),
            "local_time": fmt_local_time(_utc_to_local_dt(g.start_utc_str, LOCAL_TZ)),
            "utc_time": g.start_utc_str,
            "game_type": g.game_type or "",
        })
    return preds

def write_csv(rows, path):
    if not rows:
        with open(path, "w") as f:
            f.write("no games\n")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

# =========================
# Playoff seeding helpers (used for odds & for table rendering)
# =========================
def _div_code(name: str) -> str:
    n = (name or "").lower()
    if "atl" in n: return "ATL"
    if "met" in n: return "MET"
    if "cent" in n: return "CEN"
    if "pac" in n: return "PAC"
    return (name or "DIV")[:3].upper()

def _tie_key(r: dict):
    # Sort by points, then wins, then goal differential
    return (r.get("pts", 0), r.get("w", 0), r.get("diff", 0))

def mark_playoff_seeds(rows: List[dict]) -> None:
    for r in rows:
        r["_seed"] = ""
        r["_is_div_top3"] = False
        r["_is_wc"] = False
        r["_wc_rank"] = None

    by_conf: Dict[str, Dict[str, List[dict]]] = {}
    for r in rows:
        c = r.get("conference", "Unknown Conference")
        d = r.get("division", "Unknown Division")
        by_conf.setdefault(c, {}).setdefault(d, []).append(r)

    for conf, divs in by_conf.items():
        leftovers: List[dict] = []
        for div, teams in divs.items():
            teams.sort(key=_tie_key, reverse=True)
            for i, t in enumerate(teams):
                if i < 3:
                    t["_seed"] = f"{_div_code(div)}-{i+1}"
                    t["_is_div_top3"] = True
                else:
                    leftovers.append(t)
        leftovers.sort(key=_tie_key, reverse=True)
        for i, t in enumerate(leftovers, start=1):
            t["_wc_rank"] = i
            if i <= 2:
                t["_is_wc"] = True
                t["_seed"] = f"WC-{i}"

# =========================
# Playoff probability simulation
# =========================
def get_standings_now_rows() -> List[dict]:
    data = safe_get_json(API_STANDINGS_NOW) or {}
    items = data.get("standings") or data.get("records") or data.get("teams") or []
    rows: List[dict] = []

    def gi(d, *keys, default=0):
        for k in keys:
            v = d.get(k)
            if isinstance(v, (int, float)):
                return v
        return default

    for it in items:
        try:
            abbr = (
                (it.get("teamAbbrev") if isinstance(it.get("teamAbbrev"), str) else None)
                or (it.get("teamAbbrev", {}) or {}).get("default")
                or (it.get("team", {}) or {}).get("abbrev")
                or (it.get("team", {}) or {}).get("triCode")
                or ""
            ).upper()
            if not abbr:
                continue

            w  = int(gi(it, "wins", "w"))
            l  = int(gi(it, "losses", "l"))
            ot = int(gi(it, "otLosses", "overtimeLosses", "otl", default=0))
            gp = int(gi(it, "gamesPlayed", default=(w + l + ot)))
            pts = int(gi(it, "points", default=(2*w + ot)))

            gf = int(gi(it, "goalsFor", "goalFor", default=0))
            ga = int(gi(it, "goalsAgainst", "goalAgainst", default=0))

            l10_gf = float(gi(it, "l10GoalsFor", default=0.0))
            l10_ga = float(gi(it, "l10GoalsAgainst", default=0.0))

            conference = (
                it.get("conferenceName")
                or (it.get("conference") or {}).get("name")
                or (it.get("conference", {}) or {}).get("default")
                or "Unknown Conference"
            )
            division = (
                it.get("divisionName")
                or (it.get("division") or {}).get("name")
                or (it.get("division", {}) or {}).get("default")
                or "Unknown Division"
            )

            # Advanced metrics you asked for
            ptspct = round(pts / max(1, gp * 2), 3)
            winpct = round((w / max(1, gp)), 3)

            if (gf + ga) > 0:
                pythag = round((gf ** 2.19) / ((gf ** 2.19) + (ga ** 2.19)), 3)
            else:
                pythag = 0.500

            xP = round((gp * 2) * pythag, 2)
            xW = round(gp * pythag, 2)
            xvrP = round(pts - xP, 2)
            xvrW = round(w - xW, 2)
            pace = int(ptspct * 164)
            xTP = int(pythag * 164)

            if (l10_gf + l10_ga) > 0:
                l10pythag = round((l10_gf ** 2.19) / ((l10_gf ** 2.19) + (l10_ga ** 2.19)), 3)
            else:
                l10pythag = 0.500

            rows.append({
                "abbr": abbr,
                "name": full_team_name(it),
                "logo": primary_team_logo(abbr),
                "logo_alts": json.dumps(team_logo_candidates(abbr)),
                "conference": conference,
                "division": division,

                "gp": gp, "w": w, "l": l, "ot": ot, "pts": pts,
                "ptspct": ptspct, "winpct": winpct,
                "gf": gf, "ga": ga, "diff": int(gf - ga),

                "pythag": pythag, "xP": xP, "xW": xW,
                "xvrP": xvrP, "xvrW": xvrW,
                "pace": pace, "xTP": xTP,
                "l10pythag": l10pythag,

                "po": None, "po_str": "",
            })
        except Exception:
            continue

    rows.sort(key=lambda r: (r["conference"], r["division"], r["pts"], r["w"], r["diff"]), reverse=True)
    return rows

def _seed_and_pick_playoff_qualifiers(sim_rows: List[dict]) -> Set[str]:
    """
    NHL format:
      - Top 3 in each division
      - Next 2 by points in each conference (Wild Cards)
    Tiebreakers approximated: points, then wins, then goal diff.
    """
    rows = [dict(r) for r in sim_rows]
    mark_playoff_seeds(rows)
    qualified = set()
    for r in rows:
        if r.get("_is_div_top3") or r.get("_is_wc"):
            qualified.add(r["abbr"])
    return qualified

def simulate_playoff_probs(state: dict,
                           base_rows: List[dict],
                           today_local: datetime.date,
                           sims: int = SIMS,
                           ot_rate: float = OT_RATE) -> Dict[str, float]:
    """
    Monte Carlo the remainder of regular season using current Elo for win probs:
      - Winner gets 2 points.
      - If OT/SO (with probability `ot_rate`), loser gets 1; else (reg) loser gets 0.
    Returns dict {abbr: probability of making playoffs}.
    """
    base = {
        r["abbr"]: {
            "pts": int(r["pts"]),
            "w": int(r["w"]),
            "diff": int(r["diff"]),
            "conference": r["conference"],
            "division": r["division"],
        } for r in base_rows
    }
    remaining = get_remaining_regular_season(today_local)

    elo = state.get("elo", {})
    def _elo(k: str) -> float:
        return float(elo.get(k, ELO_INIT))

    # Precompute win probs once
    pre = []
    for g in remaining:
        he, ae = _elo(g.home_key), _elo(g.away_key)
        hxg, axg = expected_goals(he, ae)
        p_home, _ = win_probs_from_skellam(hxg, axg)
        pre.append((g.home_key, g.away_key, p_home))

    makes = {abbr: 0 for abbr in base.keys()}
    rng = random.Random(12345)

    for _ in range(sims):
        sim = {k: dict(v) for k, v in base.items()}
        for hk, ak, p_home in pre:
            if rng.random() < p_home:
                sim[hk]["pts"] += 2
                sim[hk]["w"] += 1
                if rng.random() < ot_rate:
                    sim[ak]["pts"] += 1
            else:
                sim[ak]["pts"] += 2
                sim[ak]["w"] += 1
                if rng.random() < ot_rate:
                    sim[hk]["pts"] += 1

        sim_rows = [{
            "abbr": abbr,
            "conference": v["conference"],
            "division": v["division"],
            "pts": v["pts"],
            "w": v["w"],
            "diff": v["diff"],
        } for abbr, v in sim.items()]

        qualified = _seed_and_pick_playoff_qualifiers(sim_rows)
        for abbr in qualified:
            makes[abbr] += 1

    return {abbr: makes[abbr] / float(max(1, sims)) for abbr in makes}

def attach_po_to_rows(rows: List[dict], po: Dict[str, float]) -> None:
    for r in rows:
        p = float(po.get(r["abbr"], 0.0))
        r["po"] = p
        r["po_str"] = f"{p*100:.1f}%"

# =========================
# Predictions HTML (xG only, responsive)
# =========================
def write_html(preds: List[Dict[str, Any]], path: str, report_date: str, season_line: Optional[str] = None):
    updated_time = datetime.now(tz=CENTRAL_TZ).strftime("%a, %b %d, %Y — %I:%M %p %Z")

    if not preds:
        html = """
<!doctype html>
<html><head><meta charset="utf-8"><title>NHL Predictions %%DATE%%</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#0b1020;--panel:#121933;--panel2:#0e1630;--txt:#e8ecff;--muted:#9fb1ff;--border:#1e2748;--border-strong:#29345e;--accent:#7aa2ff;}
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--txt);margin:0}
.wrapper{max-width:1150px;margin:28px auto;padding:0 16px}
h1{font-weight:800;margin:0 0 16px}
.nav{display:flex;gap:10px;margin:0 0 14px}
.nav a{padding:8px 10px;border:1px solid var(--border);border-radius:10px;color:var(--muted);text-decoration:none}
.nav a.active{background:linear-gradient(145deg,var(--panel),var(--panel2));color:#fff;border-color:var(--accent)}
.seasonline{color:var(--muted);margin:-6px 0 16px;font-weight:600}
.card{background:#121933;border:1px solid #1e2748;border-radius:16px;padding:16px;box-shadow:0 10px 30px rgba(0,0,0,.25)}
.empty{opacity:.85}
.footer{color:var(--muted);font-size:12px;margin-top:16px;text-align:left;opacity:1;line-height:1.5}
.footer a{color:var(--accent);text-decoration:underline;font-weight:600}
</style></head>
<body>
<div class="wrapper">
  <h1>NHL Predictions — %%DATE%%</h1>
  <div class="nav">
    <a href="/" class="active">Predictions</a>
    <a href="/standings">Standings</a>
  </div>
  %%SEASONLINE%%
  <div class="card empty">No games found.</div>
  <div class="footer">
    <div>Last updated at: <strong>%%UPDATED%%</strong></div>
    <div>Generated by <a href="https://x.com/reagankingisles">@ReaganKingIsles</a>. Logos © NHL/teams; loaded from NHL CDN.</div>
  </div>
</div></body></html>
""".replace("%%DATE%%", report_date)
        season_html = f'<div class="seasonline">{season_line}</div>' if season_line else ''
        html = html.replace("%%SEASONLINE%%", season_html).replace("%%UPDATED%%", updated_time)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return

    def row_html(p):
        ph = f"{p['p_home_win'] * 100:.1f}%"
        pa = f"{p['p_away_win'] * 100:.1f}%"
        alts_home = p.get("home_logo_alts", "[]")
        alts_away = p.get("away_logo_alts", "[]")
        home_name = p['home_key'] + (f" ({p['home_record']})" if p.get('home_record') else "")
        away_name = p['away_key'] + (f" ({p['away_record']})" if p.get('away_record') else "")
        time_str = p.get("local_time", "")
        return f"""
<tr>
  <td class="teams">
    <div class="team">
      <img class="logo" src="{p['away_logo']}" data-alts='{alts_away}' alt="{p['away_key']}" loading="lazy"/>
      <div class="meta">
        <div class="abbr">{away_name}</div>
      </div>
    </div>
    <div class="vs">at <span class="time" data-utc="{p.get('utc_time','')}">{time_str}</span></div>
    <div class="team home">
      <img class="logo" src="{p['home_logo']}" data-alts='{alts_home}' alt="{p['home_key']}" loading="lazy"/>
      <div class="meta">
        <div class="abbr">{home_name}</div>
      </div>
    </div>
  </td>

  <td class="prob" data-label="Win Prob">
    <div>Away: <b>{pa}</b></div>
    <div>Home: <b>{ph}</b></div>
  </td>
  <td class="ml" data-label="Implied ML">
    <div>Away: <b>{p['ml_away']:+d}</b></div>
    <div>Home: <b>{p['ml_home']:+d}</b></div>
  </td>
  <td class="xg" data-label="xG">
    <div>Away xG: <b>{p['pred_xg_away']:.2f}</b></div>
    <div>Home xG: <b>{p['pred_xg_home']:.2f}</b></div>
  </td>
</tr>"""

    rows_html = "\n".join(row_html(p) for p in preds)

    html = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>NHL Predictions %%DATE%%</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#0b1020;--panel:#121933;--panel2:#0e1630;--txt:#e8ecff;--muted:#9fb1ff;--border:#1e2748;--border-strong:#29345e;--accent:#7aa2ff;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Helvetica,Arial,sans-serif}
.wrapper{max-width:1150px;margin:28px auto;padding:0 16px}
h1{margin:0 0 18px;font-weight:800;letter-spacing:.3px}
.nav{display:flex;gap:10px;margin:0 0 14px}
.nav a{padding:8px 10px;border:1px solid var(--border);border-radius:10px;color:var(--muted);text-decoration:none}
.nav a.active{background:linear-gradient(145deg,var(--panel),var(--panel2));color:#fff;border-color:var(--accent)}
.seasonline{color:var(--muted);margin:-6px 0 16px;font-weight:600}
.subtitle{color:var(--muted);margin-bottom:18px}
.table-card{background:linear-gradient(145deg,var(--panel),var(--panel2));border:1px solid var(--border);border-radius:16px;padding:12px;box-shadow:0 10px 30px rgba(0,0,0,.25)}
table{width:100%;border-collapse:separate;border-spacing:0 8px}
thead th{text-align:left;font-weight:700;color:var(--muted);font-size:14px;padding:10px;border-bottom:1px solid var(--border)}
tbody tr{background:rgba(255,255,255,.02);border:2px solid var(--border-strong);border-radius:10px;transition:background .2s,border-color .2s}
tbody tr:hover{background:rgba(255,255,255,.05);border-color:var(--accent)}
tbody td{padding:12px 10px;vertical-align:middle}
.teams{min-width:420px}
.team{display:flex;align-items:center;gap:10px}
.team img.logo{width:34px;height:34px;object-fit:contain;filter:drop-shadow(0 1px 2px rgba(0,0,0,.4))}
.team .meta .abbr{font-weight:700}
.vs{margin:8px 6px;color:var(--muted);font-size:12px}
.vs .time{font-weight:700;color:#fff;margin-left:6px}
.prob,.ml,.xg{white-space:nowrap}
b{color:#fff}
.note{margin-top:10px;color:var(--muted);font-size:12px}
.footer{color:var(--muted);font-size:12px;margin-top:16px;text-align:left;opacity:1;line-height:1.5}
.footer a{color:var(--accent);text-decoration:underline;font-weight:600}
.footer a:hover,.footer a:focus{text-decoration:none;filter:brightness(1.15)}
@media (max-width:640px){
  .wrapper{max-width:100%;padding:0 12px}
  .table-card{padding:10px}
  table,thead,tbody,th,td,tr{display:block}
  thead{position:absolute;left:-9999px;top:-9999px}
  tbody tr{border-radius:12px;margin:10px 0;padding:6px 6px 10px}
  td.teams{padding:10px 8px 8px;border-bottom:1px solid var(--border);min-width:0}
  .team{gap:8px}
  .team img.logo{width:28px;height:28px}
  .vs{margin:6px 0 4px;font-size:11px}
  td.prob,td.ml,td.xg{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:8px 8px;white-space:normal}
  td.prob::before,td.ml::before,td.xg::before{content:attr(data-label);color:var(--muted);font-weight:700;letter-spacing:.2px}
}
</style>
</head>
<body>
  <div class="wrapper">
    <h1>NHL Predictions — %%DATE%%</h1>
    <div class="nav">
      <a href="/" class="active">Predictions</a>
      <a href="/standings">Standings</a>
    </div>
    %%SEASONLINE%%
    <div class="subtitle">Implied moneylines are vig-free (fair odds) derived from model probabilities.</div>
    <div class="table-card">
      <table>
        <thead>
          <tr>
            <th>Matchup</th>
            <th>Win Prob</th>
            <th>Implied ML</th>
            <th>xG</th>
          </tr>
        </thead>
        <tbody>%%ROWS%%</tbody>
      </table>
      <div class="note">xG = expected goals.</div>
    </div>
    <div class="footer">
      <div>Last updated at: <strong>%%UPDATED%%</strong></div>
      <div>Generated by <a href="https://x.com/reagankingisles">@ReaganKingIsles</a>. Logos © NHL/teams; loaded from NHL CDN.</div>
    </div>
  </div>
<script>
// Logo fallbacks
document.querySelectorAll('img.logo').forEach(function(img){
  let alts = []; try { alts = JSON.parse(img.dataset.alts || '[]'); } catch(e) {}
  let i = 0;
  img.onerror = function(){
    if (i < alts.length) { img.src = alts[i++]; }
    else { img.onerror = null; img.style.visibility = 'hidden'; }
  };
});
// Localize game times
document.querySelectorAll('.time[data-utc]').forEach(function(el){
  const utc = el.getAttribute('data-utc');
  if (utc) {
    const d = new Date(utc + 'Z');
    el.textContent = d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  }
});
</script>
</body></html>
"""
    season_html = f'<div class="seasonline">{season_line}</div>' if season_line else ''
    html = (html
            .replace("%%ROWS%%", rows_html)
            .replace("%%DATE%%", report_date)
            .replace("%%SEASONLINE%%", season_html)
            .replace("%%UPDATED%%", updated_time))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

# =========================
# Standings HTML (Conference→Division + Wild Card) WITH PO%
# =========================
def write_html_standings(rows: List[dict], path: str, report_date: str):
    updated_time = datetime.now(tz=CENTRAL_TZ).strftime("%a, %b %d, %Y — %I:%M %p %Z")

    # Tag playoff seeds for highlighting
    mark_playoff_seeds(rows)

    # Group by conference → division
    by_conf: Dict[str, Dict[str, List[dict]]] = {}
    for r in rows:
        c = r.get("conference", "Unknown Conference")
        d = r.get("division", "Unknown Division")
        by_conf.setdefault(c, {}).setdefault(d, []).append(r)

    def conf_order_key(c):
        cu = c.upper()
        if "EAST" in cu: return 0
        if "WEST" in cu: return 1
        return 2
    conferences = sorted(by_conf.keys(), key=lambda c: (conf_order_key(c), c))

    def thead_html() -> str:
        return """
<thead>
  <tr>
    <th data-key="seed">Seed</th>
    <th data-key="team" class="nosort">Team</th>
    <th data-key="gp">GP</th>
    <th data-key="w">W</th>
    <th data-key="l">L</th>
    <th data-key="ot">OT</th>
    <th data-key="pts">PTS</th>
    <th data-key="ptspct">P%</th>
    <th data-key="winpct">Win%</th>
    <th data-key="gf">GF</th>
    <th data-key="ga">GA</th>
    <th data-key="diff">DIFF</th>
    <th data-key="pythag">Pyth</th>
    <th data-key="xP">xP</th>
    <th data-key="xW">xW</th>
    <th data-key="xvrP">xPΔ</th>
    <th data-key="xvrW">xWΔ</th>
    <th data-key="pace">Pace</th>
    <th data-key="xTP">xTP</th>
    <th data-key="l10pythag">L10 Pyth</th>
    <th data-key="po">PO%</th>
  </tr>
</thead>"""

    def tr(r, extra_classes: str = "") -> str:
        qclass = " q" if (r.get("_is_div_top3") or r.get("_is_wc")) else ""
        cls = (extra_classes + qclass).strip()
        cls_attr = f' class="{cls}"' if cls else ""
        seed = r.get("_seed", "")
        return f"""
<tr{cls_attr}>
  <td class="seed" data-val="{seed}">{seed}</td>
  <td class="teamcell">
    <img class="logo" src="{r['logo']}" data-alts='{r['logo_alts']}' alt="{r['abbr']}" loading="lazy"/>
    <span class="abbr">{r['abbr']}</span>
    <span class="name">{r['name']}</span>
  </td>
  <td data-val="{r['gp']}">{r['gp']}</td>
  <td data-val="{r['w']}">{r['w']}</td>
  <td data-val="{r['l']}">{r['l']}</td>
  <td data-val="{r['ot']}">{r['ot']}</td>
  <td data-val="{r['pts']}"><b>{r['pts']}</b></td>
  <td data-val="{r['ptspct']}">{r['ptspct']:.3f}</td>
  <td data-val="{r['winpct']}">{r['winpct']:.3f}</td>
  <td data-val="{r['gf']}">{r['gf']}</td>
  <td data-val="{r['ga']}">{r['ga']}</td>
  <td data-val="{r['diff']}">{r['diff']:+d}</td>
  <td data-val="{r['pythag']}">{r['pythag']:.3f}</td>
  <td data-val="{r['xP']}">{r['xP']:.2f}</td>
  <td data-val="{r['xW']}">{r['xW']:.2f}</td>
  <td data-val="{r['xvrP']}">{r['xvrP']:+.2f}</td>
  <td data-val="{r['xvrW']}">{r['xvrW']:+.2f}</td>
  <td data-val="{r['pace']}">{r['pace']}</td>
  <td data-val="{r['xTP']}">{r['xTP']}</td>
  <td data-val="{r['l10pythag']}">{r['l10pythag']:.3f}</td>
  <td data-val="{(r['po'] or 0.0)}">{r['po_str'] or '—'}</td>
</tr>"""

    # Build sections per conference
    conf_sections = []
    for conf in conferences:
        divs = by_conf[conf]
        def div_order_key(d):
            du = d.upper()
            if "METRO" in du: return 0
            if "ATLANT" in du: return 1
            if "CENTRAL" in du: return 2
            if "PACIFIC" in du: return 3
            return 4
        divisions = sorted(divs.keys(), key=lambda d: (div_order_key(d), d))

        # Division tables
        div_tables = []
        for div in divisions:
            teams = list(divs[div])
            teams.sort(key=lambda r: (r["pts"], r["w"], r["diff"]), reverse=True)
            body = "\n".join(tr(t) for t in teams)
            div_tables.append(f"""
<section class="division">
  <h3 class="divhdr">{div}</h3>
  <div class="card scroller">
    <table class="standings" data-division="{div}">
      {thead_html()}
      <tbody>
        {body}
      </tbody>
    </table>
  </div>
</section>
""")

        # Wild Card table (conference-wide outside of top-3)
        conf_rows = [r for d in divisions for r in by_conf[conf][d]]
        wc_pool = [r for r in conf_rows if not r.get("_is_div_top3")]
        wc_pool.sort(key=lambda r: (r["pts"], r["w"], r["diff"]), reverse=True)

        def wc_tr(r, idx):
            cutclass = " cutline" if idx == 3 else ""
            return tr(r, extra_classes=("wc" + cutclass))

        wc_body = "\n".join(wc_tr(r, i+1) for i, r in enumerate(wc_pool))
        wc_table = f"""
<section class="wild">
  <h3 class="divhdr">Wild Card</h3>
  <div class="card scroller">
    <table class="standings wild" data-division="Wild Card">
      {thead_html()}
      <tbody>
        {wc_body}
      </tbody>
    </table>
  </div>
</section>
"""

        conf_sections.append(f"""
<section class="conference">
  <h2 class="confhdr">{conf}</h2>
  {''.join(div_tables)}
  {wc_table}
</section>
""")

    html = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>NHL Standings %%DATE%%</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#0b1020;--panel:#121933;--panel2:#0e1630;--txt:#e8ecff;--muted:#9fb1ff;--border:#1e2748;--accent:#7aa2ff;--cut:#7aa2ff55;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Helvetica,Arial,sans-serif}
.wrapper{max-width:1150px;margin:28px auto;padding:0 16px}
h1{margin:0 0 18px;font-weight:800;letter-spacing:.3px}
.nav{display:flex;gap:10px;margin:0 0 14px}
.nav a{padding:8px 10px;border:1px solid var(--border);border-radius:10px;color:var(--muted);text-decoration:none}
.nav a.active{background:linear-gradient(145deg,var(--panel),var(--panel2));color:#fff;border-color:var(--accent)}
.confhdr{margin:18px 0 8px;font-size:20px;font-weight:900;letter-spacing:.3px}
.divhdr{margin:14px 0 10px;font-size:16px;font-weight:800;letter-spacing:.2px}
.card{background:linear-gradient(145deg,var(--panel),var(--panel2));border:1px solid var(--border);border-radius:16px;padding:12px;box-shadow:0 10px 30px rgba(0,0,0,.25)}
.scroller{overflow:auto}
table{width:100%;min-width:1040px;border-collapse:separate;border-spacing:0 8px}
thead th{text-align:left;font-weight:700;color:var(--muted);font-size:14px;padding:10px;border-bottom:1px solid var(--border);cursor:pointer;white-space:nowrap}
tbody tr{background:rgba(255,255,255,.02);border:2px solid var(--border);border-radius:10px}
tbody tr.q{background:rgba(122,162,255,.08);border-color:#2a3a6e}
tbody tr.wc.q{background:rgba(122,162,255,.10);border-color:#3350a0}
tbody tr.cutline td{border-top:2px solid var(--cut)}
tbody td{padding:10px;white-space:nowrap}
.seed{font-weight:800;color:#fff}
.teamcell{display:flex;align-items:center;gap:10px}
.teamcell .logo{width:28px;height:28px;object-fit:contain;filter:drop-shadow(0 1px 2px rgba(0,0,0,.4))}
.teamcell .abbr{font-weight:700}
.teamcell .name{color:var(--muted);font-size:12px}
.footer{color:var(--muted);font-size:12px;margin-top:16px;text-align:left;line-height:1.5}
.footer a{color:var(--accent);text-decoration:underline;font-weight:600}
@media (max-width:700px){
  thead th{font-size:12px}
  tbody td{padding:8px}
  .teamcell .name{display:none}
  table{min-width:980px}
}
</style>
</head>
<body>
<div class="wrapper">
  <h1>NHL Standings — %%DATE%%</h1>
  <div class="nav">
    <a href="/">Predictions</a>
    <a href="/standings" class="active">Standings</a>
  </div>

  %%CONFSECTIONS%%

  <div class="footer">
    <div>Last updated at: <strong>%%UPDATED%%</strong></div>
    <div>Generated by <a href="https://x.com/reagankingisles">@ReaganKingIsles</a>. Logos © NHL/teams; loaded from NHL CDN.</div>
  </div>
</div>

<script>
// Per-table sorting
document.querySelectorAll('table.standings').forEach(function(tbl){
  let dir = 1, lastKey = '';
  function colIndexByKey(key){
    const ths = Array.from(tbl.tHead.rows[0].cells);
    return ths.findIndex(th => th.dataset.key === key) + 1;
  }
  function sortBy(key){
    const idx = colIndexByKey(key);
    if (!idx) return;
    const rows = Array.from(tbl.tBodies[0].rows);
    if (lastKey === key) dir = -dir; else { dir = 1; lastKey = key; }
    rows.sort((a,b)=>{
      if (key === 'team'){
        const av = a.querySelector('.abbr').textContent, bv = b.querySelector('.abbr').textContent;
        return dir * av.localeCompare(bv);
      }
      const aVal = parseFloat(a.querySelector(`td:nth-child(${idx})`)?.dataset.val ?? 'NaN');
      const bVal = parseFloat(b.querySelector(`td:nth-child(${idx})`)?.dataset.val ?? 'NaN');
      return dir * ((aVal>bVal)-(aVal<bVal));
    });
    rows.forEach(r=>tbl.tBodies[0].appendChild(r));
  }
  tbl.tHead.querySelectorAll('th').forEach(th=>{
    if (!th.classList.contains('nosort')){
      th.addEventListener('click', ()=> sortBy(th.dataset.key));
    }
  });
});
// Logo fallbacks
document.querySelectorAll('img.logo').forEach(img=>{
  let alts=[]; try{ alts=JSON.parse(img.dataset.alts||'[]'); }catch(e){}
  let i=0; img.onerror=()=>{ if(i<alts.length){ img.src=alts[i++]; } else { img.style.visibility='hidden'; } };
});
</script>
</body></html>
"""
    html = (html
            .replace("%%DATE%%", report_date)
            .replace("%%CONFSECTIONS%%", "\n".join(conf_sections))
            .replace("%%UPDATED%%", updated_time))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

# =========================
# Backtest + season-to-date accuracy (quick version)
# =========================
def find_season_start(yesterday_local: datetime.date) -> datetime.date:
    start_scan = datetime(yesterday_local.year, 9, 1, tzinfo=LOCAL_TZ).date()
    d = start_scan
    while d <= yesterday_local:
        ds = d.strftime("%Y-%m-%d")
        data = safe_get_json(API_SCORE_DATE.format(date=ds))
        games = (data or {}).get("games") or []
        has_regular = any(normalize_game_type((g or {}).get("gameType")) == "R" for g in games if isinstance(g, dict))
        if has_regular:
            return d
        d += timedelta(days=1)
    return max(yesterday_local - timedelta(days=7), start_scan)

def compute_season_record_to_date() -> Tuple[int, int, float]:
    """
    Walk day-by-day from season start to yesterday.
    For each date:
      - read the official score feed (/v1/score/{date})
      - keep only FINAL regular-season games
      - compute model pick using Elo built through (date - 1)
    Returns (correct, total, pct).
    """
    today_local = datetime.now(tz=LOCAL_TZ).date()
    yesterday = today_local - timedelta(days=1)
    if yesterday < datetime(today_local.year, 9, 1, tzinfo=LOCAL_TZ).date():
        return (0, 0, 0.0)

    # Find first date with regular-season games
    season_start = find_season_start(yesterday)

    # Small memo so we don’t rebuild Elo repeatedly for the same date
    elo_snapshots: Dict[datetime.date, dict] = {}

    def elo_through(day_minus_one: datetime.date) -> dict:
        if day_minus_one not in elo_snapshots:
            elo_snapshots[day_minus_one] = get_or_build_elo_cached(day_minus_one)
        return elo_snapshots[day_minus_one]

    def final_regular_games(ds: str) -> List[dict]:
        data = safe_get_json(API_SCORE_DATE.format(date=ds)) or {}
        games = data.get("games") or []
        finals = []
        for g in games:
            # Must be regular season AND final
            if normalize_game_type(g.get("gameType")) != "R":
                continue
            st = (g.get("gameState") or "").upper()
            if st not in {"FINAL", "OFF", "COMPLETE", "COMPLETED", "GAME_OVER"}:
                continue
            home = g.get("homeTeam", {}) or {}
            away = g.get("awayTeam", {}) or {}
            finals.append({
                "home_key": canonical_team_key(home),
                "away_key": canonical_team_key(away),
                "home_score": int(home.get("score", 0)),
                "away_score": int(away.get("score", 0)),
            })
        return finals

    correct = total = 0
    d = season_start
    while d <= yesterday:
        ds = d.strftime("%Y-%m-%d")
        finals = final_regular_games(ds)
        if finals:
            # Elo snapshot as of the morning of d == built through d-1
            snap_date = d - timedelta(days=1)
            state = elo_through(snap_date)
            elo_map = state.get("elo", {}) or {}

            for g in finals:
                helo = float(elo_map.get(g["home_key"], ELO_INIT))
                aelo = float(elo_map.get(g["away_key"], ELO_INIT))
                hxg, axg = expected_goals(helo, aelo)
                p_home, _ = win_probs_from_skellam(hxg, axg)

                model_pick_home = (p_home >= 0.5)
                actual_home_win = (g["home_score"] > g["away_score"])

                total += 1
                if model_pick_home == actual_home_win:
                    correct += 1

        d += timedelta(days=1)

    pct = (correct / total * 100.0) if total else 0.0
    return correct, total, pct

# =========================
# CLI / Main
# =========================
def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="NHL predictor + backtester + standings with PO%")
    ap.add_argument("--backtest-season", dest="backtest_season", action="store_true",
                    help="Run a walk-forward backtest for this regular season (up to yesterday)")
    return ap.parse_args()

def main():
    args = parse_args()
    if args.backtest_season:
        print("Backtester entry unchanged—use your earlier backtest function if desired.")
        return

    # Build Elo up to yesterday (cached)
    today_local = datetime.now(tz=LOCAL_TZ).date()
    end_date = today_local - timedelta(days=1)
    state = get_or_build_elo_cached(end_date)

    # Predictions page
    records = get_team_records()
    correct, total, pct = compute_season_record_to_date()
    season_line = f"Season to date: {correct}-{max(total - correct, 0)} ({pct:.1f}%)" if total else "Season to date: —"
    preds = predict_day(state, today_local, records)
    write_csv(preds, PREDICTIONS_CSV)
    write_html(preds, PREDICTIONS_HTML, report_date=today_local.isoformat(), season_line=season_line)

    # Standings with PO%
    rows = get_standings_now_rows()
    po = simulate_playoff_probs(state, rows, today_local, sims=SIMS, ot_rate=OT_RATE)
    attach_po_to_rows(rows, po)
    write_html_standings(rows, STANDINGS_HTML, report_date=today_local.isoformat())

    # Console summary
    print(f"Predictions for {today_local.isoformat()}:")
    if preds:
        for p in preds:
            print(f"{p['away_key']} @ {p['home_key']} {p['local_time']}: "
                  f"P(H)={p['p_home_win']:.3f}, P(A)={p['p_away_win']:.3f}, "
                  f"ML(H)={p['ml_home']:+d}, ML(A)={p['ml_away']:+d}, "
                  f"xG {p['pred_xg_away']:.2f}-{p['pred_xg_home']:.2f}")
    else:
        print("(no games found)")
    print(f"\nSaved to {PREDICTIONS_CSV}, {PREDICTIONS_HTML}, and {STANDINGS_HTML}")

if __name__ == "__main__":
    main()
