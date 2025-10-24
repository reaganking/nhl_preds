#!/usr/bin/env python3
"""
Daily NHL predictor + season backtester using api-web.nhle.com

Run locally:
  python app.py
  python app.py --backtest-season
"""

import csv
import json
import math
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, List, Any, Optional

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
LOCAL_TZ = ZoneInfo("America/Chicago")
CENTRAL_TZ = ZoneInfo("America/Chicago")

PREDICTIONS_CSV = "predictions_today.csv"
PREDICTIONS_HTML = "predictions_today.html"
ELO_DUMP_CSV = "elo_dump.csv"
BACKTEST_CSV = "backtest_results.csv"
BACKTEST_SUMMARY_JSON = "backtest_summary.json"

API_BASE = "https://api-web.nhle.com"
API_SCHEDULE_DATE = API_BASE + "/v1/schedule/{date}"   # YYYY-MM-DD
API_SCORE_DATE    = API_BASE + "/v1/score/{date}"      # YYYY-MM-DD
API_STANDINGS_NOW = API_BASE + "/v1/standings/now"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# =========================
# Networking
# =========================
def session_with_retries():
    s = requests.Session()
    retry = Retry(total=5, connect=5, read=5, backoff_factor=0.4,
                  status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "nhl-predictor/3.0"})
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
    """Return { 'TOR': 'W-L-OT', ... }."""
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
            if ot is None:
                ot = (it.get("overtimeLosses") if isinstance(it.get("overtimeLosses"), int) else None)
            if ot is None:
                ot = (it.get("otl") if isinstance(it.get("otl"), int) else None)
            if ot is None:
                ot = (it.get("record", {}) or {}).get("ot") or (it.get("record", {}) or {}).get("overtimeLosses") or 0
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

def modal_scoreline(home_xg: float, away_xg: float, max_goals: int = 10) -> Tuple[int, int]:
    best_h, best_a, best_p = 0, 0, -1.0
    ph = [math.exp(-home_xg) * (home_xg ** h) / math.factorial(h) for h in range(max_goals + 1)]
    pa = [math.exp(-away_xg) * (away_xg ** a) / math.factorial(a) for a in range(max_goals + 1)]
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = ph[h] * pa[a]
            if p > best_p:
                best_p = p
                best_h, best_a = h, a
    return best_h, best_a

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
    home_name: str
    away_name: str
    game_type: str
    start_local_dt: Optional[datetime]
    start_utc_str: str

def get_schedule_for_local_date(local_date: datetime.date) -> List[GameSched]:
    ds = local_date.strftime("%Y-%m-%d")
    data = safe_get_json(API_SCHEDULE_DATE.format(date=ds))
    candidates = _iter_schedule_candidates(data)
    logging.info(f"Schedule fetched: {len(candidates)} candidates for site date {ds}")

    out: List[GameSched] = []
    for g in candidates:
        start_utc = g.get("startTimeUTC") or g.get("startTimeUTCISO") or ""
        start_local_dt = _utc_to_local_dt(start_utc, LOCAL_TZ)
        if not start_local_dt or start_local_dt.date() != local_date:
            continue
        home, away = g.get("homeTeam", {}), g.get("awayTeam", {})
        gt = normalize_game_type(g.get("gameType"))
        start_utc_clean = start_utc[:-1] if isinstance(start_utc, str) and start_utc.endswith("Z") else (start_utc or "")
        out.append(GameSched(
            game_id=g.get("id") or g.get("gamePk") or g.get("gameId"),
            home_key=canonical_team_key(home),
            away_key=canonical_team_key(away),
            home_name=full_team_name(home),
            away_name=full_team_name(away),
            game_type=gt,
            start_local_dt=start_local_dt,
            start_utc_str=start_utc_clean
        ))
    logging.info(f"Schedule kept: {len(out)} games on local date {local_date.isoformat()}")
    return out

@dataclass
class GameFinal:
    date: datetime
    home_key: str
    away_key: str
    home_score: int
    away_score: int
    game_type: str

def get_finals_for_date(ds: str) -> List[GameFinal]:
    data = safe_get_json(API_SCORE_DATE.format(date=ds))
    games = (data or {}).get("games") or []
    finals: List[GameFinal] = []
    for g in games:
        if not isinstance(g, dict):
            continue
        state = (g.get("gameState") or "").upper()
        if state not in {"FINAL", "OFF", "COMPLETE", "COMPLETED", "GAME_OVER"}:
            continue
        home, away = g.get("homeTeam", {}), g.get("awayTeam", {})
        gt = normalize_game_type(g.get("gameType"))
        finals.append(GameFinal(
            date=datetime.strptime(ds, "%Y-%m-%d"),
            home_key=canonical_team_key(home),
            away_key=canonical_team_key(away),
            home_score=int(home.get("score", 0)),
            away_score=int(away.get("score", 0)),
            game_type=gt
        ))
    return finals

# =========================
# History build
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
        finals = get_finals_for_date(ds)
        for g in finals:
            if include_types and g.game_type and g.game_type not in include_types:
                continue
            ha, aa = g.home_key, g.away_key
            if ha == "UNKNOWN" or aa == "UNKNOWN":
                continue
            update_elo(state, ha, aa, g.home_score, g.away_score, g.date, ref_today_dt)
        total += len(finals)
        cur += timedelta(days=1)
    logging.info(
        f"Elo built from {start_date} to {end_date}. "
        f"{len(state.get('elo', {}))} teams rated. {total} final games ingested."
    )
    log_elo_summary(state)

# =========================
# Prediction + HTML
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
        mh, ma = modal_scoreline(hxg, axg)

        ml_home = american_moneyline(p_home)
        ml_away = american_moneyline(p_away)

        preds.append({
            "date": local_date.isoformat(),
            "gameId": g.game_id,
            "away_key": away_key,
            "home_key": home_key,
            "away_name": g.away_name,
            "home_name": g.home_name,
            "pred_xg_home": round(hxg, 2),
            "pred_xg_away": round(axg, 2),
            "pred_mode_home": mh,
            "pred_mode_away": ma,
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
            "local_time": fmt_local_time(g.start_local_dt),
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

def write_html(preds: List[Dict[str, Any]], path: str, report_date: str, season_line: Optional[str] = None):
    """
    Pretty HTML table with logos, retrying multiple logo URLs, % probs (one decimal),
    local time (browser-local via JS), team records. AWAY metrics first to align with top team.
    Elo column removed. Responsive: tablet tweaks + stacked cards on mobile.
    """
    updated_time = datetime.now(tz=CENTRAL_TZ).strftime("%a, %b %d, %Y — %I:%M %p %Z")

    # Empty-state page
    if not preds:
        html = """
<!doctype html>
<html><head><meta charset="utf-8"><title>NHL Predictions %%DATE%%</title>
<style>
:root{--bg:#0b1020;--panel:#121933;--panel2:#0e1630;--txt:#e8ecff;--muted:#9fb1ff;--border:#1e2748;--border-strong:#29345e;--accent:#7aa2ff;}
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--txt);margin:0}
.wrapper{max-width:1100px;margin:32px auto;padding:0 16px}
h1{font-weight:800;margin:0 0 16px}
.seasonline{color:var(--muted);margin:-6px 0 16px;font-weight:600}
.card{background:#121933;border:1px solid #1e2748;border-radius:16px;padding:16px;box-shadow:0 10px 30px rgba(0,0,0,.25)}
.empty{opacity:.85}
.footer{color:var(--muted);font-size:12px;margin-top:16px;text-align:left;opacity:1;line-height:1.5}
.footer a{color:var(--accent);text-decoration:underline;font-weight:600}
.footer a:hover,.footer a:focus{text-decoration:none;filter:brightness(1.15)}
</style></head>
<body><div class="wrapper">
  <h1>NHL Predictions — %%DATE%%</h1>
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

    # Row HTML builder (away-first alignment) — Elo removed from output
    def row_html(p):
        ph = f"{p['p_home_win'] * 100:.1f}%"
        pa = f"{p['p_away_win'] * 100:.1f}%"
        alts_home = p.get("home_logo_alts", "[]")
        alts_away = p.get("away_logo_alts", "[]")
        home_name = p['home_name'] + (f" ({p['home_record']})" if p.get('home_record') else "")
        away_name = p['away_name'] + (f" ({p['away_record']})" if p.get('away_record') else "")
        time_str = p.get("local_time", "")
        return f"""
<tr>
  <td class="teams">
    <div class="team">
      <img class="logo" src="{p['away_logo']}" data-alts='{alts_away}' alt="{p['away_key']}" loading="lazy"/>
      <div class="meta">
        <div class="abbr">{p['away_key']}</div>
        <div class="name">{away_name}</div>
      </div>
    </div>
    <div class="vs">at <span class="time" data-utc="{p.get('utc_time','')}">{time_str}</span></div>
    <div class="team home">
      <img class="logo" src="{p['home_logo']}" data-alts='{alts_home}' alt="{p['home_key']}" loading="lazy"/>
      <div class="meta">
        <div class="abbr">{p['home_key']}</div>
        <div class="name">{home_name}</div>
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

    # Main HTML page (no f-string; safe for JS/CSS braces)
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
.team .meta .name{font-size:12px;color:var(--muted)}
.vs{margin:8px 6px;color:var(--muted);font-size:12px}
.vs .time{font-weight:700;color:#fff;margin-left:6px}
.prob,.ml,.xg{white-space:nowrap}
b{color:#fff}
.note{margin-top:10px;color:var(--muted);font-size:12px}
.footer{color:var(--muted);font-size:12px;margin-top:16px;text-align:left;opacity:1;line-height:1.5}
.footer a{color:var(--accent);text-decoration:underline;font-weight:600}
.footer a:hover,.footer a:focus{text-decoration:none;filter:brightness(1.15)}

/* === Responsive tweaks === */
@media (max-width:900px){
  .wrapper{max-width:960px}
  thead th{font-size:13px}
  tbody td{padding:10px 8px}
  .team img.logo{width:30px;height:30px}
}

@media (max-width:640px){
  .wrapper{max-width:100%;padding:0 12px}
  .table-card{padding:10px}
  table,thead,tbody,th,td,tr{display:block}
  thead{position:absolute;left:-9999px;top:-9999px}
  tbody tr{border-radius:12px;margin:10px 0;padding:6px 6px 10px}
  td.teams{padding:10px 8px 8px;border-bottom:1px solid var(--border);min-width:0}
  .team{gap:8px}
  .team img.logo{width:28px;height:28px}
  .team .meta .name{font-size:11px}
  .vs{margin:6px 0 4px;font-size:11px}
  td.prob,td.ml,td.xg{
    display:flex;align-items:center;justify-content:space-between;gap:12px;padding:8px 8px;white-space:normal
  }
  td.prob::before,td.ml::before,td.xg::before{
    content:attr(data-label);color:var(--muted);font-weight:700;letter-spacing:.2px
  }
  td.prob b,td.ml b,td.xg b{font-weight:800}
}
</style>
</head>
<body>
  <div class="wrapper">
    <h1>NHL Predictions — %%DATE%%</h1>
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
        <tbody>
          %%ROWS%%
        </tbody>
      </table>
      <div class="note">xG = expected goals.</div>
    </div>
    <div class="footer">
      <div>Last updated at: <strong>%%UPDATED%%</strong></div>
      <div>Generated by <a href="https://x.com/reagankingisles">@ReaganKingIsles</a>. Logos © NHL/teams; loaded from NHL CDN.</div>
    </div>
  </div>
<script>
// Try multiple logo URLs in order. If all fail, hide the image.
document.querySelectorAll('img.logo').forEach(function(img){
  let alts = [];
  try { alts = JSON.parse(img.dataset.alts || '[]'); } catch(e) { alts = []; }
  let i = 0;
  img.onerror = function(){
    if (i < alts.length) { img.src = alts[i++]; }
    else { img.onerror = null; img.style.visibility = 'hidden'; }
  };
});

// Convert all game times to the viewer's local time
document.querySelectorAll('.time[data-utc]').forEach(function(el){
  const utc = el.getAttribute('data-utc');
  if (utc) {
    const d = new Date(utc + 'Z');
    el.textContent = d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  }
});

// Daily auto-refresh at 03:10 local
(function(){
  function msUntilNext(h, m){
    const now = new Date();
    const next = new Date(now); next.setHours(h, m, 0, 0);
    if (next <= now) next.setDate(next.getDate() + 1);
    return next - now;
  }
  setTimeout(function(){
    const u = new URL(location.href); u.searchParams.set('r', Date.now().toString());
    location.replace(u.toString());
  }, msUntilNext(3, 10));
  setTimeout(function(){ location.reload(); }, 24*60*60*1000);
})();
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
# Backtest + season-to-date
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

def backtest_this_season():
    today_local = datetime.now(tz=LOCAL_TZ).date()
    yesterday = today_local - timedelta(days=1)

    season_start = find_season_start(yesterday)
    logging.info(f"Backtest season start detected: {season_start}")

    warm_start = datetime(season_start.year - 1, 10, 1, tzinfo=LOCAL_TZ).date()
    state = {"elo": {}}
    if warm_start <= (season_start - timedelta(days=1)):
        build_elo_from_history(state, warm_start, season_start - timedelta(days=1), include_types=("R","P"))

    rows = []
    d = season_start
    while d <= yesterday:
        sched = [g for g in get_schedule_for_local_date(d) if (g.game_type or "R") == "R"]
        for g in sched:
            helo = state.get("elo", {}).get(g.home_key, ELO_INIT)
            aelo = state.get("elo", {}).get(g.away_key, ELO_INIT)
            hxg, axg = expected_goals(helo, aelo)
            p_home, p_away = win_probs_from_skellam(hxg, axg)
            mh, ma = modal_scoreline(hxg, axg)
            rows.append({
                "date": d.isoformat(),
                "gameId": g.game_id,
                "home_key": g.home_key,
                "away_key": g.away_key,
                "home_name": g.home_name,
                "away_name": g.away_name,
                "pred_p_home": p_home,
                "pred_p_away": p_away,
                "pred_xg_home": hxg,
                "pred_xg_away": axg,
                "pred_mode_home": mh,
                "pred_mode_away": ma,
            })
        finals = get_finals_for_date(d.strftime("%Y-%m-%d"))
        for gf in finals:
            if (gf.game_type or "") != "R":
                continue
            if gf.home_key == "UNKNOWN" or gf.away_key == "UNKNOWN":
                continue
            update_elo(state, gf.home_key, gf.away_key, gf.home_score, gf.away_score,
                       gf.date, datetime(d.year, d.month, d.day, tzinfo=LOCAL_TZ))
        d += timedelta(days=1)

    actuals: Dict[Tuple[str,str,str], Dict[str, Any]] = {}
    d = season_start
    while d <= yesterday:
        for gf in get_finals_for_date(d.strftime("%Y-%m-%d")):
            if (gf.game_type or "") != "R":
                continue
            k = (d.isoformat(), gf.home_key, gf.away_key)
            actuals[k] = {"home_score": gf.home_score, "away_score": gf.away_score, "home_win": 1 if gf.home_score > gf.away_score else 0}
        d += timedelta(days=1)

    results = []
    brier_sum = logloss_sum = 0.0
    acc_sum = 0
    mae_home_sum = mae_away_sum = mae_total_sum = 0.0
    rmse_home_sq_sum = rmse_away_sq_sum = rmse_total_sq_sum = 0.0
    n = 0

    for r in rows:
        key = (r["date"], r["home_key"], r["away_key"])
        a = actuals.get(key)
        if not a:
            continue
        p_home = max(1e-6, min(1-1e-6, r["pred_p_home"]))
        home_win = a["home_win"]
        brier = (p_home - home_win) ** 2
        logloss = - (home_win * math.log(p_home) + (1 - home_win) * math.log(1 - p_home))
        acc = 1 if ((p_home >= 0.5) == (home_win == 1)) else 0

        pred_home = r["pred_xg_home"]; pred_away = r["pred_xg_away"]
        act_home = a["home_score"];    act_away = a["away_score"]

        err_home = pred_home - act_home
        err_away = pred_away - act_away
        err_total = (pred_home + pred_away) - (act_home + act_away)

        mae_home_sum += abs(err_home); mae_away_sum += abs(err_away); mae_total_sum += abs(err_total)
        rmse_home_sq_sum += err_home**2; rmse_away_sq_sum += err_away**2; rmse_total_sq_sum += err_total**2

        results.append({
            **r,
            "home_score": act_home, "away_score": act_away, "home_win": home_win,
            "picked_correct": acc, "brier": brier, "logloss": logloss,
            "abs_err_home_goals": abs(err_home), "abs_err_away_goals": abs(err_away), "abs_err_total_goals": abs(err_total),
        })

        brier_sum += brier; logloss_sum += logloss; acc_sum += acc; n += 1

    if n == 0:
        logging.warning("Backtest found no regular-season games to evaluate.")
        write_csv([], BACKTEST_CSV)
        with open(BACKTEST_SUMMARY_JSON, "w") as f:
            json.dump({"games_evaluated": 0}, f, indent=2)
        return

    brier = brier_sum / n; logloss = logloss_sum / n; accuracy = acc_sum / n
    mae_home = mae_home_sum / n; mae_away = mae_away_sum / n; mae_total = mae_total_sum / n
    rmse_home = math.sqrt(rmse_home_sq_sum / n); rmse_away = math.sqrt(rmse_away_sq_sum / n); rmse_total = math.sqrt(rmse_total_sq_sum / n)

    for row in results:
        for k in ("pred_p_home", "pred_p_away", "pred_xg_home", "pred_xg_away", "brier", "logloss",
                  "abs_err_home_goals", "abs_err_away_goals", "abs_err_total_goals"):
            row[k] = round(row[k], 4)
    write_csv(results, BACKTEST_CSV)
    summary = {
        "games_evaluated": n, "accuracy": round(accuracy, 4), "brier": round(brier, 5), "logloss": round(logloss, 5),
        "mae_home_goals": round(mae_home, 4), "mae_away_goals": round(mae_away, 4), "mae_total_goals": round(mae_total, 4),
        "rmse_home_goals": round(rmse_home, 4), "rmse_away_goals": round(rmse_away, 4), "rmse_total_goals": round(rmse_total, 4),
    }
    with open(BACKTEST_SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Backtest complete on {n} games. Acc={summary['accuracy']}, Brier={summary['brier']}, LogLoss={summary['logloss']}")

def compute_season_record_to_date() -> Tuple[int, int, float]:
    today_local = datetime.now(tz=LOCAL_TZ).date()
    yesterday = today_local - timedelta(days=1)
    if yesterday < datetime(today_local.year, 9, 1, tzinfo=LOCAL_TZ).date():
        return (0, 0, 0.0)

    season_start = find_season_start(yesterday)
    warm_start = datetime(season_start.year - 1, 10, 1, tzinfo=LOCAL_TZ).date()
    state = {"elo": {}}
    if warm_start <= (season_start - timedelta(days=1)):
        build_elo_from_history(state, warm_start, season_start - timedelta(days=1), include_types=("R","P"))

    correct = total = 0
    d = season_start
    while d <= yesterday:
        sched = [g for g in get_schedule_for_local_date(d) if (g.game_type or "R") == "R"]
        finals = get_finals_for_date(d.strftime("%Y-%m-%d"))
        for g in sched:
            helo = state.get("elo", {}).get(g.home_key, ELO_INIT)
            aelo = state.get("elo", {}).get(g.away_key, ELO_INIT)
            hxg, axg = expected_goals(helo, aelo)
            p_home, _ = win_probs_from_skellam(hxg, axg)
            for gf in finals:
                if (gf.game_type or "") != "R":
                    continue
                if gf.home_key == g.home_key and gf.away_key == g.away_key:
                    total += 1
                    home_win = gf.home_score > gf.away_score
                    model_pick_home = p_home >= 0.5
                    if model_pick_home == home_win:
                        correct += 1
                    break
        for gf in finals:
            if (gf.game_type or "") != "R":
                continue
            if gf.home_key == "UNKNOWN" or gf.away_key == "UNKNOWN":
                continue
            update_elo(state, gf.home_key, gf.away_key, gf.home_score, gf.away_score,
                       gf.date, datetime(d.year, d.month, d.day, tzinfo=LOCAL_TZ))
        d += timedelta(days=1)

    pct = (correct / total * 100.0) if total else 0.0
    return correct, total, pct

# =========================
# CLI / Main
# =========================
def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="NHL predictor + backtester")
    ap.add_argument("--backtest-season", dest="backtest_season", action="store_true",
                    help="Run a walk-forward backtest for this regular season (up to yesterday)")
    return ap.parse_args()

def main():
    args = parse_args()
    if args.backtest_season:
        backtest_this_season()
        return

    # Daily prediction
    today_local = datetime.now(tz=LOCAL_TZ).date()
    start_date = datetime(today_local.year - 1, 10, 1, tzinfo=LOCAL_TZ).date()
    end_date = today_local - timedelta(days=1)
    state = {"elo": {}}
    build_elo_from_history(state, start_date, end_date, include_types=("R","P"))

    records = get_team_records()
    correct, total, pct = compute_season_record_to_date()
    season_line = f"Season to date: {correct}-{max(total - correct, 0)} ({pct:.1f}%)" if total else "Season to date: —"

    preds = predict_day(state, today_local, records)
    write_csv(preds, PREDICTIONS_CSV)
    write_html(preds, PREDICTIONS_HTML, report_date=today_local.isoformat(), season_line=season_line)

    print(f"Predictions for {today_local.isoformat()}:")
    if preds:
        for p in preds:
            print(f"{p['away_key']} @ {p['home_key']} {p['local_time']}: "
                  f"P(H)={p['p_home_win']:.3f}, P(A)={p['p_away_win']:.3f}, "
                  f"ML(H)={p['ml_home']:+d}, ML(A)={p['ml_away']:+d}, "
                  f"xG {p['pred_xg_away']:.2f}-{p['pred_xg_home']:.2f}, "
                  f"Modal {p['pred_mode_away']}-{p['pred_mode_home']}")
    else:
        print("(no games found)")
    print(f"\nSaved to {PREDICTIONS_CSV} and {PREDICTIONS_HTML}")

if __name__ == "__main__":
    main()
