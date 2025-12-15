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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)



# =========================
# Config
# =========================
ELO_INIT = 1500.0
K_BASE = 18.0
HOME_ADV_ELO = 45.0
PLAYOFF_HOME_ADV_ELO = 55.0
OT_SO_PENALTY = 6.0
REG_WIN_BONUS = 4.0

REG_WIN_BONUS_DECAY_DAYS = 28  # ~1 month

LOGO_DIR = Path("static/nhl-logos")
TEAM_COLOR_FILE = Path("data/nhl_team_colors.csv")

LOCAL_TZ = ZoneInfo("America/Chicago")
CENTRAL_TZ = ZoneInfo("America/Chicago")
UTC = timezone.utc

SESSION = requests.Session()
ADAPTER = HTTPAdapter(
    max_retries=Retry(
        total=5,
        backoff_factor=0.3,
        status_forcelist=(500, 502, 503, 504),
    )
)
SESSION.mount("https://", ADAPTER)
SESSION.mount("http://", ADAPTER)

# =========================
# Utility functions
# =========================

def _today_local() -> datetime.date:
    return datetime.now(tz=LOCAL_TZ).date()


def _ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _cache_path(name: str) -> Path:
    return Path("/tmp") / name


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any):
    _ensure_dir(path)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f)
    tmp.replace(path)


# =========================
# NHL API helpers
# =========================

API_BASE = "https://api-web.nhle.com"

# (all your existing NHL API helpers, data classes, etc. are unchanged here)

# ...  (omitting unchanged helpers for brevity; keep everything from your current app.py) ...


# =========================
# Elo + caching
# =========================

def build_elo_from_history(
    state: Dict[str, Any],
    start_date: datetime.date,
    end_date: datetime.date,
    include_types=("R", "P"),
):
    """
    Given an existing state dict with 'elo' table (may be empty) and optional
    other fields, walk games from start_date to end_date inclusive and update
    Elo in-place.
    """
    # (this function is unchanged from your current version)
    # ...


def get_or_build_elo_cached(end_date):
    """
    Incrementally maintain Elo from Oct 1 of the previous season up to end_date.

    Cache format:
      {
        "elo": {...},
        "last_date": "YYYY-MM-DD"
      }

    We only build new Elo entries for the gap (last_date+1 → end_date).
    Cache is stored in /tmp/elo_state.json.
    """
    key = "elo_state.json"
    p = _cache_path(key)

    cached = _read_json(p)

    # ---- LOAD EXISTING CACHE OR INITIALIZE NEW ONE ----
    if cached and isinstance(cached, dict) and "elo" in cached:
        state = cached
        last_date_str = cached.get("last_date")
        try:
            last_date = (
                datetime.fromisoformat(last_date_str).date()
                if last_date_str
                else None
            )
        except Exception:
            last_date = None
    else:
        state = {"elo": {}, "last_date": None}
        last_date = None

    # ---- IF ALREADY UP TO DATE, RETURN ----
    if last_date and last_date >= end_date:
        return state

    # ---- DETERMINE START DATE FOR INCREMENTAL UPDATE ----
    if last_date is None:
        # First-ever build: start from Oct 1 of previous season
        start_date = datetime(end_date.year - 1, 10, 1, tzinfo=LOCAL_TZ).date()
    else:
        # Incremental build: continue from day after last_date
        start_date = last_date + timedelta(days=1)

    # Safety: do nothing if somehow start_date > end_date
    if start_date > end_date:
        return state

    # ---- PERFORM INCREMENTAL BUILD ----
    logger.info(f"Elo incremental build from {start_date} to {end_date}")

    build_elo_from_history(
        state,
        start_date,
        end_date,
        include_types=("R", "P"),
    )

    # ---- SAVE UPDATED STATE ----
    state["last_date"] = end_date.isoformat()
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
        # ... rest of prediction logic unchanged ...


# (everything that follows in app.py – season record, standings, playoff sims,
# write_html, write_html_standings, etc. – stays exactly as in your current file)
# Make sure you keep all remaining functions from your existing app.py below.
