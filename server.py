#!/usr/bin/env python3
from flask import Flask, Response, request, abort, jsonify
from flask_compress import Compress
from datetime import datetime, timedelta
from threading import Thread
from zoneinfo import ZoneInfo
import os

from app import (
    build_elo_from_history,
    get_team_records,
    predict_day,
    write_html,
    compute_season_record_to_date,
    get_standings_now_rows,
    simulate_playoff_probs,
    attach_po_to_rows,
    write_html_standings,
    LOCAL_TZ,
    CENTRAL_TZ,
    get_or_build_elo_cached,          # NEW: cached Elo
)

app = Flask(__name__)
Compress(app)  # gzip/br

# In-memory caches
_cached = { "date": None, "html": None, "games": 0, "updated_ct": None }
_cached_standings = { "date": None, "html": None, "stamp": None }

REFRESH_TOKEN = os.environ.get("REFRESH_TOKEN", "")
PRED_HTML_PATH = "/tmp/predictions.html"
STAND_HTML_PATH = "/tmp/standings.html"

# Sim settings: fast for on-demand, fuller for cron warm
FAST_SIMS = int(os.environ.get("FAST_SIMS", "120"))
FULL_SIMS = int(os.environ.get("SIMS", "300"))
OT_RATE = float(os.environ.get("OT_RATE", "0.23"))

def _today_local():
    return datetime.now(tz=LOCAL_TZ).date()

def _generate_predictions_html_for(date_obj):
    # Elo to yesterday (cached)
    end_date = date_obj - timedelta(days=1)
    state = get_or_build_elo_cached(end_date)

    # Records + season line
    records = get_team_records()
    correct, total, pct = compute_season_record_to_date()
    season_line = f"Season to date: {correct}-{max(total - correct, 0)} ({pct:.1f}%)" if total else "Season to date: —"

    preds = predict_day(state, date_obj, records)
    write_html(preds, PRED_HTML_PATH, report_date=date_obj.isoformat(), season_line=season_line)

    with open(PRED_HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    _cached.update({
        "date": date_obj,
        "html": html,
        "games": len(preds),
        "updated_ct": datetime.now(tz=CENTRAL_TZ).strftime("%a, %b %d, %Y — %I:%M %p %Z"),
    })
    return html

def _generate_standings_html_for(date_obj, sims: int):
    # Elo to yesterday (cached)
    end_date = date_obj - timedelta(days=1)
    state = get_or_build_elo_cached(end_date)

    rows = get_standings_now_rows()
    po = simulate_playoff_probs(state, rows, date_obj, sims=sims, ot_rate=OT_RATE)
    attach_po_to_rows(rows, po)
    write_html_standings(rows, STAND_HTML_PATH, report_date=date_obj.isoformat())

    with open(STAND_HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    _cached_standings.update({
        "date": date_obj,
        "html": html,
        "stamp": datetime.now(tz=CENTRAL_TZ),
    })
    return html

def _warm_now():
    today = _today_local()
    _generate_predictions_html_for(today)
    _generate_standings_html_for(today, sims=FULL_SIMS)

@app.route("/")
def index():
    today = _today_local()
    if not _cached["html"]:
        try:
            with open(PRED_HTML_PATH, "r", encoding="utf-8") as f:
                _cached["html"] = f.read()
                _cached["date"] = today
        except Exception:
            _generate_predictions_html_for(today)
    resp = Response(_cached["html"], mimetype="text/html")
    resp.headers["Cache-Control"] = "public, max-age=300"
    return resp

@app.route("/standings")
def standings():
    """Serve cached standings immediately. If cache is old (>2h), refresh in background."""
    now = datetime.now(tz=CENTRAL_TZ)
    stale_minutes = 120
    if _cached_standings["html"]:
        # Serve instantly; maybe refresh
        if not _cached_standings["stamp"] or (now - _cached_standings["stamp"]).total_seconds() > stale_minutes * 60:
            Thread(target=lambda: _generate_standings_html_for(_today_local(), sims=FAST_SIMS), daemon=True).start()
        resp = Response(_cached_standings["html"], mimetype="text/html")
        resp.headers["Cache-Control"] = "public, max-age=900"
        return resp

    # No cache yet: try disk, else compute fast
    try:
        with open(STAND_HTML_PATH, "r", encoding="utf-8") as f:
            _cached_standings["html"] = f.read()
            _cached_standings["date"] = _today_local()
            _cached_standings["stamp"] = now
            resp = Response(_cached_standings["html"], mimetype="text/html")
            resp.headers["Cache-Control"] = "public, max-age=900"
            return resp
    except Exception:
        # Compute a FAST version synchronously, then return
        html = _generate_standings_html_for(_today_local(), sims=FAST_SIMS)
        resp = Response(html, mimetype="text/html")
        resp.headers["Cache-Control"] = "public, max-age=900"
        return resp

@app.route("/refresh")
def refresh():
    token = request.args.get("token", "")
    if REFRESH_TOKEN and token != REFRESH_TOKEN:
        abort(403)
    today = _today_local()
    _generate_predictions_html_for(today)
    _generate_standings_html_for(today, sims=FULL_SIMS)
    return Response("OK\n", mimetype="text/plain")

@app.route("/warm")
def warm():
    token = request.args.get("token", "")
    if REFRESH_TOKEN and token != REFRESH_TOKEN:
        abort(403)
    Thread(target=_warm_now, daemon=True).start()
    return ("", 204)

@app.route("/status.json")
def status():
    d = _cached["date"].isoformat() if _cached["date"] else None
    sd = _cached_standings["date"].isoformat() if _cached_standings["date"] else None
    return jsonify({
        "predictions": {"date": d, "games": _cached["games"], "updated_ct": _cached["updated_ct"], "ok": bool(_cached["html"])},
        "standings": {"date": sd, "ok": bool(_cached_standings["html"])},
    })

@app.route("/health")
def health():
    return "ok\n", 200

@app.route("/ping")
def ping():
    # As cheap as it gets; great for keep-alive services
    return ("", 204)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
