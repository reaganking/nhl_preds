#!/usr/bin/env python3
from flask import Flask, Response, request, abort, jsonify
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
    ELO_INIT,
)

app = Flask(__name__)

# In-memory caches
_cached = {          # predictions
    "date": None,    # date object
    "html": None,    # str
    "games": 0,      # int
    "updated_ct": None,  # str
}
_cached_standings = { "date": None, "html": None }

REFRESH_TOKEN = os.environ.get("REFRESH_TOKEN", "")
PRED_HTML_PATH = "/tmp/predictions.html"
STAND_HTML_PATH = "/tmp/standings.html"
SIMS = int(os.environ.get("SIMS", "300"))
OT_RATE = float(os.environ.get("OT_RATE", "0.23"))

def _today_local():
    return datetime.now(tz=LOCAL_TZ).date()

def _generate_predictions_html_for(date_obj):
    # Elo to yesterday
    start_date = datetime(date_obj.year - 1, 10, 1, tzinfo=LOCAL_TZ).date()
    end_date = date_obj - timedelta(days=1)
    state = {"elo": {}}
    build_elo_from_history(state, start_date, end_date, include_types=("R","P"))

    # Records + season line
    records = get_team_records()
    correct, total, pct = compute_season_record_to_date()
    season_line = f"Season to date: {correct}-{max(total - correct, 0)} ({pct:.1f}%)" if total else "Season to date: —"

    # Predictions
    preds = predict_day(state, date_obj, records)
    write_html(preds, PRED_HTML_PATH, report_date=date_obj.isoformat(), season_line=season_line)

    with open(PRED_HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    _cached["date"] = date_obj
    _cached["html"] = html
    _cached["games"] = len(preds)
    _cached["updated_ct"] = datetime.now(tz=CENTRAL_TZ).strftime("%a, %b %d, %Y — %I:%M %p %Z")
    return html

def _generate_standings_html_for(date_obj):
    # Rebuild Elo to yesterday (for PO% simulation)
    start_date = datetime(date_obj.year - 1, 10, 1, tzinfo=LOCAL_TZ).date()
    end_date = date_obj - timedelta(days=1)
    state = {"elo": {}}
    build_elo_from_history(state, start_date, end_date, include_types=("R","P"))

    rows = get_standings_now_rows()
    po = simulate_playoff_probs(state, rows, date_obj, sims=SIMS, ot_rate=OT_RATE)
    attach_po_to_rows(rows, po)
    write_html_standings(rows, STAND_HTML_PATH, report_date=date_obj.isoformat())
    with open(STAND_HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()
    _cached_standings["date"] = date_obj
    _cached_standings["html"] = html
    return html

def _warm_now():
    try:
        today = _today_local()
        _generate_predictions_html_for(today)
        _generate_standings_html_for(today)
    except Exception:
        pass

@app.route("/")
def index():
    today = _today_local()

    if not _cached["html"]:
        try:
            with open(PRED_HTML_PATH, "r", encoding="utf-8") as f:
                _cached["html"] = f.read()
            _cached["date"] = today
            _cached["games"] = 0
            _cached["updated_ct"] = None
        except Exception:
            pass

    if not _cached["html"]:
        _generate_predictions_html_for(today)

    return Response(_cached["html"], mimetype="text/html")

@app.route("/standings")
def standings():
    if not _cached_standings["html"]:
        try:
            with open(STAND_HTML_PATH, "r", encoding="utf-8") as f:
                _cached_standings["html"] = f.read()
        except Exception:
            _generate_standings_html_for(_today_local())
    return Response(_cached_standings["html"], mimetype="text/html")

@app.route("/refresh")
def refresh():
    token = request.args.get("token", "")
    if REFRESH_TOKEN and token != REFRESH_TOKEN:
        abort(403)
    today = _today_local()
    _generate_predictions_html_for(today)
    _generate_standings_html_for(today)
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
