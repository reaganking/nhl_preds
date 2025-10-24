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
    LOCAL_TZ,
)

app = Flask(__name__)

# In-memory HTML cache + meta (date, counts, timestamp)
_cached = {
    "date": None,        # date object
    "html": None,        # str
    "games": 0,          # int
    "updated_ct": None,  # str, already rendered "Last updated" line's time in CT
}

REFRESH_TOKEN = os.environ.get("REFRESH_TOKEN", "")
CENTRAL_TZ = ZoneInfo("America/Chicago")
OUTPUT_PATH = "/tmp/predictions.html"


def _today_local():
    return datetime.now(tz=LOCAL_TZ).date()


def _generate_html_for(date_obj):
    """Build Elo from last season till yesterday, generate today’s predictions + HTML."""
    # Build Elo (walk-forward) up to yesterday
    start_date = datetime(date_obj.year - 1, 10, 1, tzinfo=LOCAL_TZ).date()
    end_date = date_obj - timedelta(days=1)
    state = {"elo": {}}
    build_elo_from_history(state, start_date, end_date, include_types=("R", "P"))

    # Records for HTML
    records = get_team_records()

    # Season-to-date accuracy line
    correct, total, pct = compute_season_record_to_date()
    season_line = (
        f"Season to date: {correct}-{max(total - correct, 0)} ({pct:.1f}%)"
        if total else "Season to date: —"
    )

    # Today’s predictions
    preds = predict_day(state, date_obj, records)

    # Write pretty HTML (also writes "Last updated at ..." in the footer)
    write_html(preds, OUTPUT_PATH, report_date=date_obj.isoformat(), season_line=season_line)

    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    _cached["date"] = date_obj
    _cached["html"] = html
    _cached["games"] = len(preds)
    # Extract the rendered timestamp from the footer (or recompute here as a backup)
    _cached["updated_ct"] = datetime.now(tz=CENTRAL_TZ).strftime("%a, %b %d, %Y — %I:%M %p %Z")
    return html


def _warm_now():
    try:
        _generate_html_for(_today_local())
    except Exception:
        # avoid crashing the worker thread
        pass


@app.route("/")
def index():
    today = _today_local()

    # If memory cache is empty (cold start), serve disk HTML immediately if present
    if not _cached["html"]:
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                _cached["html"] = f.read()
            _cached["date"] = today          # good enough for display
            _cached["games"] = 0              # unknown without parsing; status.json is optional
            _cached["updated_ct"] = None
        except Exception:
            pass

    # If we still don't have HTML (first ever hit), generate synchronously once.
    if not _cached["html"]:
        _generate_html_for(today)

    return Response(_cached["html"], mimetype="text/html")


@app.route("/refresh")
def refresh():
    """Synchronous rebuild; returns small text body (OK)."""
    token = request.args.get("token", "")
    if REFRESH_TOKEN and token != REFRESH_TOKEN:
        abort(403)
    html = _generate_html_for(_today_local())
    return Response("OK\n", mimetype="text/plain")


@app.route("/warm")
def warm():
    """Background rebuild; returns zero-byte 204 so cron-job.org never complains."""
    token = request.args.get("token", "")
    if REFRESH_TOKEN and token != REFRESH_TOKEN:
        abort(403)
    Thread(target=_warm_now, daemon=True).start()
    return ("", 204)


@app.route("/status.json")
def status():
    """Lightweight status for monitors."""
    d = _cached["date"].isoformat() if _cached["date"] else None
    return jsonify({
        "date": d,
        "games": _cached["games"],
        "updated_ct": _cached["updated_ct"],
        "ok": bool(_cached["html"]),
    })


@app.route("/health")
def health():
    return "ok\n", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
