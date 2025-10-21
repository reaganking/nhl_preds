# server.py
from flask import Flask, Response
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os

# Import your existing functions from app.py
from app import (
    build_elo_from_history,
    get_team_records,
    predict_day,
    write_html,
    compute_season_record_to_date,
    LOCAL_TZ,
)

app = Flask(__name__)

# Simple cache in memory so we don’t recompute on every hit
_cached_html = {"date": None, "html": None}

def _generate_html_for(date_obj):
    """Build predictions HTML for the given local date and return the HTML string."""
    # Build Elo: from last Oct 1 to yesterday
    start_date = datetime(date_obj.year - 1, 10, 1, tzinfo=LOCAL_TZ).date()
    end_date = date_obj - timedelta(days=1)

    state = {"elo": {}}
    build_elo_from_history(state, start_date, end_date, include_types=("R", "P"))

    records = get_team_records()

    # Season record through yesterday
    correct, total, pct = compute_season_record_to_date()
    season_line = f"Season to date: {correct}-{max(total - correct, 0)} ({pct:.1f}%)" if total else "Season to date: —"

    preds = predict_day(state, date_obj, records)

    # Write to a temp file then read back (re-uses your existing write_html)
    tmp_path = "/tmp/predictions.html"
    write_html(preds, tmp_path, report_date=date_obj.isoformat(), season_line=season_line)
    with open(tmp_path, "r", encoding="utf-8") as f:
        return f.read()

def _today_local():
    return datetime.now(tz=LOCAL_TZ).date()

@app.route("/")
def index():
    today = _today_local()

    # Rebuild if date changed or cache empty
    if _cached_html["date"] != today or not _cached_html["html"]:
        _cached_html["html"] = _generate_html_for(today)
        _cached_html["date"] = today

    return Response(_cached_html["html"], mimetype="text/html")

@app.route("/refresh")
def refresh():
    """Force a rebuild (used by the daily scheduled job and for manual refresh)."""
    today = _today_local()
    _cached_html["html"] = _generate_html_for(today)
    _cached_html["date"] = today
    return "OK\n", 200

if __name__ == "__main__":
    # Render runs on $PORT
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
