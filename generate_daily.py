# generate_daily.py
from datetime import datetime
from app import LOCAL_TZ
from server import _generate_html_for, _today_local  # reuse the same logic

if __name__ == "__main__":
    # Build and print a short message; HTML is not saved permanently here,
    # the web service caches in memory on first request and /refresh endpoint.
    today = _today_local()
    _ = _generate_html_for(today)
    print(f"Generated HTML for {today.isoformat()}")
