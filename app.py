#!/usr/bin/env python3
"""ZentroMart voice support UI.   python app.py   then open http://127.0.0.1:7860

  --port 7860        --host 127.0.0.1 (use 0.0.0.0 to share on your network; microphones need https or localhost)
  --no-browser       don't open the page automatically
The previous Gradio demo is still available as app_gradio.py.
"""
import argparse
import logging
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    import uvicorn
    from voiceops.server import build_default_app
    app = build_default_app()
    url = f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '') else a.host}:{a.port}"
    if not a.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"\n  ZentroMart support is running at {url}\n")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
