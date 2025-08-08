# RotationalStiffnessApp.py
import sys
from pathlib import Path

def main():
    app_path = Path(__file__).with_name("app.py")
    if not app_path.exists():
        raise FileNotFoundError(f"Could not find {app_path}")

    # Streamlit CLI entry point
    import streamlit.web.cli as stcli
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
