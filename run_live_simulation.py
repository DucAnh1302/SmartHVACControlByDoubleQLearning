"""
Launcher for HVAC Agent Live Simulation GUI.
Usage:
    python run_live_simulation.py
"""

import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

from app.live_simulation_gui import main

if __name__ == "__main__":
    main()
