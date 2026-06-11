#!/usr/bin/env python3
# compat shim — moved to gen2/run_funnel_optimizer.py
import sys, pathlib, runpy
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
runpy.run_path(str(pathlib.Path(__file__).resolve().parent / "gen2" / "run_funnel_optimizer.py"), run_name="__main__")
