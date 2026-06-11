#!/usr/bin/env python3
# compat shim — moved to common/measure_real.py
import sys, pathlib, runpy
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
runpy.run_path(str(pathlib.Path(__file__).resolve().parent / "common" / "measure_real.py"), run_name="__main__")
