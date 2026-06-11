#!/usr/bin/env python3
# compat shim — moved to gen1/test_reward_sanity.py
import sys, pathlib, runpy
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
runpy.run_path(str(pathlib.Path(__file__).resolve().parent / "gen1" / "test_reward_sanity.py"), run_name="__main__")
