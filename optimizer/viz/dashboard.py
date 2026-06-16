#!/usr/bin/env python3
"""dashboard.py — live Optuna dashboard for a funnel-optimizer campaign.

Reconstructs an Optuna study from a campaign JSONL into a JournalStorage file,
then launches optuna-dashboard against it. With --live, it keeps tailing the
JSONL and appends new episodes as trials; optuna-dashboard auto-refreshes, so
you watch reward / param-importance / slice / parallel-coordinate update as
run_funnel_optimizer.py runs.

    # one-shot snapshot of the latest campaign, opens http://127.0.0.1:8080/
    python3 optimizer/viz/dashboard.py

    # follow a live run (in another terminal, while the optimizer writes the log)
    python3 optimizer/viz/dashboard.py --live --log tinymac_accel_run1.jsonl

    # just rebuild the study file without launching the server
    python3 optimizer/viz/dashboard.py --no-serve --storage /tmp/funnel.db

Storage default is a JournalStorage file next to the log. optuna-dashboard reads
it live; deleting it forces a clean rebuild (or pass --rebuild).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from viz.campaign_data import DEFAULT_LOG, build_study, load_campaign_rows  # noqa: E402


def _make_storage(path: Path):
    """JournalStorage backed by a file — what optuna-dashboard reads live."""
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    return JournalStorage(JournalFileBackend(str(path)))


def _sync(storage, log_path: Path, campaign: str, study_name: str) -> int:
    """(Re)build/append trials from the current JSONL. Returns total trial count."""
    rows = load_campaign_rows(log_path, campaign)
    study, added, _specs = build_study(rows, study_name=study_name, storage=storage)
    return len(study.trials), added


def main() -> None:
    ap = argparse.ArgumentParser(description="Live Optuna dashboard for a campaign log")
    ap.add_argument("--log", default=str(DEFAULT_LOG), help="campaign JSONL")
    ap.add_argument("--campaign", default="latest",
                    help="campaign_id | 'latest' | 'all'")
    ap.add_argument("--storage", default=None,
                    help="JournalStorage file (default: <log>.optuna-journal)")
    ap.add_argument("--study-name", default=None, help="Optuna study name")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--live", action="store_true",
                    help="tail the log and append new episodes as they appear")
    ap.add_argument("--interval", type=float, default=3.0,
                    help="seconds between log polls in --live mode")
    ap.add_argument("--rebuild", action="store_true",
                    help="delete any existing storage file first (clean rebuild)")
    ap.add_argument("--no-serve", action="store_true",
                    help="build/sync the study but do not launch the dashboard")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"campaign log not found: {log_path}")
        sys.exit(1)

    storage_path = Path(args.storage) if args.storage \
        else log_path.with_suffix(log_path.suffix + ".optuna-journal")
    study_name = args.study_name or f"funnel-{args.campaign}"

    if args.rebuild and storage_path.exists():
        storage_path.unlink()
        print(f"removed existing storage {storage_path}")

    storage = _make_storage(storage_path)

    total, added = _sync(storage, log_path, args.campaign, study_name)
    print(f"Study '{study_name}': {total} trials ({added} new) → {storage_path}")

    if args.no_serve:
        return

    # Launch optuna-dashboard against the journal storage.
    cmd = ["optuna-dashboard", str(storage_path),
           "--host", args.host, "--port", str(args.port)]
    print(f"Launching: {' '.join(cmd)}")
    print(f"  → open http://{args.host}:{args.port}/  (study: {study_name})")
    proc = subprocess.Popen(cmd)

    try:
        if args.live:
            print(f"Live mode: polling {log_path.name} every {args.interval:.0f}s "
                  f"(Ctrl-C to stop)")
            while True:
                time.sleep(args.interval)
                if proc.poll() is not None:
                    print("dashboard process exited")
                    break
                try:
                    total, added = _sync(storage, log_path, args.campaign, study_name)
                    if added:
                        print(f"  +{added} episodes → {total} trials")
                except Exception as exc:  # noqa: BLE001
                    print(f"  [warn] sync failed: {exc}")
        else:
            proc.wait()
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
