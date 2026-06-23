"""optimizer.viz — visualization for funnel-optimizer campaign logs.

Two entry points:

  report.py     static self-contained HTML report (reward vs params, history,
                fidelity funnel, param importances, slice/contour).
  dashboard.py  reconstruct an Optuna study from a campaign JSONL into a
                JournalStorage file and launch optuna-dashboard, with an
                optional --live tail loop.

Both consume the JSONL rows written by run_funnel_optimizer.py (one episode
per line: config / fidelity / f3_reward / episode_reward / best_reward / ...).
"""

from .campaign_data import (
    CampaignData,
    load_campaign_rows,
    build_study,
    obs_objective,
)

__all__ = ["CampaignData", "load_campaign_rows", "build_study", "obs_objective"]
