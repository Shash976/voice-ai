"""gen2 — multi-fidelity funnel optimizer with promotion actions and a surrogate.

FunnelEnv exposes a sequential decision problem (kill/re-proxy/promote/commit)
over a 4-stage fidelity ladder. The promotion policy (LinUCB bandit, or fixed
gates as a baseline) is trained offline against logged funnel traces. A
gradient-boosted surrogate predicts F3 metrics from cheaper F0–F2 observables.
See docs/08_funnel_optimizer.md.
"""
