"""gen1 — single-step black-box design-space exploration (DSE).

Grid search, cascade with fixed gates, and physical track (ORFS-backed env).
Agents (random, evo, UCB, Bayesian, enumerate) each propose one full config
per trial; no episode structure or promotion policy. See docs/04_optimizer.md.
"""
