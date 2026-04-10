# Strategy Matrix

This document summarizes the role of each Glitch bot in the broader ensemble.

| Bot | Primary TF | Strategy Type | Core Behavior | Role In Ensemble |
| --- | --- | --- | --- | --- |
| Viper | M5 | momentum + EMA pullback | reacts quickly to trend continuation and pullback re-entry | fast directional execution |
| Cobra | H1 | structure + price action | trades support/resistance behavior and candle structure | higher-level confirmation and structure bias |
| Taipan | M30 | session breakout | trades directional expansion from session-defined ranges | prop-challenge candidate for breakout logic |
| Mamba | M15 | mean reversion | fades stretched moves in range conditions | balance against trend-heavy modules |
| Anaconda | H4 | breakout validation | focuses on slower, stronger structural continuation | swing confirmation layer |
| Hydra | M1 | regime routing | adapts between trend and range playbooks | tactical short-horizon adapter |
| Oracle | Multi-bot | coordination | manages overlap, conflict, and aggregate exposure | portfolio governor |

## Strategic Coverage

The Glitch stack is intentionally diversified across trading styles:

- trend continuation
- breakout
- mean reversion
- structure-based price action
- regime-aware adaptation
- portfolio-level orchestration

## Why The Oracle Matters

Without the Oracle, the ensemble is just a collection of bots.

With the Oracle, the system can:

- prevent duplicate risk
- prefer aligned signals
- soften conflict between mean-reversion and trend modules
- enforce portfolio-level discipline

## Satellite Strategies

Indian King Cobra and Terciopelo sit beside the core Ouroboros stack as standalone products.

They are part of the broader Glitch ecosystem, but not part of the in-repo six-snake ensemble described here.

## Prop-Firm Relevance

The most prop-friendly strategic building blocks in this family are:

- Taipan-style session breakout logic
- Anaconda-style higher-timeframe breakout confirmation
- Hydra-style centralized risk and regime awareness

These are the concepts most likely to carry forward into the cTrader migration.
