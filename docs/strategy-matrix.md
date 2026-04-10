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
| Indian King Cobra | Multi-timeframe | single-bot momentum framework | one unified bot that segments execution by timeframe role and per-asset profiles, with optional ML and news gating | filtered scalping layer |
| Oracle | Multi-bot | coordination | manages overlap, conflict, and aggregate exposure | portfolio governor |

## Strategic Coverage

The Glitch stack is intentionally diversified across trading styles:

- trend continuation
- breakout
- mean reversion
- structure-based price action
- filtered momentum scalping
- regime-aware adaptation
- portfolio-level orchestration

## Why The Oracle Matters

Without the Oracle, the ensemble is just a collection of bots.

With the Oracle, the system can:

- prevent duplicate risk
- prefer aligned signals
- soften conflict between mean-reversion and trend modules
- enforce portfolio-level discipline

## Indian King Cobra Structure

Indian King Cobra is intentionally modeled as one bot.

Its strategy segmentation happens inside the bot rather than by splitting into multiple separate executables. The main divisions are:

- execution timeframe versus confirmation timeframe
- asset-specific parameter blocks
- ML-required, ML-soft, or ML-disabled symbol modes
- session and news filtering by asset context

## Prop-Firm Relevance

The most prop-friendly strategic building blocks in this family are:

- Taipan-style session breakout logic
- Anaconda-style higher-timeframe breakout confirmation
- Hydra-style centralized risk and regime awareness

These are the concepts most likely to carry forward into the cTrader migration.
