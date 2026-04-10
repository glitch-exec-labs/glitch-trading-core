# Indian King Cobra

## Positioning

Indian King Cobra should be presented as a single unified bot.

It is not a mini-ensemble and not a collection of child bots. The strategy segmentation happens inside one executable through:

- lower-timeframe execution logic
- higher-timeframe confirmation logic
- per-asset parameter blocks
- ML-required, ML-soft, or ML-disabled symbol behavior
- session and news-aware filters

## Core Idea

Indian King Cobra is a momentum scalping framework that stays unified at the bot level while varying its behavior by asset and timeframe role.

## Structural Model

### Execution layer

- fast momentum and pullback logic
- lower-timeframe trade discovery
- spread, ADX, RSI, and volume gating

### Confirmation layer

- higher-timeframe trend agreement
- asset-specific session boundaries
- optional ML confirmation
- optional news veto

### Asset segmentation

The bot is one strategy engine with symbol-specific tuning, rather than separate bots for BTC, gold, FX, or oil.

## Why This Matters

This framing makes the project easier to explain and easier to productize:

- one bot identity
- one codebase
- one product story
- many internal operating profiles by asset and timeframe

## Recommended One-Line Description

Indian King Cobra is a unified momentum scalping bot with timeframe-aware execution, asset-specific tuning, ML gating, and news filters.
