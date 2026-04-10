# cTrader Track

This folder documents the cTrader migration target for Glitch Trading Core.

## Goal

Keep the trading concepts the same while replacing the MT5-specific runtime with a cTrader-native execution layer.

## Preserve

- Strategy intent per bot
- Oracle conflict handling
- Prop-firm risk controls
- Shared feature engineering and labeling concepts
- Session filters and regime routing

## Replace

- MT5 terminal connectivity
- MT5 account/session bootstrapping
- MT5 order placement and position tracking
- MT5 symbol metadata and point-value handling

## Expected cTrader Architecture

- Signal layer: strategy logic translated from MT5 Python rules
- Risk layer: prop-firm guard, portfolio guard, symbol exposure limits
- Execution layer: cTrader API adapter and order lifecycle manager
- Data layer: candle ingestion, spread/session metadata, feature snapshots
- Coordination layer: Oracle-style consensus and conflict resolution

## Migration Notes

- Start with `Taipan`, `Anaconda`, and `Hydra` concepts for the first cTrader prop stack
- Keep feature schemas compatible where possible so MT5 historical data can still inform model training
- Separate platform-specific code from strategy code early to reduce rewrite cost
