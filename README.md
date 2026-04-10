# Glitch Trading Core

Core Glitch trading bots, oracle logic, and shared strategy modules for MT5 execution, with a parallel cTrader architecture track.

## Repo Layout

- `mt5/bots/`: current Python bot implementations for MT5
- `mt5/shared/`: shared indicators, risk guards, collectors, and orchestration modules
- `mt5/configs/`: sanitized example configs only
- `ctrader/docs/`: cTrader migration notes and target architecture
- `docs/`: platform and repository reference notes

## Included Bots

- `viper.py`: M5 momentum and EMA-pullback execution
- `cobra.py`: H1 support/resistance price-action execution
- `taipan.py`: M30 session breakout execution
- `mamba.py`: M15 Bollinger mean-reversion execution
- `anaconda.py`: H4 breakout execution
- `hydra.py`: M1 regime-routed execution
- `oracle.py`: portfolio-level coordination and conflict management

## Platform Direction

This repository keeps the strategy concept stable across two execution targets:

- `MT5`: current reference implementation and shared Python modules
- `cTrader`: target deployment platform for the next production generation

The intent is to preserve signal logic, risk policy, and oracle behavior while replacing the broker/execution layer for cTrader.

## Safety Notes

- Only sanitized example configs are included
- No state files, broker credentials, secrets, models, or training data are committed here
- Treat `mt5/configs/*.example.json` as templates for local setup only
