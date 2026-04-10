# MT5 Track

This folder contains the current MT5-oriented Python implementation of the Glitch bot family.

## Contents

- [`bots/`](./bots): executable strategy entrypoints
- [`shared/`](./shared): indicators, risk guards, collectors, selectors, and support utilities
- [`configs/`](./configs): sanitized example configs

Current bot family in this track:

- `viper.py`
- `cobra.py`
- `taipan.py`
- `mamba.py`
- `anaconda.py`
- `hydra.py`
- `king_cobra_auto.py`
- `oracle.py`

## Intent

The MT5 track is the current reference implementation.

It should remain useful as:

- the source of truth for legacy strategy behavior
- the baseline for feature and risk logic
- the comparison point for the cTrader migration

## Important

This repository intentionally excludes:

- live credentials
- deployment state
- training data
- model artifacts
