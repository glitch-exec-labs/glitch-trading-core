# Platform Map

## MT5 Today

`mt5/` contains the current Python execution stack:

- Bot entrypoints in `mt5/bots/`
- Shared infrastructure in `mt5/shared/`
- Sanitized example configs in `mt5/configs/`

## cTrader Next

`ctrader/` is the design space for the Linux deployment target.

The recommended separation is:

- strategy rules
- feature extraction
- portfolio and prop-firm controls
- broker adapter
- observability and state persistence

## Design Principle

The strategy concept should remain platform-agnostic wherever possible:

- indicators and signal definitions should not depend on MT5 APIs
- broker objects should be isolated behind an execution adapter
- oracle logic should consume normalized intents rather than MT5-specific payloads
- training schemas should remain stable across platform migrations
