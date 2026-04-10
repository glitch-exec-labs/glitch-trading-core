# Platform Map

## MT5 Today

`mt5/` contains the live reference implementation:

- bot entrypoints in `mt5/bots/`
- shared infrastructure in `mt5/shared/`
- sanitized example configs in `mt5/configs/`

This is the practical baseline for current strategy behavior.

## Other Proven Execution Paths

The broader Glitch ecosystem has already explored more than one broker path:

- MT5 for the current snake stack
- Interactive Brokers in the Terciopelo lineage
- Kraken API in the crypto strategy lineage

## cTrader Next

`ctrader/` defines the target shape for the next deployment environment.

The cTrader track should emphasize:

- strategy rules
- normalized signal payloads
- feature extraction
- portfolio and prop-firm controls
- broker adapter isolation
- observability and state persistence

## Platform Split Principle

The strategy concept should remain platform-agnostic wherever possible:

- indicators and signal definitions should not depend directly on MT5 APIs
- broker objects should be isolated behind an execution adapter
- Oracle logic should consume normalized intents rather than MT5-specific payloads
- training schemas should stay stable across migrations

## Recommended Boundary

```text
Strategy logic      -> portable
Risk logic          -> portable
Feature engineering -> portable
Oracle logic        -> portable
Broker adapter      -> platform-specific
Order lifecycle     -> platform-specific
Session bootstrap   -> platform-specific
```
