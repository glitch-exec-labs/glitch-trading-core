# Repo Ecosystem

This document defines the recommended repository structure for the Glitch trading ecosystem.

## Primary Repo

### Glitch Trading Core

Role:

- umbrella engineering repository
- shared architecture and platform direction
- Oracle and ensemble concepts
- shared modules, adapters, and platform migration work

Suggested description:

Core Glitch trading architecture, Oracle coordination, and shared strategy modules for MT5 today and cTrader next.

Important boundary:

- this repo should link to satellite products
- it should not house their strategy code long term

## Flagship Strategy Brand

### Ouroboros Snake Strategy

Role:

- flagship coordinated ensemble strategy
- Oracle plus the six-snake stack
- the final and primary Glitch multi-bot identity

Suggested description:

Glitch's flagship ensemble strategy, combining Oracle coordination with the six-snake execution stack.

## Strategy Lineage Guidance

Public repo positioning should tell a simple story:

- **Indian King Cobra** and **Terciopelo** are earlier V1-era strategy lines from the broader project history
- **Ouroboros Snake Strategy** is the flagship and final coordinated strategy brand for the trading family
- **Glitch Trading Core** is the umbrella architecture layer that explains how the family evolved

That wording helps visitors understand the older repos without competing with the flagship narrative.

+## Satellite Strategy Repos

### Indian King Cobra

Role:

- earlier V1-era standalone single-bot strategy line
- momentum scalping framework
- internally segmented by timeframe role and per-asset profiles

Suggested description:

Indian King Cobra is a unified momentum scalping bot with timeframe-aware execution, asset-specific tuning, ML gating, and news filters.

Repo:

- [glitch-trade-indian-king-cobra](https://github.com/glitch-exec-labs/glitch-trade-indian-king-cobra)

### Terciopelo

Role:

- earlier V1-era standalone equities strategy line
- relative-value and mean-reversion focus
- different asset universe and different audience from the snake stack

Suggested description:

Terciopelo is a Mag7-focused relative-value and mean-reversion trading system with technical scoring and LLM-assisted news filtering.

Repo:

- [glitch-trade-terciopelo](https://github.com/glitch-exec-labs/glitch-trade-terciopelo)

Implementation note:

- there is an Interactive Brokers-oriented Terciopelo lineage
- there is also an MT5-oriented Terciopelo lineage
- this makes Terciopelo a strong example of broker-portable strategy identity

## Broker Portability Story

The repo ecosystem should communicate that Glitch is not married to a single broker or venue.

The architecture direction is:

- strategy logic should be portable
- risk logic should be portable
- broker adapters should be swappable
- MT5, cTrader, Kraken, IB, and other integrations should be treated as execution layers, not as strategy definitions

Evidence already exists in the current code history:

- the snake stack is MT5-oriented today
- Terciopelo has an Interactive Brokers variant and an MT5-oriented variant
- Kraken strategy code exists in the broader Glitch codebase history

## Practical Naming Guidance

- use **Glitch Trading Core** for the umbrella repository
- use **Ouroboros Snake Strategy** for the flagship ensemble brand
- use **Indian King Cobra** and **Terciopelo** as standalone repo names and product identities
- link satellite repos from the umbrella repo instead of presenting them as in-repo core components
