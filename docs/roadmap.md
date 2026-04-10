# Roadmap

## Phase 1: Public Repo Foundation

- publish the MT5 strategy core cleanly
- remove secrets and live state
- document the architecture and strategy roles
- establish a cTrader migration direction

## Phase 2: Strategy / Platform Separation

- isolate broker-specific code from signal logic
- standardize normalized signal payloads
- centralize portfolio controls behind cleaner interfaces
- prepare shared feature schema for MT5 and cTrader

## Phase 3: cTrader Migration

- implement a cTrader execution adapter
- port the highest-value prop-friendly strategies first
- keep Oracle coordination behavior intact
- support Linux deployment on GCP

## Phase 4: Unified Bot Direction

- distill the strongest ideas from Taipan, Hydra, Viper, and Anaconda
- build a single prop-focused execution stack
- keep ensemble logic available for research and fallback

## Near-Term Priorities

- richer cTrader scaffolding
- strategy-to-adapter separation
- shared config conventions across platforms
- deployment docs for Linux and cTrader
