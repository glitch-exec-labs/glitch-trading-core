# Contributing

Thanks for contributing to Glitch Trading Core.

This repo is the architecture hub for the Glitch trading family, so good contributions here usually improve clarity, platform portability, or shared strategy understanding across the wider ecosystem.

## Good Contribution Targets

- architecture and documentation clarity
- sanitized config examples
- platform-portable abstractions
- repo navigation improvements
- typo fixes, diagrams, and public-facing docs polish

## Please Avoid

- committing live credentials, broker state, logs, or model artifacts
- adding environment-specific deployment secrets
- mixing private operating details into public documentation
- changing repo branding or family structure without explaining why

## Workflow

1. Branch from `main`.
2. Keep changes scoped and reviewable.
3. Prefer additive documentation or safe refactors over broad rewrites.
4. If you change public behavior or structure, update the relevant docs in the same PR.
5. Open a pull request with a short summary, validation notes, and any ecosystem impact.

## Notes For Public Contributions

- treat every config committed here as a sanitized template
- preserve Apache 2.0 licensing and attribution files
- keep the MT5 and cTrader platform narrative internally consistent
