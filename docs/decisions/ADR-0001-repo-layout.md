# ADR-0001 - Repository layout

Date: 2026-07-02
Status: Accepted

## Context

The thesis work needs both direct access to a local Flux.jl fork and a clean place to track progress, scripts, artifacts, audit notes, and agent context.

## Decision

Use two sibling repositories:

- `Flux.jl/`: local fork of Flux.jl for source inspection and implementation work.
- `ddp-flux-thesis/`: thesis-control repository for dashboards, notes, scripts, logs, artifacts, decisions, and context files.

## Consequences

- Flux pull requests remain clean and focused.
- Thesis notes do not pollute the Flux repository.
- Carlo can inspect progress from the README dashboard.
- Agents can recover context from Markdown files instead of relying on hidden chat memory.
