# Review tool routing failure

Date: 2026-10-19.

The `research` agent launch returned `Model not found: deepseek-v4-flash/.`.
The review continued through the available `general` agent.

A read-tool lookup for `scripts/remote/leonardo/run-timing-pilot.sh` also failed.
The actual launcher is `scripts/remote/leonardo/run-static-timing.sh`.

All local shell checks succeeded. These failures did not change campaign artifacts.
