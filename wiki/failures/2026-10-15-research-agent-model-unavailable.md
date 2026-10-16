# Research-agent model unavailable

Date: 2026-10-15

The research skill requested a research-agent invocation for the Tiny ImageNet proposal.
The Task tool rejected the invocation before research started:

```text
Model not found: deepseek-v4-flash/.
```

The invocation used `subagent_type: research` and description `Research Tiny ImageNet benchmark`.
No remote job or code execution occurred through the agent.
Direct web search and primary-document reads completed the research.
The agent configuration remains unchanged. This was a tool-availability failure, not a product failure.

The later `Compare conditional DDP semantics` research invocation failed with the same unavailable model.
Direct local source inspection and primary web documentation completed that research too.

A later documentation patch targeted a sentence absent from the devlog and failed context matching.
The patch made no change. The existing report and session records remained intact.

The `research-web` subagent invocation for the PyTorch/Tiny ImageNet environment pin verification failed the same day with HTTP 403 `SUBSCRIPTION_REQUIRED`.
Direct web search completed the verification.
No remote job or code execution was affected.

Later the same day, the `research-docs` and `research-web` subagent types failed with the same HTTP 403 `SUBSCRIPTION_REQUIRED` during the Phase 1 shared-specification session.
That session completed the fact gathering with `explore` subagents, pinned local package sources (Optimisers 0.4.9, Flux, NNlib, CUDA.jl), direct web search, and an independent verification subagent.
No remote job, training, or Flux source change was affected.
