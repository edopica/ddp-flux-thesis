# MoE inactivity and PyTorch conditional graphs

Date: 2026-10-15

## Request and findings

The user requested broad Italian answers to two supervisor questions about MoE parameter dropping and PyTorch conditional graphs.
Local code and external documentation were necessary. Git history was unnecessary because the question concerns current behavior.

- Temporary expert inactivity differs from token dropping and structural parameter removal.
- The wrapper averages over every rank, with zero contributions for locally absent gradients.
- Globally absent leaves skip the inner optimizer update and preserve Adam state.
- A dense zero gradient remains present. Packed expert tensors do not provide independent absence semantics for each expert slice.
- The fixed model and optimizer-state structure excludes post-setup expert removal or addition.
- PyTorch DDP uses graph traversal, autograd hooks, a usage map, and gradient buckets.
- Its reducer preserves the gradient field for globally unused parameters. Explicit `zero_grad(set_to_none=True)` preserves skip-step semantics.
- Standard dynamic DDP still communicates buckets containing unused parameters.
- `skip_all_reduce_unused_params=True` requires an unchanged unused set across ranks throughout training.
- `static_graph=True` requires stable used/unused sets and the documented training-graph contract. It does not mean compilation.
- Replicated DDP support does not provide expert parallelism, token dispatch, or learned routing.

## Sources

Local source: `../flux-integration/src/distributed/public_api.jl`, especially lines 472–483, 617–675, and 745–801.
The deferred-behavior audit documents the fixed-structure contract in section 4.9.
The latest context records the October 13–14 numerical guards, which supersede earlier audit limitations.

External references use a pinned version rather than the moving stable alias:

- [PyTorch 2.8 DDP API](https://docs.pytorch.org/docs/2.8/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch 2.8 reducer source](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/distributed/c10d/reducer.cpp)
- [Optimizer zero_grad](https://docs.pytorch.org/docs/2.8/generated/torch.optim.Optimizer.zero_grad.html)
- [DDP design note](https://docs.pytorch.org/docs/2.8/notes/ddp.html), whose internal-design description explicitly dates from v1.4
- [Switch Transformers](https://jmlr.org/papers/v23/21-0998.html)
- [DeepSpeed MoE](https://www.deepspeed.ai/tutorials/mixture-of-experts/)

## Evidence and outcome

This session used source inspection and documentation. No runtime equivalence or performance experiment ran.
`git status --short` passed and identified existing proposal documentation before these edits.
The research-agent invocation failed before execution because its configured model was unavailable. Direct research completed the task.
The README progress table remains unchanged because no implementation milestone changed.

**Exact next action:** clarify whether the supervisor means temporary expert inactivity, token dropping, or structural expert removal.
