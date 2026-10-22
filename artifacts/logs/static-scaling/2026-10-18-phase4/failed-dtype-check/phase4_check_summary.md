# Phase 4 Distributed-Correctness Summary

| Field | Value |
|---|---|
| Schema | `phase4-check-v1` |
| Created | `2026-10-18T21:34:37Z` |
| Finished | `2026-10-18T21:36:31Z` |
| Device | `cuda` |
| World sizes | `1,2,4` |
| Frameworks | `flux,pytorch` |
| Categories | `202` |
| Verdict | **FAIL** |

## Categories

| Category | Status | max_abs | Violations | Detail |
|---|---|---:|---:|---|
| `integrity.fixture` | PASS | `0` | 0 | files_checked=4 problems=none |
| `bundle.load` | PASS | `0` | 0 | arrays=62 |
| `reference.load.flux` | PASS | `0` | 0 | npz=/home/kurapica/Projects/ddp_flux/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-18-phase4/reference/flux/cuda/phase4_reference.npz manifest=/home/kurapica/Projects/ddp_flux/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-18-phase4/reference/flux/cuda/phase4_reference_manifest.json |
| `reference.load.pytorch` | PASS | `0` | 0 | npz=/home/kurapica/Projects/ddp_flux/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-18-phase4/reference/pytorch/cuda/phase3_reference.npz manifest=/home/kurapica/Projects/ddp_flux/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-18-phase4/reference/pytorch/cuda/phase3_reference_manifest.json |
| `integrity.bundle` | PASS | `0` | 0 | arrays=62 parameters=11271432 hashes_checked=4 problems=none |
| `integrity.reference.flux` | PASS | `0` | 0 | arrays=337 problems=none |
| `reference.inputs.flux` | PASS | `0` | 0 | arrays=9 |
| `integrity.reference.pytorch` | PASS | `0` | 0 | arrays=337 problems=none |
| `reference.inputs.pytorch` | PASS | `0` | 0 | arrays=9 |
| `shards.schedule_fixture` | PASS | `0` | 0 | samples=99840 updates=195 problems=none |
| `integrity.ddp.flux.w1` | PASS | `0` | 0 | arrays_checked=450 hashes_checked=451 problems=none |
| `updates.flux.w1` | PASS | `0` | 0 | expected=3 problems=none |
| `shards.global_batch.flux.w1` | PASS | `0` | 0 | local=12 problems=none |
| `replicas.flux.w1` | PASS | `0` | 0 | states=param/m/v/beta_power problems=none |
| `integrity.shard_reference.flux.w1` | PASS | `0` | 0 | arrays=337 expected=337 path=/home/kurapica/Projects/ddp_flux/ddp-flux-thesis/artifacts/logs/static-scaling/2026-10-18-phase4/shard_reference/flux/cuda/w1/phase4_reference.npz |
| `same.flux.w1.grads.u1` | PASS | `1.63912773e-07` | 0 | synchronized=sum_r(localgrad)/1 arrays=62 |
| `same.flux.w1.grads.u3` | PASS | `0.000499079004` | 0 | synchronized=sum_r(localgrad)/1 arrays=62 |
| `control.shard_vs_full.flux.w1` | PASS | `0.000565052032` | 0 | non_gating_batch_split_envelope grads.u1=1.37835741e-07 grads.u2=0 grads.u3=0.000565052032 param.u3=3.43844295e-06 m.u3=6.17278274e-05 v.u3=8.45884642e-08 |
| `oracle.flux.w1` | PASS | `2.31993189e-07` | 0 | float64_decoupled_adamw_updates=3 arrays=186 |
| `same.flux.w1.param.u3` | PASS | `8.05594027e-07` | 0 | arrays=62 arrays=62 |
| `same.flux.w1.m.u3` | PASS | `5.13368286e-05` | 0 | arrays=62 arrays=62 |
| `same.flux.w1.v.u3` | PASS | `5.75491867e-08` | 0 | arrays=62 arrays=62 |
| `same.flux.w1.beta_power.u3` | PASS | `0` | 0 | expected=beta^(3+1) arrays=62 |
| `same.flux.w1.eta` | PASS | `0` | 0 | arrays=6 |
| `same.flux.w1.loss` | PASS | `0` | 0 | mean_of_rank_local_losses arrays=3 |
| `same.flux.w1.logits.u1` | PASS | `0` | 0 | rank_order_concatenation arrays=1 |
| `same.flux.w1.logits.u3` | PASS | `6.55651093e-07` | 0 | rank_order_concatenation arrays=1 |
| `protocol.flux.w1` | PASS | `0` | 0 | updates=3 entries_per_update=64 rank_traces=1 problems=none |
| `integrity.ddp.flux.w2` | PASS | `0` | 0 | arrays_checked=900 hashes_checked=902 problems=none |
| `updates.flux.w2` | PASS | `0` | 0 | expected=3 problems=none |
| `shards.global_batch.flux.w2` | PASS | `0` | 0 | local=6 problems=none |
| `replicas.flux.w2` | PASS | `0` | 0 | states=param/m/v/beta_power problems=none |
| `integrity.shard_reference.flux.w2` | PASS | `0` | 0 | arrays=337 expected=337 path=/home/kurapica/Projects/ddp_flux/ddp-flux-thesis/artifacts/logs/static-scaling/2026-10-18-phase4/shard_reference/flux/cuda/w2/phase4_reference.npz |
| `same.flux.w2.grads.u1` | PASS | `1.5553087e-07` | 0 | synchronized=sum_r(localgrad)/2 arrays=62 |
| `same.flux.w2.grads.u3` | PASS | `0.000244891271` | 0 | synchronized=sum_r(localgrad)/2 arrays=62 |
| `control.shard_vs_full.flux.w2` | PASS | `0.000718047842` | 0 | non_gating_batch_split_envelope grads.u1=2.38418579e-07 grads.u2=0.000281442888 grads.u3=0.000718047842 param.u3=3.43844295e-06 m.u3=7.69896433e-05 v.u3=9.81126504e-08 |
| `oracle.flux.w2` | PASS | `2.33121093e-07` | 0 | float64_decoupled_adamw_updates=3 arrays=186 |
| `same.flux.w2.param.u3` | PASS | `1.16601586e-06` | 0 | arrays=62 arrays=62 |
| `same.flux.w2.m.u3` | PASS | `2.44891271e-05` | 0 | arrays=62 arrays=62 |
| `same.flux.w2.v.u3` | PASS | `3.8617145e-08` | 0 | arrays=62 arrays=62 |
| `same.flux.w2.beta_power.u3` | PASS | `0` | 0 | expected=beta^(3+1) arrays=62 |
| `same.flux.w2.eta` | PASS | `0` | 0 | arrays=6 |
| `same.flux.w2.loss` | PASS | `2.38418579e-07` | 0 | mean_of_rank_local_losses arrays=3 |
| `same.flux.w2.logits.u1` | PASS | `0` | 0 | rank_order_concatenation arrays=1 |
| `same.flux.w2.logits.u3` | PASS | `5.96046448e-07` | 0 | rank_order_concatenation arrays=1 |
| `protocol.flux.w2` | PASS | `0` | 0 | updates=3 entries_per_update=64 rank_traces=2 problems=none |
| `integrity.ddp.flux.w4` | PASS | `0` | 0 | arrays_checked=1800 hashes_checked=1804 problems=none |
| `updates.flux.w4` | PASS | `0` | 0 | expected=3 problems=none |
| `shards.global_batch.flux.w4` | PASS | `0` | 0 | local=3 problems=none |
| `replicas.flux.w4` | PASS | `0` | 0 | states=param/m/v/beta_power problems=none |
| `integrity.shard_reference.flux.w4` | PASS | `0` | 0 | arrays=337 expected=337 path=/home/kurapica/Projects/ddp_flux/ddp-flux-thesis/artifacts/logs/static-scaling/2026-10-18-phase4/shard_reference/flux/cuda/w4/phase4_reference.npz |
| `same.flux.w4.grads.u1` | PASS | `1.49477273e-07` | 0 | synchronized=sum_r(localgrad)/4 arrays=62 |
| `same.flux.w4.grads.u3` | PASS | `0.000359893776` | 0 | synchronized=sum_r(localgrad)/4 arrays=62 |
| `control.shard_vs_full.flux.w4` | PASS | `0.000608099625` | 0 | non_gating_batch_split_envelope grads.u1=3.57627869e-07 grads.u2=0.000360909849 grads.u3=0.000608099625 param.u3=2.66358256e-06 m.u3=6.06100075e-05 v.u3=9.02073225e-08 |
| `oracle.flux.w4` | PASS | `2.31799411e-07` | 0 | float64_decoupled_adamw_updates=3 arrays=186 |
| `same.flux.w4.param.u3` | PASS | `4.78327274e-06` | 0 | arrays=62 arrays=62 |
| `same.flux.w4.m.u3` | PASS | `5.31408004e-05` | 0 | arrays=62 arrays=62 |
| `same.flux.w4.v.u3` | PASS | `5.22377377e-08` | 0 | arrays=62 arrays=62 |
| `same.flux.w4.beta_power.u3` | PASS | `0` | 0 | expected=beta^(3+1) arrays=62 |
| `same.flux.w4.eta` | PASS | `0` | 0 | arrays=6 |
| `same.flux.w4.loss` | PASS | `3.57627869e-07` | 0 | mean_of_rank_local_losses arrays=3 |
| `same.flux.w4.logits.u1` | PASS | `0` | 0 | rank_order_concatenation arrays=1 |
| `same.flux.w4.logits.u3` | PASS | `1.10268593e-06` | 0 | rank_order_concatenation arrays=1 |
| `protocol.flux.w4` | PASS | `0` | 0 | updates=3 entries_per_update=64 rank_traces=4 problems=none |
| `integrity.ddp.pytorch.w1` | PASS | `0` | 0 | arrays_checked=389 hashes_checked=390 problems=none |
| `updates.pytorch.w1` | PASS | `0` | 0 | expected=3 problems=none |
| `shards.global_batch.pytorch.w1` | PASS | `0` | 0 | local=12 problems=none |
| `replicas.pytorch.w1` | PASS | `0` | 0 | states=param/m/v/step problems=none |
| `integrity.shard_reference.pytorch.w1` | PASS | `0` | 0 | arrays=337 expected=337 path=/home/kurapica/Projects/ddp_flux/ddp-flux-thesis/artifacts/logs/static-scaling/2026-10-18-phase4/shard_reference/pytorch/cuda/w1/phase4_reference.npz |
| `py.syncgrad_replicas.w1` | PASS | `0` | 0 | world=1 problems=none |
| `same.pytorch.w1.grads.u1` | PASS | `0` | 0 | rank0 native DDP syncgrad arrays=62 |
| `same.pytorch.w1.grads.u3` | PASS | `0` | 0 | rank0 native DDP syncgrad arrays=62 |
| `control.shard_vs_full.pytorch.w1` | PASS | `0` | 0 | non_gating_batch_split_envelope grads.u1=0 grads.u2=0 grads.u3=0 param.u3=0 m.u3=0 v.u3=0 |
| `oracle.pytorch.w1` | PASS | `1.59117114e-07` | 0 | float64_decoupled_adamw_updates=3 arrays=186 |
| `same.pytorch.w1.param.u3` | PASS | `0` | 0 | arrays=62 arrays=62 |
| `same.pytorch.w1.m.u3` | PASS | `0` | 0 | arrays=62 arrays=62 |
| `same.pytorch.w1.v.u3` | PASS | `0` | 0 | arrays=62 arrays=62 |
| `same.pytorch.w1.step.u3` | PASS | `0` | 0 | actual=3 reference=3 |
| `same.pytorch.w1.eta` | PASS | `0` | 0 | arrays=6 |
| `same.pytorch.w1.loss` | PASS | `0` | 0 | mean_of_rank_local_losses arrays=3 |
| `same.pytorch.w1.logits.u1` | PASS | `0` | 0 | rank_order_concatenation arrays=1 |
| `same.pytorch.w1.logits.u3` | PASS | `0` | 0 | rank_order_concatenation arrays=1 |
| `protocol.pytorch.w1` | PASS | `0` | 0 | find_unused_parameters=false static_graph=false backend=nccl problems=none |
| `integrity.ddp.pytorch.w2` | PASS | `0` | 0 | arrays_checked=778 hashes_checked=780 problems=none |
| `updates.pytorch.w2` | PASS | `0` | 0 | expected=3 problems=none |
| `shards.global_batch.pytorch.w2` | PASS | `0` | 0 | local=6 problems=none |
| `replicas.pytorch.w2` | PASS | `0` | 0 | states=param/m/v/step problems=none |
| `integrity.shard_reference.pytorch.w2` | PASS | `0` | 0 | arrays=337 expected=337 path=/home/kurapica/Projects/ddp_flux/ddp-flux-thesis/artifacts/logs/static-scaling/2026-10-18-phase4/shard_reference/pytorch/cuda/w2/phase4_reference.npz |
| `py.syncgrad_replicas.w2` | PASS | `0` | 0 | world=2 problems=none |
| `same.pytorch.w2.grads.u1` | PASS | `0` | 0 | rank0 native DDP syncgrad arrays=62 |
| `same.pytorch.w2.grads.u3` | PASS | `0` | 0 | rank0 native DDP syncgrad arrays=62 |
| `control.shard_vs_full.pytorch.w2` | PASS | `0.000677028671` | 0 | non_gating_batch_split_envelope grads.u1=0.000266477466 grads.u2=0.000677028671 grads.u3=0.000592095777 param.u3=7.88737088e-06 m.u3=7.05905259e-05 v.u3=1.31500201e-07 |
| `oracle.pytorch.w2` | PASS | `1.57734544e-07` | 0 | float64_decoupled_adamw_updates=3 arrays=186 |
| `same.pytorch.w2.param.u3` | PASS | `0` | 0 | arrays=62 arrays=62 |
| `same.pytorch.w2.m.u3` | PASS | `0` | 0 | arrays=62 arrays=62 |
| `same.pytorch.w2.v.u3` | PASS | `0` | 0 | arrays=62 arrays=62 |
| `same.pytorch.w2.step.u3` | PASS | `0` | 0 | actual=3 reference=3 |
| `same.pytorch.w2.eta` | PASS | `0` | 0 | arrays=6 |
| `same.pytorch.w2.loss` | PASS | `2.38418579e-07` | 0 | mean_of_rank_local_losses arrays=3 |
| `same.pytorch.w2.logits.u1` | PASS | `0` | 0 | rank_order_concatenation arrays=1 |
| `same.pytorch.w2.logits.u3` | PASS | `0` | 0 | rank_order_concatenation arrays=1 |
| `protocol.pytorch.w2` | PASS | `0` | 0 | find_unused_parameters=false static_graph=false backend=nccl problems=none |
| `integrity.ddp.pytorch.w4` | PASS | `0` | 0 | arrays_checked=1556 hashes_checked=1560 problems=none |
| `updates.pytorch.w4` | PASS | `0` | 0 | expected=3 problems=none |
| `shards.global_batch.pytorch.w4` | PASS | `0` | 0 | local=3 problems=none |
| `replicas.pytorch.w4` | PASS | `0` | 0 | states=param/m/v/step problems=none |
| `integrity.shard_reference.pytorch.w4` | PASS | `0` | 0 | arrays=337 expected=337 path=/home/kurapica/Projects/ddp_flux/ddp-flux-thesis/artifacts/logs/static-scaling/2026-10-18-phase4/shard_reference/pytorch/cuda/w4/phase4_reference.npz |
| `py.syncgrad_replicas.w4` | PASS | `0` | 0 | world=4 problems=none |
| `same.pytorch.w4.grads.u1` | PASS | `1.1920929e-07` | 0 | rank0 native DDP syncgrad arrays=62 |
| `same.pytorch.w4.grads.u3` | PASS | `0.000530967489` | 0 | rank0 native DDP syncgrad arrays=62 |
| `control.shard_vs_full.pytorch.w4` | PASS | `0.000506050768` | 0 | non_gating_batch_split_envelope grads.u1=0.00026653707 grads.u2=0.000506050768 grads.u3=0.000364936888 param.u3=7.88737088e-06 m.u3=8.41077417e-05 v.u3=1.0779695e-07 |
| `oracle.pytorch.w4` | PASS | `1.58002255e-07` | 0 | float64_decoupled_adamw_updates=3 arrays=186 |
| `same.pytorch.w4.param.u3` | PASS | `7.88737088e-06` | 0 | arrays=62 arrays=62 |
| `same.pytorch.w4.m.u3` | PASS | `6.35245815e-05` | 0 | arrays=62 arrays=62 |
| `same.pytorch.w4.v.u3` | PASS | `5.9742888e-08` | 0 | arrays=62 arrays=62 |
| `same.pytorch.w4.step.u3` | PASS | `0` | 0 | actual=3 reference=3 |
| `same.pytorch.w4.eta` | PASS | `0` | 0 | arrays=6 |
| `same.pytorch.w4.loss` | PASS | `3.57627869e-07` | 0 | mean_of_rank_local_losses arrays=3 |
| `same.pytorch.w4.logits.u1` | PASS | `0` | 0 | rank_order_concatenation arrays=1 |
| `same.pytorch.w4.logits.u3` | PASS | `6.55651093e-06` | 0 | rank_order_concatenation arrays=1 |
| `protocol.pytorch.w4` | PASS | `0` | 0 | find_unused_parameters=false static_graph=false backend=nccl problems=none |
| `cross.ddp.w1.grads.u1` | FAIL | `0.000302698463` | 18253 | flux_offline_vs_pytorch_ddp n=11271432 |
| `cross.ddp.w1.grads.u2` | FAIL | `0.000508360565` | 146847 | flux_offline_vs_pytorch_ddp n=11271432 |
| `cross.ddp.w1.grads.u3` | FAIL | `0.000476382673` | 193953 | flux_offline_vs_pytorch_ddp n=11271432 |
| `cross.ddp.w1.param.u3` | PASS | `4.71249223e-06` | 0 | flux_vs_pytorch_ddp n=11271432 |
| `cross.ddp.w1.m.u3` | FAIL | `5.29885292e-05` | 1614 | flux_vs_pytorch_ddp n=11271432 |
| `cross.ddp.w1.v.u3` | PASS | `2.36555934e-07` | 0 | flux_vs_pytorch_ddp n=11271432 |
| `cross.ddp.w1.logits.u1` | PASS | `7.74860382e-07` | 0 | combined n=2400 |
| `cross.ddp.w1.logits.u3` | PASS | `2.29477882e-06` | 0 | combined n=2400 |
| `cross.ddp.w1.loss.u1` | PASS | `4.76837158e-07` | 0 | rank_average n=1 |
| `cross.ddp.w1.loss.u2` | PASS | `0` | 0 | rank_average n=1 |
| `cross.ddp.w1.loss.u3` | PASS | `4.76837158e-07` | 0 | rank_average n=1 |
| `cross.ddp.w1.eta.u1` | PASS | `0` | 0 | n=1 |
| `cross.ddp.w1.eta.u2` | PASS | `0` | 0 | n=1 |
| `cross.ddp.w1.eta.u3` | PASS | `0` | 0 | n=1 |
| `cross.reference.flux.w1.grads.u1` | FAIL | `0.000302698463` | 18253 | flux_ddp_vs_pytorch_reference n=11271432 |
| `cross.reference.flux.w1.grads.u3` | FAIL | `0.000476382673` | 193953 | flux_ddp_vs_pytorch_reference n=11271432 |
| `cross.reference.flux.w1.param.u3` | PASS | `4.71249223e-06` | 0 | flux_ddp_vs_pytorch_reference n=11271432 |
| `cross.reference.flux.w1.m.u3` | FAIL | `5.29885292e-05` | 1614 | flux_ddp_vs_pytorch_reference n=11271432 |
| `cross.reference.flux.w1.v.u3` | PASS | `2.36555934e-07` | 0 | flux_ddp_vs_pytorch_reference n=11271432 |
| `cross.reference.flux.w1.logits.u1` | PASS | `7.74860382e-07` | 0 | combined n=2400 |
| `cross.reference.flux.w1.logits.u3` | PASS | `2.29477882e-06` | 0 | combined n=2400 |
| `cross.reference.flux.w1.loss.u1` | PASS | `4.76837158e-07` | 0 | rank_average n=1 |
| `cross.reference.flux.w1.loss.u2` | PASS | `0` | 0 | rank_average n=1 |
| `cross.reference.flux.w1.loss.u3` | PASS | `4.76837158e-07` | 0 | rank_average n=1 |
| `cross.reference.flux.w1.eta.u1` | PASS | `0` | 0 | n=1 |
| `cross.reference.flux.w1.eta.u2` | PASS | `0` | 0 | n=1 |
| `cross.reference.flux.w1.eta.u3` | PASS | `0` | 0 | n=1 |
| `cross.ddp.w2.grads.u1` | FAIL | `6.70794398e-05` | 1304 | flux_offline_vs_pytorch_ddp n=11271432 |
| `cross.ddp.w2.grads.u2` | FAIL | `0.000520486385` | 118478 | flux_offline_vs_pytorch_ddp n=11271432 |
| `cross.ddp.w2.grads.u3` | FAIL | `0.000284809619` | 58590 | flux_offline_vs_pytorch_ddp n=11271432 |
| `cross.ddp.w2.param.u3` | PASS | `7.88737088e-06` | 0 | flux_vs_pytorch_ddp n=11271432 |
| `cross.ddp.w2.m.u3` | FAIL | `4.75859269e-05` | 675 | flux_vs_pytorch_ddp n=11271432 |
| `cross.ddp.w2.v.u3` | PASS | `2.38418579e-07` | 0 | flux_vs_pytorch_ddp n=11271432 |
| `cross.ddp.w2.logits.u1` | PASS | `7.15255737e-07` | 0 | combined n=2400 |
| `cross.ddp.w2.logits.u3` | PASS | `6.2584877e-06` | 0 | combined n=2400 |
| `cross.ddp.w2.loss.u1` | PASS | `0` | 0 | rank_average n=1 |
| `cross.ddp.w2.loss.u2` | PASS | `2.38418579e-07` | 0 | rank_average n=1 |
| `cross.ddp.w2.loss.u3` | PASS | `2.38418579e-07` | 0 | rank_average n=1 |
| `cross.ddp.w2.eta.u1` | PASS | `0` | 0 | n=1 |
| `cross.ddp.w2.eta.u2` | PASS | `0` | 0 | n=1 |
| `cross.ddp.w2.eta.u3` | PASS | `0` | 0 | n=1 |
| `cross.reference.flux.w2.grads.u1` | FAIL | `0.00030259043` | 18254 | flux_ddp_vs_pytorch_reference n=11271432 |
| `cross.reference.flux.w2.grads.u3` | FAIL | `0.000650951639` | 106666 | flux_ddp_vs_pytorch_reference n=11271432 |
| `cross.reference.flux.w2.param.u3` | PASS | `4.78327274e-06` | 0 | flux_ddp_vs_pytorch_reference n=11271432 |
| `cross.reference.flux.w2.m.u3` | FAIL | `5.85801899e-05` | 1346 | flux_ddp_vs_pytorch_reference n=11271432 |
| `cross.reference.flux.w2.v.u3` | PASS | `2.32830644e-07` | 0 | flux_ddp_vs_pytorch_reference n=11271432 |
| `cross.reference.flux.w2.logits.u1` | PASS | `7.74860382e-07` | 0 | combined n=2400 |
| `cross.reference.flux.w2.logits.u3` | PASS | `1.96695328e-06` | 0 | combined n=2400 |
| `cross.reference.flux.w2.loss.u1` | PASS | `0` | 0 | rank_average n=1 |
| `cross.reference.flux.w2.loss.u2` | PASS | `2.38418579e-07` | 0 | rank_average n=1 |
| `cross.reference.flux.w2.loss.u3` | PASS | `0` | 0 | rank_average n=1 |
| `cross.reference.flux.w2.eta.u1` | PASS | `0` | 0 | n=1 |
| `cross.reference.flux.w2.eta.u2` | PASS | `0` | 0 | n=1 |
| `cross.reference.flux.w2.eta.u3` | PASS | `0` | 0 | n=1 |
| `cross.ddp.w4.grads.u1` | FAIL | `6.70785084e-05` | 1307 | flux_offline_vs_pytorch_ddp n=11271432 |
| `cross.ddp.w4.grads.u2` | FAIL | `0.000377124175` | 50716 | flux_offline_vs_pytorch_ddp n=11271432 |
| `cross.ddp.w4.grads.u3` | FAIL | `0.000315119512` | 55890 | flux_offline_vs_pytorch_ddp n=11271432 |
| `cross.ddp.w4.param.u3` | PASS | `4.83170152e-06` | 0 | flux_vs_pytorch_ddp n=11271432 |
| `cross.ddp.w4.m.u3` | FAIL | `4.55807894e-05` | 496 | flux_vs_pytorch_ddp n=11271432 |
| `cross.ddp.w4.v.u3` | PASS | `2.34693289e-07` | 0 | flux_vs_pytorch_ddp n=11271432 |
| `cross.ddp.w4.logits.u1` | PASS | `7.15255737e-07` | 0 | combined n=2400 |
| `cross.ddp.w4.logits.u3` | PASS | `1.22189522e-06` | 0 | combined n=2400 |
| `cross.ddp.w4.loss.u1` | PASS | `0` | 0 | rank_average n=1 |
| `cross.ddp.w4.loss.u2` | PASS | `3.57627869e-07` | 0 | rank_average n=1 |
| `cross.ddp.w4.loss.u3` | PASS | `1.1920929e-07` | 0 | rank_average n=1 |
| `cross.ddp.w4.eta.u1` | PASS | `0` | 0 | n=1 |
| `cross.ddp.w4.eta.u2` | PASS | `0` | 0 | n=1 |
| `cross.ddp.w4.eta.u3` | PASS | `0` | 0 | n=1 |
| `cross.reference.flux.w4.grads.u1` | FAIL | `0.000302613713` | 18248 | flux_ddp_vs_pytorch_reference n=11271432 |
| `cross.reference.flux.w4.grads.u3` | FAIL | `0.000625493005` | 112842 | flux_ddp_vs_pytorch_reference n=11271432 |
| `cross.reference.flux.w4.param.u3` | PASS | `2.85636634e-06` | 0 | flux_ddp_vs_pytorch_reference n=11271432 |
| `cross.reference.flux.w4.m.u3` | FAIL | `6.40042126e-05` | 1365 | flux_ddp_vs_pytorch_reference n=11271432 |
| `cross.reference.flux.w4.v.u3` | PASS | `2.32830644e-07` | 0 | flux_ddp_vs_pytorch_reference n=11271432 |
| `cross.reference.flux.w4.logits.u1` | PASS | `7.74860382e-07` | 0 | combined n=2400 |
| `cross.reference.flux.w4.logits.u3` | PASS | `1.81794167e-06` | 0 | combined n=2400 |
| `cross.reference.flux.w4.loss.u1` | PASS | `1.1920929e-07` | 0 | rank_average n=1 |
| `cross.reference.flux.w4.loss.u2` | PASS | `4.76837158e-07` | 0 | rank_average n=1 |
| `cross.reference.flux.w4.loss.u3` | PASS | `1.1920929e-07` | 0 | rank_average n=1 |
| `cross.reference.flux.w4.eta.u1` | PASS | `0` | 0 | n=1 |
| `cross.reference.flux.w4.eta.u2` | PASS | `0` | 0 | n=1 |
| `cross.reference.flux.w4.eta.u3` | PASS | `0` | 0 | n=1 |

## Verdict

```text
PHASE4_CHECK FAIL cross.ddp.w1.grads.u1 cross.ddp.w1.grads.u2 cross.ddp.w1.grads.u3 cross.ddp.w1.m.u3 cross.reference.flux.w1.grads.u1 cross.reference.flux.w1.grads.u3 cross.reference.flux.w1.m.u3 cross.ddp.w2.grads.u1 cross.ddp.w2.grads.u2 cross.ddp.w2.grads.u3 cross.ddp.w2.m.u3 cross.reference.flux.w2.grads.u1 cross.reference.flux.w2.grads.u3 cross.reference.flux.w2.m.u3 cross.ddp.w4.grads.u1 cross.ddp.w4.grads.u2 cross.ddp.w4.grads.u3 cross.ddp.w4.m.u3 cross.reference.flux.w4.grads.u1 cross.reference.flux.w4.grads.u3 cross.reference.flux.w4.m.u3
```
