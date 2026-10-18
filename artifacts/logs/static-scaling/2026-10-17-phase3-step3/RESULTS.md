# Step 3 results — full three-update comparison (FAILED)

Date: 2026-10-17. Source: `console-driver.log` (job 60904928, node `lrdn0125`).
The unchanged rule `|a-b| <= 1e-5 + 1e-4*max(|a|,|b|)` did not pass.
Per-leaf relative L2 is supplementary evidence only.

## Per-category checks

| Check line | Status | Detail |
|---|---|---|
| `phase3.eta` | PASS | max_abs_diff=0 max_rel_diff=0 worst=eta.u1 violations=0 |
| `phase3.activations` | FAIL | max_abs_diff=0.00764757395 max_rel_diff=1 worst=act.stage4.block1 violations=2705938 arrays=act.stage2.block0:616769;act.stage2.block1:864972;act.stage3.block0:346553;act.stage3.block1:460153;act.stage4.block0:179821;+2more |
| `phase3.logits` | FAIL | max_abs_diff=0.000564932823 max_rel_diff=0.683415258 worst=logits.u3 violations=4981 arrays=logits.u1:1670;logits.u2:1623;logits.u3:1688 |
| `phase3.loss` | PASS | max_abs_diff=0.000131607056 max_rel_diff=2.36939646e-05 worst=loss.u2 violations=0 |
| `phase3.grads.u1` | FAIL | max_abs_diff=0.00524188578 max_rel_diff=1.99996119 worst=grad.u1.stem.conv.weight violations=3842324 arrays=grad.u1.stem.conv.weight:1714;grad.u1.stem.gn.weight:64;grad.u1.stem.gn.bias:62;grad.u1.stage1.block0.conv1.weight:34903;grad.u1.stage1.block0.conv2.weight:34341;+56more |
| `phase3.grads.u3` | FAIL | max_abs_diff=0.00748962909 max_rel_diff=1.99995497 worst=grad.u3.stem.conv.weight violations=3906482 arrays=grad.u3.stem.conv.weight:1709;grad.u3.stem.gn.weight:62;grad.u3.stem.gn.bias:63;grad.u3.stage1.block0.conv1.weight:35083;grad.u3.stage1.block0.conv2.weight:34311;+56more |
| `phase3.params.u3` | FAIL | max_abs_diff=1.1857599e-05 max_rel_diff=1.98433172 worst=param.u3.stage3.block1.conv1.weight violations=5 arrays=param.u3.stage2.block1.conv1.weight:1;param.u3.stage3.block0.conv1.weight:1;param.u3.stage3.block0.conv2.weight:1;param.u3.stage3.block1.conv1.weight:1;param.u3.stage4.block0.conv1.weight:1 |
| `phase3.m.u3` | FAIL | max_abs_diff=0.00147970638 max_rel_diff=1.99998828 worst=m.u3.stem.conv.weight violations=721437 arrays=m.u3.stem.conv.weight:1653;m.u3.stem.gn.weight:59;m.u3.stem.gn.bias:53;m.u3.stage1.block0.conv1.weight:26600;m.u3.stage1.block0.conv2.weight:23516;+55more |
| `phase3.v.u3` | PASS | max_abs_diff=3.09478492e-06 max_rel_diff=1 worst=v.u3.classifier.weight violations=0 |
| `phase3.beta_power.u3` | PASS | max_abs_diff=0 max_rel_diff=0 worst=beta_power.u3.stem.conv.weight violations=0 |
| `phase3.step` | PASS | reference_step=3 flux_updates=3 expected=3 |

## Per-leaf L2 summary

| Summary line | Data |
|---|---|
| `PHASE3_LEAF_L2_SUMMARY` | `category=grads.u1 leaves=62 max_rel_l2=4.897306e-02 worst_leaf=grad.u1.stage1.block1.conv1.weight zero_ref_nonzero_actual=0 nonzero_ref_zero_actual=0 below_floor=0` |
| `PHASE3_LEAF_L2_SUMMARY` | `category=grads.u3 leaves=62 max_rel_l2=5.608752e-02 worst_leaf=grad.u3.stage1.block1.conv1.weight zero_ref_nonzero_actual=0 nonzero_ref_zero_actual=0 below_floor=0` |
| `PHASE3_LEAF_L2_SUMMARY` | `category=params.u3 leaves=62 max_rel_l2=1.148905e-01 worst_leaf=param.u3.stage1.block1.gn1.bias zero_ref_nonzero_actual=0 nonzero_ref_zero_actual=0 below_floor=0` |
| `PHASE3_LEAF_L2_SUMMARY` | `category=param_increment.u3 leaves=62 max_rel_l2=1.148905e-01 worst_leaf=delta.u3.stage1.block1.gn1.bias zero_ref_nonzero_actual=0 nonzero_ref_zero_actual=0 below_floor=0` |
| `PHASE3_PARAM_INCREMENT_SUMMARY` | `category=param_increment.u3 leaves=62 max_rel_l2=1.148905e-01 worst_leaf=delta.u3.stage1.block1.gn1.bias zero_ref_nonzero_actual=0 nonzero_ref_zero_actual=0 below_floor=0` |
| `PHASE3_LEAF_L2_SUMMARY` | `category=m.u3 leaves=62 max_rel_l2=3.943915e-02 worst_leaf=m.u3.stage1.block1.conv1.weight zero_ref_nonzero_actual=0 nonzero_ref_zero_actual=0 below_floor=0` |
| `PHASE3_LEAF_L2_SUMMARY` | `category=v.u3 leaves=62 max_rel_l2=5.166543e-02 worst_leaf=v.u3.stem.conv.weight zero_ref_nonzero_actual=0 nonzero_ref_zero_actual=0 below_floor=0` |

## Worst leaves by valid relative L2 (descending)


### `grads.u1`

1. `grad.u1.stage1.block1.conv1.weight` — rel_l2=4.897306e-02
2. `grad.u1.stage1.block0.conv1.weight` — rel_l2=4.526946e-02
3. `grad.u1.stage1.block0.conv2.weight` — rel_l2=4.395762e-02
4. `grad.u1.stem.conv.weight` — rel_l2=4.364193e-02
5. `grad.u1.stem.gn.bias` — rel_l2=4.296387e-02
6. `grad.u1.stage1.block1.gn1.weight` — rel_l2=4.126794e-02
7. `grad.u1.stage1.block0.gn1.weight` — rel_l2=4.108317e-02
8. `grad.u1.stage1.block1.conv2.weight` — rel_l2=4.068238e-02
9. `grad.u1.stage1.block1.gn1.bias` — rel_l2=4.027871e-02
10. `grad.u1.stage1.block1.gn2.weight` — rel_l2=3.909008e-02

### `grads.u3`

1. `grad.u3.stage1.block1.conv1.weight` — rel_l2=5.608752e-02
2. `grad.u3.stage1.block0.conv1.weight` — rel_l2=5.217298e-02
3. `grad.u3.stem.gn.weight` — rel_l2=4.785363e-02
4. `grad.u3.stage1.block1.gn1.bias` — rel_l2=4.741191e-02
5. `grad.u3.stem.conv.weight` — rel_l2=4.731877e-02
6. `grad.u3.stage1.block0.conv2.weight` — rel_l2=4.701230e-02
7. `grad.u3.stage1.block1.gn1.weight` — rel_l2=4.648743e-02
8. `grad.u3.stem.gn.bias` — rel_l2=4.537203e-02
9. `grad.u3.stage1.block0.gn2.weight` — rel_l2=4.391019e-02
10. `grad.u3.stage1.block0.gn1.weight` — rel_l2=4.372536e-02

### `params.u3`

1. `param.u3.stage1.block1.gn1.bias` — rel_l2=1.148905e-01
2. `param.u3.stage1.block0.gn2.bias` — rel_l2=9.838551e-02
3. `param.u3.stage1.block0.gn1.bias` — rel_l2=9.054066e-02
4. `param.u3.stage2.block0.gn1.bias` — rel_l2=8.425368e-02
5. `param.u3.stem.gn.bias` — rel_l2=6.381367e-02
6. `param.u3.stage3.block0.gn1.bias` — rel_l2=5.992393e-02
7. `param.u3.stage2.block0.gn2.bias` — rel_l2=5.927627e-02
8. `param.u3.stage2.block0.proj_gn.bias` — rel_l2=5.927627e-02
9. `param.u3.stage2.block1.gn2.bias` — rel_l2=3.820658e-02
10. `param.u3.stage3.block0.gn2.bias` — rel_l2=3.613289e-02

### `param_increment.u3`

1. `delta.u3.stage1.block1.gn1.bias` — rel_l2=1.148905e-01
2. `delta.u3.stem.gn.weight` — rel_l2=1.080788e-01
3. `delta.u3.stage3.block0.gn1.weight` — rel_l2=1.042810e-01
4. `delta.u3.stage1.block1.gn1.weight` — rel_l2=1.026422e-01
5. `delta.u3.stage1.block0.gn2.weight` — rel_l2=9.931462e-02
6. `delta.u3.stage1.block0.gn2.bias` — rel_l2=9.838551e-02
7. `delta.u3.stage1.block0.conv1.weight` — rel_l2=9.650880e-02
8. `delta.u3.stage2.block0.gn1.weight` — rel_l2=9.456556e-02
9. `delta.u3.stem.conv.weight` — rel_l2=9.256024e-02
10. `delta.u3.stage1.block1.conv1.weight` — rel_l2=9.215754e-02

### `m.u3`

1. `m.u3.stage1.block1.conv1.weight` — rel_l2=3.943915e-02
2. `m.u3.stem.conv.weight` — rel_l2=3.934713e-02
3. `m.u3.stage1.block0.conv1.weight` — rel_l2=3.725651e-02
4. `m.u3.stage1.block0.conv2.weight` — rel_l2=3.515998e-02
5. `m.u3.stem.gn.weight` — rel_l2=3.469806e-02
6. `m.u3.stem.gn.bias` — rel_l2=3.292590e-02
7. `m.u3.stage1.block0.gn2.weight` — rel_l2=3.113638e-02
8. `m.u3.stage1.block0.gn1.weight` — rel_l2=3.092658e-02
9. `m.u3.stage1.block1.gn1.bias` — rel_l2=2.935742e-02
10. `m.u3.stage1.block1.gn1.weight` — rel_l2=2.918140e-02

### `v.u3`

1. `v.u3.stem.conv.weight` — rel_l2=5.166543e-02
2. `v.u3.stage1.block1.conv1.weight` — rel_l2=4.742758e-02
3. `v.u3.stage1.block1.gn1.bias` — rel_l2=4.366151e-02
4. `v.u3.stage1.block0.conv2.weight` — rel_l2=3.979365e-02
5. `v.u3.stage1.block1.gn1.weight` — rel_l2=3.809818e-02
6. `v.u3.stage1.block0.gn2.weight` — rel_l2=3.721749e-02
7. `v.u3.stage1.block0.conv1.weight` — rel_l2=3.592663e-02
8. `v.u3.stem.gn.bias` — rel_l2=3.393738e-02
9. `v.u3.stage1.block0.gn1.weight` — rel_l2=3.385350e-02
10. `v.u3.stage1.block1.gn2.weight` — rel_l2=3.096894e-02

