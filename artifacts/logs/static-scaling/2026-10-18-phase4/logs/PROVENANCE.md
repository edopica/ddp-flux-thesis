# Provenance - static-scaling Phase 4 remote check

All values below were captured by the driver from the live run; no result is estimated.
Missing optional provenance is printed as `<unset>`, not invented.

| Item | Value |
|---|---|
| Host | `lrdn0983.leonardo.local` |
| SLURM job | `61296074` |
| SLURM node list | `lrdn0983` |
| Start (UTC) | `2026-10-18T19:21:44Z` |
| End (UTC) | `2026-10-18T19:25:57Z` |
| Driver | `scripts/remote/leonardo/run-phase4-check.sh` |
| Driver SHA-256 | `e16612b5a1a6d299221bdcac929770bc1ada95365597eb5799705868be53e7fd` |
| Status | **PASS** |

## Source identity (provided by the operator)

| Field | Value |
|---|---|
| `SOURCE_COMMIT` | `<unset>` |
| `SOURCE_STATE_SHA256` | `<unset>` |
| `FLUX_COMMIT` | `<unset>` |
| `FLUX_STATE_SHA256` | `<unset>` |

## Source and input SHA-256

```text
# static_scaling sources
62f3c64d7a2bb9ca28133545ec5303140458660ca3784823be56238aa82febeb  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/check_all.jl
ee3f10b248ff9a6075f5e828e030096bf5b6f46b5d30a608dcfd5e16b464f038  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/check_dataset.py
8135a2417c271b811bc39b2f32a8f800d14475764b60348351637629af07abda  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/check_fixtures_data.jl
4f9421548609b7ba4f0047593a2124cc053a381d5f1529049190ce25e98377ee  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/check_fixtures_operators.jl
43ad641bfaad58f52e5aa2ab1405113dde29b107a27c5e21f8f549d978b985a1  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/check_phase3_flux.jl
85297025dfd9b2222e2ecbae6c5bdca80f0e724f80b680fcdc67fa6653723515  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/check_phase3_reference.py
51633091cf0ebcbf4a0504911d76f31e1d547ef028f89dd3cb9876760180fac8  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/check_phase4.py
dfd28b9f4911b5d7f10558904a146970ef5c3624d51bdd0d55ed5920ca39497e  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/check_precision_math.jl
27edd256f917cc8101c3d881d16949695c71f4fac7265eaccb1a8498c44885e5  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/compare_reference_arrays.py
15205ada4dac120d52ad10f851e43386fd1135ed756e42bd2507f9af7089f855  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/dataset_adapter.py
383f1fa02fd11e308292eda3d9e41ac916236b17cd3406e9c370bc64b2e63d16  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/diagnose_phase3_flux.jl
3316869b7ab18d69b1c7127df85b9c674743a3be25e691052babaa62326536dc  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/diagnose_phase3_flux_stage2.jl
cb37fc124b7dab1ffcd43ed0f02fa56dc2f425333980c5f30ea67fd89ea26e97  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/diagnose_phase3_torch_stage2.py
95af202557ff715913617432c473e3b4c4554b994b82602441818ec97e25359b  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/make_fixtures.py
1529d738327aa7e2a9764eb0a218a1932f8d330a3cd4e083c2eb9fcd74a47241  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/make_initial_bundle.py
0a9a2645f2bf0bd66e94e1351d3538e050a8261dee451c77dd4b7b282cdab21d  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/make_phase3_reference.py
653d1709e86707a8ebaba29cb35c0e2064b8dadc136e12a1d16248489acab12b  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/make_schedule.py
9bc2d752bb7c5543374ed4ed8e107048bec95ec17d5e293a2bade23f0a7055a4  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/model_flux.jl
77ac48e815b58948ab64b16938d988cae35f4719f084c25432703f6ffadfdf76  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/model_torch.py
b570933f425a6d555afd066651a8a2b341a211faa2877fc71e128abd9779fe1d  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/negative_tests_local.py
32d85954fdc7eba2b0073106faddde195adb73819c4b85855204bf08cc6351e6  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_common.jl
ceb7b710fd934f9baa1775fa713ca2c689bd294a92794af5642a6203b5007400  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_flux.jl
75c1905118b0d53b96f48efa1c32fc4cfbe09782aee605f125013391923d2712  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/phase4_torch.py
3efd8f4a1b37da847a7bd456138dff86a8dfb927e9cdfb0ebfb64011564c2b81  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/precision.jl
e0f8e5e58c673dd1e64ad0d0ca34196261e39f21d630ba10dbdceb9c7153065f  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/prepare_dataset.py
098cc7ab2072f9c2f7bebe577db26f9a757020de8eb29c80e2f3a1cdbe4ff63e  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/replay.jl
e270c774b6cd5d34af55805cdb2c3caf0af832d125b7e2531f7e14eb4a5abad0  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/selftest_local.py
e0aef52ec41a4bf5822793141141522d8b09af166d0fb595df0673d1363141fe  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/test_precision_policy.jl
# remote drivers
e16612b5a1a6d299221bdcac929770bc1ada95365597eb5799705868be53e7fd  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/remote/leonardo/run-phase4-check.sh
ddb4e3b8a107341ccb8f1004248450836dc1bad9a88b0b6837169adfd65212b1  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/remote/leonardo/precompile-phase3.sh
# inputs
6614a00ba7cac588b331b65cef4240f27624944c7a9df0383fdbeee3124066b3  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/data_fixture.npz
51448d5905c5dfd9a47b945d96102d19ea55073fbfe9f249fd268cca03455a65  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/fixture_manifest.json
aba66a8ef7bbc82fc81db8cd629092b917cd7d85efafed82553f853c45352662  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/schedule_fixture.npz
a790b03c1d68361a09171ae53870bd49ffad73c80144b2db54263bed6f9d0317  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-16-phase2/initial_bundle.npz
97e5007cbb557f4c4a476a00c7e8076e8c59858e58d597d0df88529985da0783  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-18-phase3-final/phase3_reference.npz
5ba6c29f3a6167f4341e8e78b0685cb93a5b76e83575beca8db82d8585753eb7  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/artifacts/baselines/static-scaling/2026-10-18-phase3-final/phase3_reference_manifest.json
# environments
6e75702e0e98a6fe3a3c253a278be827f771295d950dee67eae38367bb8f032d  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/julia-env/Project.toml
c8077e0c6504adedfdce5a35dd1a2ba6bde057fc45ee9bd7ba1edbd35ee23874  /leonardo/home/userexternal/epicazio/projects/ddp-flux-thesis/scripts/static_scaling/julia-env/Manifest.toml
13970e0a9b8595595ce63a9e05d69de0b381fff86d16f774c4e0d04d8bb38853  /leonardo/home/userexternal/epicazio/projects/flux-integration/test/Project.toml
MISSING /leonardo/home/userexternal/epicazio/projects/flux-integration/test/Manifest.toml
```

## Tool versions

The complete tool/version transcript is `/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/logs/a-source-metadata.log`.

```text
slurm 25.05.7-BullSequana.1.2
python 3.11.17
numpy 2.4.6
torch 2.9.1+cu126 cuda 12.6
cuda_available True
cudnn 91002
nccl (2, 27, 5)
julia version 1.12.6
```

## Artifact SHA-256

```text
7de0106f383743a57ed3be2226493ef2b9ccda407bc7ae549d88e6d20d9e1479  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/check/phase4_check_report.json
7de0106f383743a57ed3be2226493ef2b9ccda407bc7ae549d88e6d20d9e1479  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/check/phase4_report.json
24a5f46fd071955fc02c802d13199d38a08f8d63759a791850c9f76f587f324b  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/check/phase4_report.txt
c0e1807a760514ea467d7ad181d58136fc9d026276e0de19f011f498b99e9a95  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/flux/w1/rank0.json
1fd0b229e29d3b0be01d9fe0f64891b7c860879a488e59b0dbc0e0bf07b3f8a7  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/flux/w1/rank0.npz
8e2988ba1fda97ab2c7949279044a46f9bd3e6f19fdd731a2cd11cd52b094cf9  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/flux/w1/summary.json
5efccbad3bde3e52e0031dbd4193fa5411ba540baeefc7675e132181640d2da4  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/flux/w2/rank0.json
e1f8a3dc058a935b9bf124351c30108f64337be5bc9c1db0589c24e22ee75494  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/flux/w2/rank0.npz
57a12c48be7c244a0fc95e78601637f0aeb02ff29d606a5cfce982b24e91d133  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/flux/w2/rank1.json
60ccb60ed7e81cd8339e34d84e38b78ebdeff07e7f248d701854e6fa9d99b9b1  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/flux/w2/rank1.npz
153d7668917098a24704cd8d2edb2793c7e38c9a59844d6ae8df803cbea12cdc  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/flux/w2/summary.json
a2b504711753a43fbf48cfcac49367a988b9befde45a4790cc372037beae8ab2  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/flux/w4/rank0.json
11cfdeb18294087ae7cb585ffd69711250560b49438aa8f56bd63ef711abdadb  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/flux/w4/rank0.npz
549b49c433dbd1717c42f780c96e28a70cc5779dacec48f1c312fd8fae170e69  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/flux/w4/rank1.json
41a083f3d7ebb60a6d219753da9162b8003948137fee01d422a48fca4dd1223d  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/flux/w4/rank1.npz
dd6bf9a6e5b5ba383178c6c51da1cf245d2d7f649f82ce085f44a36f4c720b92  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/flux/w4/rank2.json
7778aa14f10ad098bb3749c0cc833cc5c229f59d73d320bc46aa29fd7dd81c44  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/flux/w4/rank2.npz
f839caf09bbbd11b10d87dac7e71a966bda0c44410b30c1908a98208355aec98  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/flux/w4/rank3.json
d3bf2f6f92032775a7f3c6414c856883c5d9c238744fa60fe629060dc8a9402e  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/flux/w4/rank3.npz
d71c9f433b650f592df142f32530514a590e505550516e63786176436da18690  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/flux/w4/summary.json
d873eb0b92c3366a2153a9182666908d4fe4b92a5c63b35a1e3ae919813b691d  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/pytorch/w1/rank0.json
f00eb3e117300ef5326e1811deb743f62336067eab5ab36c1af76b770fbe6508  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/pytorch/w1/rank0.npz
9b316510b5c1788608a441e6ddab155359188054a67e8a9fdf6958e87b7b2a9c  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/pytorch/w1/summary.json
1ecfd06fe12b791e53d8277ab2afad9754349031e8a7bc23b2b5f980d341adc1  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/pytorch/w2/rank0.json
01b2ff0a12588cd5ce0d6cdf32730589676768685fd873fbc709c78fcf65bc17  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/pytorch/w2/rank0.npz
146ae164c0ba968c9006d005a7fea4307cd1d2fbcefd47fa08618254628eae66  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/pytorch/w2/rank1.json
64446975965506b6276eeb1f66bc9af61c6e2bef7d671b83394c200545ebe20e  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/pytorch/w2/rank1.npz
9523ab88c6d7b37f7085b7bc5362c8385248587bd6e76abfa2eac2aef21a3899  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/pytorch/w2/summary.json
cf96620f7e37dd1e12d118f759506a0ac63d8363ac0630b2b3acd3ce691a460e  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/pytorch/w4/rank0.json
e34afad0f9fdcbe968fac54af9588fa5aae8e86cdf230f80ea696d7c013b32a8  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/pytorch/w4/rank0.npz
8482550be8c67e392f55f90db1c3c3f1b88d6bff2e753fd2571c2f2ed56ad0fc  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/pytorch/w4/rank1.json
2aac04a087c04185043bb8c21739a1f8b9189cf62cd95001f9ee05b5e37ba7e2  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/pytorch/w4/rank1.npz
9423b4a2b41c8f5c367af405cd98dea8a2bc7e3ad12abf86bcd04260577e5b61  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/pytorch/w4/rank2.json
46b9e23ba00a9b2b3cd5485396278b9f350895aff7c4f36853492a33fa16e10b  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/pytorch/w4/rank2.npz
b3ddcdbf0bdc88df76adf9eb83b6ba19954bb610be49afe2dbb396965ea0eb49  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/pytorch/w4/rank3.json
683f05dc43961c367ee66f4bfbc000a74981d0cfa9a60bc9cdfd9745c980c0cb  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/pytorch/w4/rank3.npz
7dd8c00994a211109cf93520e15b493581810808c8d4d899eec70f1881df74f5  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/ddp/pytorch/w4/summary.json
cb7a1c47602fd426ca5aa4b415c021faba9a706c44738317095256641c4f9d79  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/reference/flux/cuda/phase4_reference.json
cb7a1c47602fd426ca5aa4b415c021faba9a706c44738317095256641c4f9d79  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/reference/flux/cuda/phase4_reference_manifest.json
227999c81cdf3807ffb949f6105e2de9aaa62f10b20f251c36f6c93e268e1adb  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/reference/flux/cuda/phase4_reference.npz
ec6606865e41393656cba19d19c029f8e38fa7be45b6a982a82a2d7438101ac3  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/reference/pytorch/cuda/phase3_reference_manifest.json
97e5007cbb557f4c4a476a00c7e8076e8c59858e58d597d0df88529985da0783  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/reference/pytorch/cuda/phase3_reference.npz
d31c1232213eea062e11ba14a52bd6bc3d3f519619802da2b52372e850b24cc4  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/flux/cuda/w1/phase4_reference.json
d31c1232213eea062e11ba14a52bd6bc3d3f519619802da2b52372e850b24cc4  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/flux/cuda/w1/phase4_reference_manifest.json
e3b5f1d35e47396ae69e211c52df1f059b9dc7067ee1aa09ac263a3c271cf9e2  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/flux/cuda/w1/phase4_reference.npz
0e110f914e19e8e9fe06ef4569d88d2eb8953acab31feecedb2194a2a25f0064  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/flux/cuda/w2/phase4_reference.json
0e110f914e19e8e9fe06ef4569d88d2eb8953acab31feecedb2194a2a25f0064  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/flux/cuda/w2/phase4_reference_manifest.json
bf8d1a3a4b392f4c96653bcd6b794c0832819c59da2d17b840661b9824be8e34  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/flux/cuda/w2/phase4_reference.npz
d3ab0578fd1fc85b6a26f7ef54150882158b097c247dbf8e176de4c5969e1f84  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/flux/cuda/w4/phase4_reference.json
d3ab0578fd1fc85b6a26f7ef54150882158b097c247dbf8e176de4c5969e1f84  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/flux/cuda/w4/phase4_reference_manifest.json
4500cb2f570d5c3e9f75938eb8490b521277fe6bf3eab3a2e0fb2e88e3ed1151  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/flux/cuda/w4/phase4_reference.npz
24fbf1f1a75ca7f1489d45c4ea4b6c397c2ce178bd5a7ec336af5ed9c0321de1  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/pytorch/cuda/w1/phase4_reference.json
c56d5c5d01147bdaf6c521a912499f80dbf956c15b3a3b5a165509df56c790cf  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/pytorch/cuda/w1/phase4_reference_manifest.json
b079575e47578d4c40186cb68b6ee6ddd74545fe725e72f0d7f80c1d5c8d08c4  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/pytorch/cuda/w1/phase4_reference.npz
96db4eac3dc60e6944e5ac8a767f4c83f2d55c377cc77d07d0d69d2b637e0e6c  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/pytorch/cuda/w2/phase4_reference.json
a8ab4de65795ca89d23615dee971ac12bd4908ea5c9e99c30a4d32f97757aab2  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/pytorch/cuda/w2/phase4_reference_manifest.json
6ec80f12d7b261c6b840e8c1c261124dc7939e8929d1e352a66ddbc899bfa749  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/pytorch/cuda/w2/phase4_reference.npz
e6584aae8c214b1539862087f8a756a1643982d45e2a54a878b1d0093d5f4708  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/pytorch/cuda/w4/phase4_reference.json
63b4058be878bc07a7ef5d7d09fe2b291acf9edf10ae24007a68be4897b6f41a  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/pytorch/cuda/w4/phase4_reference_manifest.json
fa1dd3de2af6a613a67f85fc5c6e41575a90b8dd87e851bb338d5aade33a14ae  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/shard_reference/pytorch/cuda/w4/phase4_reference.npz
eecc2d5fe6c9198bbd5ff62b35b5fa4601a42192ade07e63bcbdba1065fd87bb  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/split/flux/cuda/split_control.json
11e9ba2bbf2cb1a8ad3f010d6ab78bd458de9e2b14cb775ae7df93680768c092  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/split/flux/cuda/split_control.npz
a4e2001406dfa919809d3f29d9023df7d245345ac4a1832e8806a070c046a1ed  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/split/pytorch/cuda/split_control.json
5ddbe7864d9080e51e0e5782b37826e421b7441cbcaef00d4a354a9241689c86  /leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-evidence/2026-10-18-phase4/split/pytorch/cuda/split_control.npz
```

## Failure record

No stage failed.
