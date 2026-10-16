# Run from the thesis repo root:
# julia --startup-file=no --project=/home/kurapica/Projects/ddp_flux/flux-integration experiments/static-scaling/probes/probe_tensor_layout.jl
using Flux, Zygote

d = Flux.Dense(512, 200)
@assert size(d.weight) == (200, 512)
x2 = randn(Float32, 512, 4)
@assert size(d(x2)) == (200, 4)
W = randn(Float32, 200, 512)
bb = randn(Float32, 200)
d2 = Flux.Dense(W, bb)
@assert d2(x2) ≈ W * x2 .+ bb
g = only(Zygote.gradient(W -> sum(Flux.Dense(W, bb)(x2)), W))
@assert size(g) == (200, 512)
println("R1 dense_weight=", size(d.weight), " grad_shape=", size(g), " no_transpose=true")

wt = reshape(Float32.(1:6), 1, 1, 2, 3)
xp = reshape(Float32.(1:30), 1, 1, 5, 6)
xf = permutedims(xp, (4, 3, 2, 1))
wf = permutedims(wt, (4, 3, 2, 1))
@assert size(Flux.CrossCor((3, 2), 1 => 1).weight) == (3, 2, 1, 1)
yc = Flux.CrossCor(wf)(xf)
ref = zeros(Float32, 4, 4)
for i in 1:4, j in 1:4
    acc = 0f0
    for a in 1:2, b in 1:3
        acc += xp[1, 1, j + a - 1, i + b - 1] * wt[1, 1, a, b]
    end
    ref[i, j] = acc
end
@assert vec(yc) ≈ vec(ref)
@assert !(Flux.Conv(wf)(xf) ≈ yc)
@assert Flux.Conv(reverse(wf; dims = (1, 2)))(xf) ≈ yc
cc = Flux.CrossCor(wf)
gs = only(Zygote.gradient(m -> sum(m(xf)), cc))
@assert size(gs.weight) == (3, 2, 1, 1)
@assert !iszero(gs.weight)
println("R2 crosscor_output=", vec(yc))
println("R2 pytorch_semantics_reference=", vec(ref))
println("R2 crosscor_matches=true plain_conv_differs=true reversed_conv_matches=true grad_shape=", size(gs.weight))
