# Run from the thesis repo root:
# julia --startup-file=no --project=/home/kurapica/Projects/ddp_flux/flux-integration experiments/static-scaling/probes/probe_optimizer_contract.jl
using Optimisers

rd = Optimisers.AdamW(; eta = 1e-3, beta = (0.9, 0.999), lambda = 1e-4, epsilon = 1e-8, couple = true)
rn = Optimisers.AdamW(; eta = 1e-3, beta = (0.9, 0.999), lambda = 0.0, epsilon = 1e-8, couple = true)
println("R3 decay_rule=", rd)
println("R3 no_decay_rule=", rn)
try
    Optimisers.AdamW(1e-3; beta = (0.9, 0.999), lambda = 1e-4, epsilon = 1e-8)
    println("R3 mixed_constructor=accepted (UNEXPECTED)")
catch e
    println("R3 mixed_constructor=", typeof(e))
end

x = Float32[1, 2]
s = Optimisers.init(rd, x)
b = Float32.(rd.beta)
println("R4 s=0 beta_power=", s[3])
@assert collect(s[3]) ≈ collect(b) .^ 1
s1, _ = Optimisers.apply!(rd, s, x, Float32[0.1, 0.2])
@assert collect(s1[3]) ≈ collect(b) .^ 2
s2, _ = Optimisers.apply!(rd, s1, x, Float32[0.1, 0.2])
@assert collect(s2[3]) ≈ collect(b) .^ 3
@assert s2[3] == s1[3] .* b
sr = 2
bt = reduce((acc, _) -> acc .* b, 1:sr; init = b)
@assert bt == s2[3]
srec = (copy(s2[1]), copy(s2[2]), bt)
sa, _ = Optimisers.apply!(rd, s2, x, Float32[0.3, -0.1])
sc, _ = Optimisers.apply!(rd, srec, x, Float32[0.3, -0.1])
@assert sa == sc
println("R4 s=1 beta_power=", s1[3], " s=2 beta_power=", s2[3])
println("R4 stored=beta^(s+1)=true reconstruction_repeated_multiplication_exact=true continuation_from_exchanged_state_exact=true")
