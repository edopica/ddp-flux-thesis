# Direct CPU worker. Launch exactly once under MPI; communication stays outside AD.
using Flux, MPI, Optimisers, Functors, JSON, Sockets, SHA
const DU = Flux.DistributedUtils
opts = Dict(split(a[3:end], '='; limit=2) for a in ARGS)
@assert get(opts, "backend", "mpi") == "mpi"
DU.initialize(DU.MPIBackend; cuda_devices=missing, amdgpu_devices=missing)
b = DU.get_distributed_backend(DU.MPIBackend)
r, w = MPI.Comm_rank(MPI.COMM_WORLD), MPI.Comm_size(MPI.COMM_WORLD)
@assert w == parse(Int, opts["world"])
shared = MPI.Comm_split_type(MPI.COMM_WORLD, MPI.COMM_TYPE_SHARED, r)
fluxroot = dirname(dirname(pathof(Flux)))
source_hash = bytes2hex(sha256(join([relpath(joinpath(d,f),fluxroot)*bytes2hex(sha256(read(joinpath(d,f)))) for dir in ("src","ext") for (d,_,fs) in walkdir(joinpath(fluxroot,dir)) for f in sort(fs)],"\n")))
metadata = (source_sha256=source_hash,rank=r, host=gethostname(), processor=MPI.Get_processor_name(),
    local_rank=MPI.Comm_rank(shared), local_size=MPI.Comm_size(shared),
    flux=pathof(Flux), project=Base.active_project(), depot=DEPOT_PATH,
    mpi=MPI.Get_library_version(), julia=string(VERSION),
    manifest_sha256=bytes2hex(sha256(read(joinpath(dirname(Base.active_project()),"Manifest.toml")))),
    slurm_node=get(ENV,"SLURM_NODEID",""),slurm_local=get(ENV,"SLURM_LOCALID",""))
meta = MPI.gather(metadata, MPI.COMM_WORLD; root=0)
if r == 0
    @assert length(unique(x.host for x in meta)) == parse(Int, opts["nodes"])
    @assert all(x.local_size == parse(Int, opts["ranks-per-node"]) for x in meta)
end
@assert realpath(dirname(dirname(pathof(Flux)))) == realpath(joinpath(@__DIR__,"../../../flux-multinode"))
records = []
function leaves(x, path="root")
    x isa AbstractArray{<:Number} && return [(path, x)]
    x isa Number && return [(path, [x])]
    x isa Optimisers.Leaf && return leaves(x.state, path*".state")
    ch = Functors.children(x)
    isempty(ch) && return []
    reduce(vcat, (leaves(v, path*"."*string(k)) for (k,v) in pairs(ch)); init=[])
end
function compare(a, c, label)
    aa, cc = leaves(a), leaves(c)
    @assert first.(aa) == first.(cc)
    for ((path,x),(_,y)) in zip(aa,cc)
        @assert size(x) == size(y)
        @assert all(isfinite,x) && all(isfinite,y)
        err = maximum(abs.(x .- y); init=0.0)
        rel = maximum(abs.(x .- y) ./ max.(abs.(y),eps(Float64)); init=0.0)
        @assert all(abs.(x .- y) .<= 1e-10 .+ 1e-9 .* abs.(y)) "$label $path $err"
        push!(records, (label=label,path=path,error=err,relative_error=rel,actual=collect(x),reference=collect(y)))
    end
end
model() = (first=(weight=reshape([1.,2.,3.,4.]./16,2,2),bias=[1.,-1.]./16),last=(weight=reshape([2.,-1.]./8,1,2),bias=[1.]/32))
predict(m,x) = m.last.weight * tanh.(m.first.weight*x .+ m.first.bias) .+ m.last.bias
loss(m,ids) = sum(abs2, predict(m, vcat(reshape(ids./16,1,:),reshape((ids.+1)./32,1,:))) .- reshape(ids./8,1,:))/length(ids)
# Check cyclic padding including rejection before training.
for n in unique([0,1,w-1,w,w+1,10])
    if n == 0
        rejected = try DU.DistributedDataContainer(b, Int[]); false catch e; e isa ArgumentError end
        @assert rejected
    else
        d = DU.DistributedDataContainer(b,collect(1:n))
        expected = [mod1(i,n) for i in r*cld(n,w)+1:(r+1)*cld(n,w)]
        @assert d.idxs == expected
    end
end
trace = []
for rule in (Optimisers.Descent(0.01), Optimisers.Adam(0.01))
    m = Functors.fmap(x -> x .+ r/16, model())
    m = DU.synchronize!!(b, DU.FluxDistributedModel(m))
    ref = model()
    st = Optimisers.setup(DU.DistributedOptimizer(b,rule),m)
    sr = Optimisers.setup(rule,ref)
    # Populate state with rank-distinct local updates, then broadcast root state/model.
    warm = Flux.gradient(x -> loss(x,[1,2].+r),m)[1]
    localstate = Optimisers.setup(rule,m)
    localstate,m = Optimisers.update!(localstate,m,warm)
    function copy_state!(dst,src)
        if dst isa Optimisers.Leaf
            dst.state = deepcopy(src.state)
        else
            foreach(copy_state!, values(Functors.children(dst)), values(Functors.children(src)))
        end
    end
    copy_state!(st,localstate)
    st = DU.synchronize!!(b,st)
    m = DU.synchronize!!(b,DU.FluxDistributedModel(m))
    sr,ref = Optimisers.update!(sr,ref,Flux.gradient(x -> loss(x,[1,2]),ref)[1])
    compare(st,sr,"populated state")
    compare(m,ref,"populated model")
    n = 3w-1
    shard = DU.DistributedDataContainer(b,collect(1:n)).idxs
    for step in 1:6
        ix = isodd(step) ? (1:2) : (3:3)
        ids = shard[ix]
        globalids = [mod1(rank*3+j,n) for rank in 0:w-1 for j in ix]
        g = Flux.gradient(x -> loss(x,ids),m)[1]
        compare(g,Flux.gradient(x -> loss(x,ids),ref)[1],"local gradient $step")
        expected = Flux.gradient(x -> loss(x,globalids),ref)[1]
        avg = deepcopy(g)
        for (path,x) in leaves(avg)
            push!(trace,(step=step,phase="diagnostic",path=path,dtype=string(eltype(x)),count=length(x),device="cpu",collective="avg"))
            DU.allreduce!(b,x,DU.avg)
        end
        compare(avg,expected,"mean gradient $step")
        # Missing /W must be distinguishable with these rank-distinct inputs.
        @assert any(!isapprox(w*x,y;atol=1e-10,rtol=1e-9) for ((_,x),(_,y)) in zip(leaves(avg),leaves(expected)))
        st,m = Optimisers.update!(st,m,g) # Native optimizer averages once.
        sr,ref = Optimisers.update!(sr,ref,expected)
        compare(m,ref,"parameters $step")
        compare(st,sr,"state $step")
    end
end
allrecords = MPI.gather(records,MPI.COMM_WORLD;root=0)
traces = MPI.gather(trace,MPI.COMM_WORLD;root=0)
if r == 0
    @assert all(==(trace),traces)
    out = get(opts,"output","artifacts/logs/multinode/p0/acceptance-$w.json")
    mkpath(dirname(out))
    open(out,"w") do io
        JSON.print(io,Dict("metadata"=>meta,"comparisons"=>allrecords,"trace"=>trace,"world"=>w,"status"=>"PASS"))
    end
    println("BASE_DDP_ACCEPTANCE PASS world=$w")
end
MPI.Barrier(MPI.COMM_WORLD)
