using MPI
using Flux
using Flux: DistributedUtils
using MLUtils

function main()
    DistributedUtils.initialize(DistributedUtils.MPIBackend)
    backend = DistributedUtils.get_distributed_backend(DistributedUtils.MPIBackend)
    
    rank = DistributedUtils.local_rank(backend)
    world_size = DistributedUtils.total_workers(backend)
    
    println("Rank $rank: initializing with world size $world_size")
    
    # Dataset size N=9
    N = 9
    data = 1:N
    
    try
        println("Rank $rank: constructing DistributedDataContainer")
        ddc = DistributedUtils.DistributedDataContainer(backend, data)
        println("Rank $rank: successfully created DDC with $(length(ddc)) items.")
        println("Rank $rank: items = $([ddc[i] for i in 1:length(ddc)])")
    catch e
        println("Rank $rank: caught exception:")
        showerror(stdout, e)
        println()
    end
end

main()