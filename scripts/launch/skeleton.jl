using MPI
using Flux
using Flux: DistributedUtils

function main()
    DistributedUtils.initialize(DistributedUtils.MPIBackend)
    backend = DistributedUtils.get_distributed_backend(DistributedUtils.MPIBackend)
    
    rank = DistributedUtils.local_rank(backend)
    world = DistributedUtils.total_workers(backend)
    
    println("Hello from rank $(rank) of $(world)")
    
    # Simple synchronization test
    val = rank == 0 ? [42.0] : [0.0]
    DistributedUtils.bcast!(backend, val)
    
    println("Rank $(rank) received broadcast value: $(val[1])")
    
    MPI.Barrier(backend.comm)
    println("Rank $(rank) finished successfully.")
end

main()
