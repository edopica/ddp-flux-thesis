using MPI
using Flux
using Flux: DistributedUtils

function main()
    println("--- Flux MPI Health Check ---")
    
    # Print environment info before initialization
    println("Checking PMI environment variables...")
    pmi_vars = filter(k -> occursin("PMI", k) || occursin("SLURM", k) || occursin("OMPI", k), keys(ENV))
    if isempty(pmi_vars)
        println("  No PMI/SLURM/OMPI environment variables found.")
    else
        for k in sort(collect(pmi_vars))
            println("  $k = $(ENV[k])")
        end
    end

    println("\nInitializing MPI Backend...")
    try
        DistributedUtils.initialize(DistributedUtils.MPIBackend)
    catch e
        println("❌ Initialization failed: ", e)
        exit(1)
    end
    
    println("\nMPI Library Version:")
    println("  $(MPI.Get_library_version())")

    backend = DistributedUtils.get_distributed_backend(DistributedUtils.MPIBackend)
    
    rank = DistributedUtils.local_rank(backend)
    world = DistributedUtils.total_workers(backend)
    
    println("\n--- Worker Info ---")
    println("Hello from rank $(rank) of $(world)")
    
    # Simple synchronization test
    val = rank == 0 ? [42.0] : [0.0]
    DistributedUtils.bcast!(backend, val)
    
    println("Rank $(rank) received broadcast value: $(val[1])")
    
    MPI.Barrier(backend.comm)
    
    if rank == 0
        println("\n✅ Health check passed: All ranks synchronized successfully.")
    end
end

main()
