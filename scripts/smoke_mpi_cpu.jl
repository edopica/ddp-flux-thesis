using MPI

MPI.Init()

try
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    world = MPI.Comm_size(comm)

    local_value = rank + 1
    total = MPI.Allreduce(local_value, +, comm)
    expected = div(world * (world + 1), 2)

    @assert total == expected "MPI allreduce failed on rank $(rank): got $(total), expected $(expected)"

    println("rank=$(rank) world=$(world) allreduce_sum=$(total)")
    MPI.Barrier(comm)
finally
    MPI.Finalize()
end
