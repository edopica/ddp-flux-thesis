using MPI
using CUDA
using NCCL

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

CUDA.device!(rank % CUDA.ndevices())
println("Rank $rank using device $(CUDA.device()) (Total visible: $(CUDA.ndevices()))")

nccl_id = rank == 0 ? NCCL.UniqueID() : NCCL.LibNCCL.ncclUniqueId(ntuple(i->Int8(0), 128))

id_array = zeros(Int8, 128)
if rank == 0
    for i in 1:128
        id_array[i] = nccl_id.internal[i]
    end
end
MPI.Bcast!(id_array, 0, comm)
if rank != 0
    nccl_id = NCCL.LibNCCL.ncclUniqueId(ntuple(i->id_array[i], 128))
end

nccl_comm = NCCL.Communicator(size, rank; unique_id=nccl_id)
println("Rank $rank initialized NCCL successfully!")
