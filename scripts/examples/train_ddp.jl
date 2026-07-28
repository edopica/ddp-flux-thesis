#!/usr/bin/env julia

# ---------------------------------------------------------------------------
# End-to-End Distributed Data Parallel (DDP) Training Example
#
# This script demonstrates how to train a model across multiple processes
# (and optionally GPUs) using Flux's DistributedUtils.
# 
# Features covered:
# - Conditional backend initialization (NCCL for GPU, MPI for CPU)
# - Sharding data with DistributedDataContainer
# - Synchronizing model parameters and optimizer state
# - Distributed gradient averaging
# - Handling conditional execution graphs (unused parameters)
# ---------------------------------------------------------------------------

using Pkg
# Ensure we have the required packages loaded
using MPI
using Flux
using Flux: DistributedUtils
using Optimisers
using MLUtils
using Zygote
using Random
using Printf
using JLD2

# Conditionally load CUDA/NCCL if they are installed in the environment
const USE_CUDA = try
    using CUDA, NCCL, cuDNN
    CUDA.functional()
catch
    false
end

# ---------------------------------------------------------------------------
# 1. Model Definition (Conditional Graph)
# ---------------------------------------------------------------------------
# We use a Two-Head Model to demonstrate a conditional graph where different
# ranks might use different parts of the network, resulting in some parameters
# being unused on certain ranks during a forward/backward pass.

struct TwoHeadModel
    backbone
    head_a
    head_b
end
Flux.@functor TwoHeadModel

function (m::TwoHeadModel)(x, use_head_a::Bool)
    features = m.backbone(x)
    if use_head_a
        return m.head_a(features)
    else
        return m.head_b(features)
    end
end

# ---------------------------------------------------------------------------
# 2. Main Training Function
# ---------------------------------------------------------------------------

function main()
    # A. Initialize Backend
    if USE_CUDA
        DistributedUtils.initialize(DistributedUtils.NCCLBackend)
        backend = DistributedUtils.get_distributed_backend(DistributedUtils.NCCLBackend)
    else
        DistributedUtils.initialize(DistributedUtils.MPIBackend)
        backend = DistributedUtils.get_distributed_backend(DistributedUtils.MPIBackend)
    end

    rank = DistributedUtils.local_rank(backend)
    world = DistributedUtils.total_workers(backend)
    
    if rank == 0
        println("="^60)
        println("DDP Training Example")
        println("Backend: $(typeof(backend))")
        println("World Size: $world")
        println("="^60)
    end

    # B. Data Preparation
    # Generate some synthetic validation data
    Random.seed!(42)
    val_dataset = (rand(Float32, 10, 200), rand(Float32, 2, 200))
    ddp_val_data = DistributedUtils.DistributedDataContainer(backend, val_dataset)
    val_data_loader = DataLoader(ddp_val_data, batchsize=32, shuffle=false)

    # Generate some synthetic training data
    Random.seed!(42) # Use a fixed global seed so all ranks generate the identical global dataset before sharding
    global_X = rand(Float32, 10, 1000)
    global_Y = rand(Float32, 2, 1000)
    dataset = (global_X, global_Y)
    
    # Wrap in DistributedDataContainer to shard the dataset across ranks
    ddp_data = DistributedUtils.DistributedDataContainer(backend, dataset)
    
    Random.seed!(42 + rank)
    
    # Use MLUtils DataLoader on the sharded data
    data_loader = DataLoader(ddp_data, batchsize=32, shuffle=true)

    # C. Model Setup
    Random.seed!(1234) # Ensure identical initialization before sync (optional but good practice)
    model = TwoHeadModel(
        Dense(10 => 32, relu), # Backbone
        Dense(32 => 2),        # Head A
        Dense(32 => 2)         # Head B
    )
    
    if USE_CUDA
        model = fmap(gpu, model)
    end

    # Synchronize initial parameters from Rank 0 to all other ranks
    ddp_model = DistributedUtils.FluxDistributedModel(model)
    model = DistributedUtils.synchronize!!(backend, ddp_model; root=0)

    # D. Optimizer Setup
    # Wrap the optimizer rule with DistributedOptimizer to average gradients automatically
    dist_opt = DistributedUtils.DistributedOptimizer(backend, Optimisers.Adam(0.01))
    
    # Setup optimizer state
    opt_state = Optimisers.setup(dist_opt, model)
    
    # Synchronize initial optimizer state from Rank 0 to all other ranks
    opt_state = DistributedUtils.synchronize!!(backend, opt_state; root=0)

    # E. Training Loop
    epochs = 5
    for epoch in 1:epochs
        total_loss = 0.0f0
        batches = 0
        
        for (x, y) in data_loader
            if USE_CUDA
                x, y = gpu(x), gpu(y)
            end
            
            # Rank 0 uses head_a, everyone else uses head_b.
            # This creates a conditional graph where some parameters are unused by some ranks.
            use_head_a = (rank == 0)
            
            l, gs = Zygote.withgradient(model) do m
                y_hat = m(x, use_head_a)
                Flux.Losses.mse(y_hat, y)
            end
            gs = gs[1]
            
            # Since our graph is conditional, we must resolve unused parameters
            # (which Zygote returns as `nothing`) by replacing them with zero-tensors
            # of the correct shape and type before the distributed allreduce happens.
            gs = DistributedUtils.resolve_unused_parameters!!(backend, gs, model)
            
            # DistributedOptimizer automatically averages the gradients across all ranks here
            opt_state, model = Optimisers.update(opt_state, model, gs)
            
            total_loss += l
            batches += 1
        end
        
        # Calculate local average loss
        avg_loss = total_loss / batches
        
        # Compute global average loss using an allreduce!
        # allreduce! expects an array, so we wrap our scalar in an array
        global_loss = DistributedUtils.allreduce!(backend, [avg_loss], +)[1] / world
        
        if rank == 0
            @printf("Epoch %d/%d | Global Train Loss: %.4f\n", epoch, epochs, global_loss)
        end
        
        # Validation Loop
        val_total_loss = 0.0f0
        val_batches = 0
        
        for (x, y) in val_data_loader
            if USE_CUDA
                x, y = gpu(x), gpu(y)
            end
            
            # Rank 0 uses head_a, everyone else uses head_b.
            use_head_a = (rank == 0)
            
            # Forward pass without tracking gradients
            y_hat = model(x, use_head_a)
            l = Flux.Losses.mse(y_hat, y)
            
            val_total_loss += l
            val_batches += 1
        end
        
        avg_val_loss = val_total_loss / val_batches
        global_val_loss = DistributedUtils.allreduce!(backend, [avg_val_loss], +)[1] / world
        
        if rank == 0
            @printf("Epoch %d/%d | Global Val Loss: %.4f\n", epoch, epochs, global_val_loss)
        end
    end
    
    if rank == 0
        println("Training complete. Saving model checkpoint...")
        model_state = Flux.state(model)
        jldsave("ddp_model_checkpoint.jld2"; model_state)
    end
end

main()
