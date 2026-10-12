# GENERATED from notebook/ddp_mnist_quickstart.ipynb (cells tagged "ddp-script").
# Edit the notebook cells and rerun the notebook to regenerate this file.

# 1. Backend: MPI for CPU processes, NCCL for CUDA devices. This example uses CPUs.
using MPI
using Flux
using Flux: DistributedUtils
using Optimisers
using MLUtils
using MLDatasets
using Random, Printf

DistributedUtils.initialize(DistributedUtils.MPIBackend)
backend = DistributedUtils.get_distributed_backend(DistributedUtils.MPIBackend)
rank = DistributedUtils.local_rank(backend)
world = DistributedUtils.total_workers(backend)

# 2. Network: a plain MLP. MLDatasets returns 28x28 images; flatten maps them to 784 inputs.
Random.seed!(1234)
model = Chain(
    Flux.flatten,
    Dense(784 => 128, relu),
    Dense(128 => 10),
)

# Every rank must start from identical parameters: rank 0 broadcasts its copy.
model = DistributedUtils.synchronize!!(
    backend, DistributedUtils.FluxDistributedModel(model); root = 0)

# 3. Data: a deterministic MNIST subset, sharded across ranks.
const TRAIN_N = 12000
const TEST_N = 2000
const BATCH_SIZE = 32
const EPOCHS = 3

function load_split(split, n)
    d = MNIST(split)
    x = d.features[:, :, 1:n]
    y = Flux.onehotbatch(d.targets[1:n], 0:9)
    return x, y
end

x_train, y_train = load_split(:train, TRAIN_N)
x_test, y_test = load_split(:test, TEST_N)

# DistributedDataContainer gives every rank its own slice of the same dataset.
train_data = DistributedUtils.DistributedDataContainer(backend, (x_train, y_train))
test_data = DistributedUtils.DistributedDataContainer(backend, (x_test, y_test))

train_loader = DataLoader(train_data, batchsize = BATCH_SIZE, shuffle = true)
test_loader = DataLoader(test_data, batchsize = BATCH_SIZE, shuffle = false)

# 4. Optimizer: wrap Adam in DistributedOptimizer. Flux.setup returns a
# DistributedOptimizerState that averages gradients once per update, outside AD.
opt_rule = DistributedUtils.DistributedOptimizer(backend, Optimisers.Adam(1.0f-3))
opt_state = Flux.setup(opt_rule, model)

# 5. Evaluation: loss and accuracy over the local shard, summed across ranks.
function evaluate(model, loader, backend)
    correct = 0
    total = 0
    loss_sum = 0.0
    for (x, y) in loader
        y_hat = model(x)
        loss_sum += sum(Flux.Losses.logitcrossentropy(y_hat, y; agg = identity))
        correct += sum(Flux.onecold(y_hat) .== Flux.onecold(y))
        total += size(y, 2)
    end
    buf = Float64[loss_sum, correct, total]
    DistributedUtils.allreduce!(backend, buf, +)
    return (loss = buf[1] / buf[3], accuracy = buf[2] / buf[3])
end

# 5b. Training loop: Flux.train! updates through the distributed wrapper state.
# Rank 0 alone writes the shared metrics file.
metrics_path = get(ENV, "DDP_QUICKSTART_METRICS", joinpath(pwd(), "ddp_mnist_metrics.tsv"))

if rank == 0
    open(metrics_path, "w") do io
        println(io, "epoch\ttest_loss\ttest_accuracy\telapsed_s")
    end
end

for epoch in 1:EPOCHS
    started = time()
    Flux.train!(model, train_loader, opt_state) do m, x, y
        Flux.Losses.logitcrossentropy(m(x), y)
    end
    m = evaluate(model, test_loader, backend)
    elapsed = time() - started
    if rank == 0
        @printf("epoch %d/%d  test_loss=%.4f  test_accuracy=%.4f  (%.1f s)\n",
            epoch, EPOCHS, m.loss, m.accuracy, elapsed)
        open(metrics_path, "a") do io
            @printf(io, "%d\t%.6f\t%.6f\t%.1f\n", epoch, m.loss, m.accuracy, elapsed)
        end
    end
end
