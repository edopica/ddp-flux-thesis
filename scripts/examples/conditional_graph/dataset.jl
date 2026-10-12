# Deterministic balanced two-class toy dataset for the Phase K diagnostic.
#
# Task A and task B use the same class geometry rotated by 90 degrees, so both
# heads see learnable structure and equal shapes. Class 0 samples are generated
# first in each pool; consumption order comes from the saved schedule
# permutation, not from this storage order.
module CGDataset

using Random, SHA, Serialization, Downloads

export TASKS, NCLASSES, TRAIN_PER_TASK, VAL_PER_TASK, NOISE, CENTERS,
    make_toy_dataset, dataset_digest

const TASKS = (:a, :b)
const NCLASSES = 2
const TRAIN_PER_TASK = 512   # 256 per class
const VAL_PER_TASK = 128     # 64 per class
const NOISE = 0.5
const CENTERS = Dict(
    :a => [-1.2 1.2; -1.2 1.2],  # class 0 (-1.2, -1.2), class 1 (1.2, 1.2)
    :b => [-1.2 1.2; 1.2 -1.2],  # class 0 (-1.2, 1.2), class 1 (1.2, -1.2)
)

function generate_pool(task::Symbol, n_per_class::Int, seed::Integer)
    centers = CENTERS[task]
    rng = MersenneTwister(seed)
    n = 2 * n_per_class
    X = Matrix{Float64}(undef, 2, n)
    Y = zeros(Float64, NCLASSES, n)
    for label in 0:1
        for j in 1:n_per_class
            i = label * n_per_class + j
            X[:, i] = centers[:, label + 1] .+ NOISE .* randn(rng, 2)
            Y[label + 1, i] = 1.0
        end
    end
    return (; x = X, y = Y)
end

function dataset_digest(pools)
    io = IOBuffer()
    for task in TASKS, split in (:train, :val)
        serialize(io, pools[task][split].x)
        serialize(io, pools[task][split].y)
        hasproperty(pools[task][split], :indices) && serialize(io, pools[task][split].indices)
    end
    return bytes2hex(SHA.sha256(seekstart(io)))
end

# Official gzip IDX files, decoded directly (loader version 1). Each image's
# 784 bytes retain IDX row-major pixel order. gzip validates its CRC on decode.
const IDX_SOURCES = (
    (:a, "mnist", "https://storage.googleapis.com/cvdf-datasets/mnist/"),
    (:b, "fashion-mnist", "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"),
)

function read_idx(path, expected_dims)
    bytes = read(`gzip -dc $path`)
    io = IOBuffer(bytes)
    read(io, UInt8) == 0 && read(io, UInt8) == 0 || error("invalid IDX header: $path")
    read(io, UInt8) == 0x08 || error("IDX must contain unsigned bytes: $path")
    nd = Int(read(io, UInt8))
    nd == length(expected_dims) || error("invalid IDX dimensions: $path")
    dims = [Int(ntoh(read(io, UInt32))) for _ in 1:nd]
    dims == collect(expected_dims) || error("unexpected IDX shape $dims: $path")
    data = read(io)
    length(data) == prod(dims) || error("invalid IDX payload length: $path")
    return data
end

function balanced_subset(images, labels, per_class, seed)
    rng = MersenneTwister(seed)
    indices = Int[]
    for label in 0:9
        candidates = findall(==(label), labels)
        length(candidates) >= per_class || error("insufficient samples for class $label")
        append!(indices, shuffle(rng, candidates)[1:per_class])
    end
    # Interleave classes in storage, including the prediction panel's first 20.
    indices = vec(permutedims(reshape(indices, per_class, 10)))
    x = Float32.(reshape(images, 784, :)[:, indices]) ./ 255f0
    y = zeros(Float32, 10, length(indices))
    for (j, i) in enumerate(indices)
        y[Int(labels[i]) + 1, j] = 1f0
    end
    return (; x, y, indices, labels = Int.(labels[indices]))
end

function make_mnist_fashion_dataset(cache; train_per_task=5120, val_per_task=1000,
        seed_subset=505)
    train_per_task > 0 && train_per_task % 10 == 0 || error("train count must divide by ten")
    val_per_task > 0 && val_per_task % 10 == 0 || error("validation count must divide by ten")
    mkpath(cache)
    pools = Dict{Symbol,NamedTuple}()
    sources = NamedTuple[]
    for (ti, (task, name, base)) in enumerate(IDX_SOURCES)
        splits = Dict{Symbol,NamedTuple}()
        for (si, (split, prefix, count, take)) in enumerate((
                (:train, "train", 60000, train_per_task), (:val, "t10k", 10000, val_per_task)))
            paths = String[]
            for suffix in ("images-idx3-ubyte.gz", "labels-idx1-ubyte.gz")
                filename = "$prefix-$suffix"
                path = joinpath(cache, "$name-$filename")
                url = base * filename
                if !isfile(path)
                    tmp = path * ".download"
                    Downloads.download(url, tmp)
                    mv(tmp, path)
                end
                push!(paths, path)
                push!(sources, (; task, dataset=name, split, url, filename,
                    sha256=bytes2hex(SHA.sha256(read(path)))))
            end
            images = read_idx(paths[1], (count, 28, 28))
            labels = read_idx(paths[2], (count,))
            all(l -> 0 <= l <= 9, labels) || error("IDX labels outside 0:9")
            splits[split] = balanced_subset(images, labels, take ÷ 10,
                seed_subset + 10ti + si)
        end
        pools[task] = (; train=splits[:train], val=splits[:val])
    end
    return (; name=Symbol("mnist-fashion"), pools, tasks=TASKS, nclasses=10,
        train_per_task, val_per_task, sources, seed_subset,
        loader="official-idx-gzip-v1; Julia Downloads; external gzip",
        preprocessing="IDX row-major flatten; Float32(byte)/255; no augmentation",
        label_encoding="0:9; onehot row=label+1",
        task_mapping=(a="MNIST digits", b="Fashion-MNIST clothing"),
        digest=dataset_digest(pools))
end

function write_provenance(out, dataset)
    hasproperty(dataset, :sources) || return
    open(joinpath(out, "sources.tsv"), "w") do io
        println(io, "task\tdataset\tsplit\turl\tfilename\tsha256")
        for s in dataset.sources
            println(io, join(values(s), '\t'))
        end
    end
    open(joinpath(out, "subset_indices.tsv"), "w") do io
        println(io, "task\tsplit\tpool_index\tofficial_index\tlabel")
        for task in TASKS, split in (:train, :val)
            pool = dataset.pools[task][split]
            for i in eachindex(pool.indices)
                println(io, join((task, split, i, pool.indices[i], pool.labels[i]), '\t'))
            end
        end
    end
end

function make_toy_dataset(; seed_train::Integer=101, seed_val::Integer=202,
        train_per_task::Int=TRAIN_PER_TASK, val_per_task::Int=VAL_PER_TASK)
    pools = Dict{Symbol,NamedTuple}()
    for (i, task) in enumerate(TASKS)
        pools[task] = (;
            train = generate_pool(task, train_per_task ÷ 2, seed_train + i),
            val = generate_pool(task, val_per_task ÷ 2, seed_val + i),
        )
    end
    return (;
        name = :toy,
        pools,
        tasks = TASKS,
        nclasses = NCLASSES,
        centers = CENTERS,
        noise = NOISE,
        train_per_task,
        val_per_task,
        seeds = (; train = seed_train, val = seed_val),
        digest = dataset_digest(pools),
    )
end

end # module
