"""
    ReferenceLoop

Deterministic, single-process reference training loop for Flux.jl DDP thesis.

This module provides a modular, deterministic training loop that produces
bit-identical outputs on every run (same seed, same code). It serves as
the correctness baseline against which all future DDP implementations are compared.

**Constraints:**
- CPU-only, no MPI, no GPU
- Float32 arithmetic throughout
- Single-threaded execution
- `Random.seed!()` at entry for determinism
- Uses Zygote for gradients, Optimisers.jl for optimizer setup/updates
- Does NOT use Flux's built-in `train!`
"""
module ReferenceLoop

using Random
using Flux
using Zygote
using Optimisers
using Functors
using JLD2

export ReferenceConfig, run_reference_loop, save_baseline, load_baseline

# ---------------------------------------------------------------------------
# Default callables
# ---------------------------------------------------------------------------

"""
    default_model_fn()

Returns the default model: a 2-layer MLP matching the Flux docs GPU example.
Chain(Dense(1 => 256, tanh), Dense(256 => 1)) — approximately 769 parameters.
"""
function default_model_fn()
    Chain(Dense(1 => 256, tanh), Dense(256 => 1))
end

"""
    default_data_fn()

Returns synthetic cubic data: 16 points on y = x³, x ∈ [-1, 1].
Both x and y are Float32, shaped (1, 16).
"""
function default_data_fn()
    x = reshape(Float32.(range(-1.0f0, 1.0f0, length=16)), 1, :)
    y = x .^ 3
    return (x, y)
end

"""
    default_loss_fn(model, x, y)

Mean squared error loss.
"""
function default_loss_fn(model, x, y)
    Flux.mse(model(x), y)
end

"""
    default_setup_fn(model)

Sets up the optimizer state using Descent(0.01f0) via Optimisers.jl.
"""
function default_setup_fn(model)
    Optimisers.setup(Descent(0.01f0), model)
end

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

"""
    ReferenceConfig

Configuration for a deterministic reference training loop.

# Fields
- `seed::Int`: Random seed for determinism (default: 42)
- `model_fn`: Callable returning a Flux model (default: 2-layer MLP)
- `data_fn`: Callable returning `(x, y)` tuple (default: synthetic cubic data)
- `loss_fn`: Callable `(model, x, y) -> scalar` (default: MSE)
- `setup_fn`: Callable `(model) -> optimizer state` (default: Adam(0.001f0))
- `num_steps::Int`: Number of training steps (default: 20)
- `save_loss::Bool`: Record loss at each step (default: true)
- `save_gradients::Bool`: Record gradients at each step (default: true)
- `save_parameters::Bool`: Record parameters at each step (default: true)
- `save_optimizer_state::Bool`: Record optimizer state at each step (default: true)
- `save_step_time::Bool`: Record wall time per step (default: false)
"""
Base.@kwdef struct ReferenceConfig
    seed::Int = 42
    model_fn::Any = default_model_fn
    data_fn::Any = default_data_fn
    loss_fn::Any = default_loss_fn
    setup_fn::Any = default_setup_fn
    num_steps::Int = 20

    save_loss::Bool = true
    save_gradients::Bool = true
    save_parameters::Bool = true
    save_optimizer_state::Bool = true
    save_step_time::Bool = false
end

# ---------------------------------------------------------------------------
# Deep-copy helpers (for snapshotting mutable state)
# ---------------------------------------------------------------------------

"""
    snapshot_params(model)

Returns a deep copy of all parameter arrays in the model as a flat vector of Arrays.
Uses Functors.fmap to traverse the model tree.
"""
function snapshot_params(model)
    params = []
    Functors.fmap(model; exclude=x -> x isa AbstractArray) do x
        push!(params, copy(x))
        x
    end
    return params
end

"""
    snapshot_grads(grad)

Returns a deep copy of all gradient arrays as a flat vector of Arrays.
"""
function snapshot_grads(grad)
    grads = []
    Functors.fmap(grad; exclude=x -> x isa AbstractArray) do x
        push!(grads, copy(x))
        x
    end
    return grads
end

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

"""
    run_reference_loop(config::ReferenceConfig)

Runs a deterministic single-process training loop for `config.num_steps` steps.

Returns a `NamedTuple` with:
- `config`: the input configuration (for provenance)
- `initial_model`: deep copy of the model before training
- `initial_state`: deep copy of the optimizer state before training
- `initial_gradients`: gradients from step 1
- `steps`: vector of per-step records
- `final_model`: model after training
- `final_state`: optimizer state after training
- `elapsed_total`: total wall time in seconds
"""
function run_reference_loop(config::ReferenceConfig)
    # Guarantee determinism
    Random.seed!(config.seed)

    # Build model and data
    model = config.model_fn()
    (x, y) = config.data_fn()

    # Setup optimizer
    st_opt = config.setup_fn(model)

    # Snapshot initial state
    initial_model = deepcopy(model)
    initial_state = deepcopy(st_opt)

    # Storage for per-step records
    steps = Vector{NamedTuple}(undef, config.num_steps)
    initial_gradients = nothing

    t_total_start = time()

    for step in 1:config.num_steps
        t_step_start = time()

        # Forward + backward pass (Flux docs pattern)
        l, grad = Zygote.withgradient(config.loss_fn, model, x, y)

        # Capture initial gradients (step 1)
        if step == 1
            initial_gradients = deepcopy(grad[1])
        end

        # Optimizer update
        st_opt, model = Optimisers.update(st_opt, model, grad[1])

        t_step_end = time()

        # Build per-step record
        record = (;
            loss = config.save_loss ? l : nothing,
            gradients = config.save_gradients ? deepcopy(grad[1]) : nothing,
            parameters = config.save_parameters ? deepcopy(model) : nothing,
            optimizer_state = config.save_optimizer_state ? deepcopy(st_opt) : nothing,
            step_time = config.save_step_time ? (t_step_end - t_step_start) : nothing,
        )
        steps[step] = record
    end

    t_total_end = time()

    return (
        config = config,
        initial_model = initial_model,
        initial_state = initial_state,
        initial_gradients = initial_gradients,
        steps = steps,
        final_model = model,
        final_state = st_opt,
        elapsed_total = t_total_end - t_total_start,
    )
end

# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

"""
    save_baseline(path::String, result)

Saves a reference loop result to a JLD2 file at `path`.
Creates parent directories if they don't exist.
"""
function save_baseline(path::String, result)
    mkpath(dirname(path))
    jldsave(path;
        config_seed = result.config.seed,
        config_num_steps = result.config.num_steps,
        initial_model = result.initial_model,
        initial_state = result.initial_state,
        initial_gradients = result.initial_gradients,
        steps = result.steps,
        final_model = result.final_model,
        final_state = result.final_state,
        elapsed_total = result.elapsed_total,
    )
end

"""
    load_baseline(path::String)

Loads a reference loop result from a JLD2 file at `path`.
Returns a NamedTuple matching the output of `run_reference_loop`.

Note: The `config` field is reconstructed with default callables since
functions cannot be serialized. The seed and num_steps are restored.
"""
function load_baseline(path::String)
    data = load(path)
    config = ReferenceConfig(
        seed = data["config_seed"],
        num_steps = data["config_num_steps"],
    )
    return (
        config = config,
        initial_model = data["initial_model"],
        initial_state = data["initial_state"],
        initial_gradients = data["initial_gradients"],
        steps = data["steps"],
        final_model = data["final_model"],
        final_state = data["final_state"],
        elapsed_total = data["elapsed_total"],
    )
end

end # module ReferenceLoop
